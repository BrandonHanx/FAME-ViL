# Copyright (c) Facebook, Inc. and its affiliates.

import os
from copy import deepcopy
from typing import Dict

import torch
import torch.nn.functional as F
from mmf.models.composition import NormalizationLayer
from mmf.models.fashionvil.base import FashionViLBaseModel
from mmf.modules.losses import (
    ContrastiveLoss,
    CrossEntropyLoss,
    MSELoss,
    SupervisedContrastiveLoss,
    SoftLabelCrossEntropyLoss,
)
from mmf.modules.ot import optimal_transport_dist
from mmf.utils.build import build_image_encoder
from mmf.utils.configuration import get_mmf_cache_dir
from torch import Tensor, nn
from transformers.modeling_bert import (
    BertOnlyNSPHead,
    BertForPreTraining,
    BertPredictionHeadTransform,
)


class CosSim(nn.Module):
    def __init__(self, nfeat, nclass):
        super().__init__()
        self.nfeat = nfeat
        self.nclass = nclass
        self.projection = nn.Parameter(torch.randn(nfeat, nclass), requires_grad=True)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        projection_norm = F.normalize(self.projection, p=2, dim=0)
        logits = torch.matmul(x, projection_norm)
        return logits


class FashionViLForPretraining(FashionViLBaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.task_for_inference = config.task_for_inference
        self.tasks = config.tasks
        self.double_view = config.get("double_view", False)
        self.no_sharing = config.get("no_sharing", False)
        if self.no_sharing:
            self.bert_2 = deepcopy(self.bert)

        self.contrastive_norm = NormalizationLayer()
        self.heads = nn.ModuleDict()
        self.loss_funcs = nn.ModuleDict()

        if "mpfc" in config.tasks and config.bypass_transformer:
            self.image_tokenizer = build_image_encoder(
                config.image_tokenizer,
                config.direct_features_input,
            )
            self.image_tokenizer = self.image_tokenizer.eval()
            for param in self.image_tokenizer.parameters():
                param.requires_grad = False

        self.init_heads()
        self.init_losses()

    def get_image_embedding(self, *args, **kwargs):
        if self.no_sharing:
            return self.bert_2.get_image_embedding(*args, **kwargs)
        else:
            return self.bert.get_image_embedding(*args, **kwargs)

    def get_text_embedding(self, *args, **kwargs):
        if self.no_sharing:
            return self.bert_2.get_text_embedding(*args, **kwargs)
        else:
            return self.bert.get_text_embedding(*args, **kwargs)

    def get_joint_embedding(self, *args, **kwargs):
        return self.bert.get_joint_embedding(*args, **kwargs)

    def init_heads(self):
        if "itm" in self.tasks:
            self.heads["itm"] = BertOnlyNSPHead(self.bert.config)
        if "mlm" in self.tasks:
            bert_masked_lm = BertForPreTraining.from_pretrained(
                self.config.bert_model_name,
                config=self.bert.config,
                cache_dir=os.path.join(
                    get_mmf_cache_dir(), "distributed_{}".format(-1)
                ),
            )
            self.heads["mlm"] = deepcopy(bert_masked_lm.cls.predictions)
            self.bert._tie_or_clone_weights(
                self.heads["mlm"].decoder, self.bert.embeddings.word_embeddings
            )
        if "mpfr" in self.tasks:
            self.heads["mpfr"] = nn.Sequential(
                BertPredictionHeadTransform(self.bert.config),
                nn.Linear(
                    self.bert.config.hidden_size,
                    self.config.visual_embedding_dim,
                    bias=False,
                ),
            )
            self.bert._tie_or_clone_weights(
                self.heads["mpfr"][1], self.bert.embeddings.projection
            )
            self.heads["mpfr"][1].weight = nn.Parameter(
                self.heads["mpfr"][1].weight.t()
            )
            self.heads["mpfr"][1].bias = nn.Parameter(
                torch.zeros(self.config.visual_embedding_dim)
            )
        if "mpfc" in self.tasks:
            self.heads["mpfc"] = nn.Sequential(
                BertPredictionHeadTransform(self.bert.config),
                nn.Linear(
                    self.bert.config.hidden_size,
                    1024,
                ),
            )
        if "pac" in self.tasks:
            self.heads["pac"] = CosSim(self.bert.config.hidden_size, 2232)

    def init_losses(self):
        self.loss_funcs["itc"] = ContrastiveLoss()
        if "itm" in self.tasks:
            self.loss_funcs["itm"] = CrossEntropyLoss()
        if "mlm" in self.tasks:
            self.loss_funcs["mlm"] = CrossEntropyLoss(ignore_index=-1)
        if "mpfr" in self.tasks:
            self.loss_funcs["mpfr"] = MSELoss()
        if "mpfc" in self.tasks:
            self.loss_funcs["mpfc"] = CrossEntropyLoss()
        if "mvc" in self.tasks:
            self.loss_funcs["mvc"] = SupervisedContrastiveLoss()
        if "pac" in self.tasks:
            self.loss_funcs["pac"] = SoftLabelCrossEntropyLoss()
        if "icc" in self.tasks:
            self.loss_funcs["icc"] = ContrastiveLoss()

    @torch.no_grad()
    def get_patch_labels(self, image, chunk_size=8):
        batch_size = image.shape[0]
        assert batch_size % chunk_size == 0
        # We need to override eval() as this image_tokenizer is a submodule
        self.image_tokenizer = self.image_tokenizer.eval()
        indices = []
        for i in range(batch_size // chunk_size):
            _, _, idx = self.image_tokenizer(
                image[i * chunk_size : (i + 1) * chunk_size]
            )
            indices.append(idx)
        indices = torch.cat(indices, dim=0)
        return indices.long()

    @torch.no_grad()
    def get_hard_pairs(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # Hard negative pairs mining
        # FIXME: not support multi-gpu mining
        if self.training:
            reset_train = True
            self.eval()
        else:
            reset_train = False

        itc_dict = self._forward_itc(sample_list)
        image_embeddings = itc_dict["scores"]
        text_embeddings = itc_dict["targets"]
        correlations = image_embeddings @ text_embeddings.t()
        batch_size = correlations.shape[0]
        # under double_view mode we have more than one positives
        diag = torch.eye(batch_size).bool()
        if self.double_view:
            bs = batch_size // 2
            diag[:bs, bs:] = diag[:bs, :bs]
            diag[bs:, :bs] = diag[:bs, :bs]
        correlations[diag] = -1
        # FIXME: more complicated sampling strategy
        hard_text_index = torch.argmax(correlations, dim=1)
        combine_index = torch.arange(batch_size).to(image_embeddings.device)
        combine_index_index = torch.rand(batch_size) > 0.5
        combine_index[combine_index_index] = hard_text_index[combine_index_index]

        if reset_train:
            self.train()

        sample_list["input_ids"] = sample_list["input_ids"][combine_index]
        sample_list["segment_ids"] = sample_list["segment_ids"][combine_index]
        sample_list["input_mask"] = sample_list["input_mask"][combine_index]
        if "attention_mask" in sample_list.keys():
            sample_list["attention_mask"] = sample_list["attention_mask"][combine_index]
        sample_list["targets"][combine_index_index] = 0

        return sample_list

    def flatten_for_bert(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        to_be_flattened = ["input_ids", "segment_ids"]
        to_be_flattened_dim = ["image"]
        if sample_list["task"] == "mlm":
            to_be_flattened += ["lm_label_ids", "input_ids_masked"]
        elif sample_list["task"] == "mpfr":
            to_be_flattened += ["image_masks"]
        elif sample_list["task"] == "mpfc":
            to_be_flattened += ["image_masks"]
            if "patch_labels" in sample_list.keys():
                to_be_flattened += ["patch_labels"]
        elif sample_list["task"] == "mvc":
            to_be_flattened_dim += ["image_0", "image_1"]
        flattened = self.flatten(sample_list, to_be_flattened, to_be_flattened_dim)
        return flattened

    def add_post_flatten_params(
        self, sample_list: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        b, l, _ = sample_list["image"].shape
        device = sample_list["image"].device
        sample_list["visual_embeddings_type"] = torch.zeros(
            (b, l), device=device
        ).long()

        if self.double_view and self.training and sample_list["task"] != "mvc":
            sample_list["input_ids"] = sample_list["input_ids"].repeat(2, 1)
            sample_list["segment_ids"] = sample_list["segment_ids"].repeat(2, 1)
            sample_list["input_mask"] = sample_list["input_mask"].repeat(2, 1)
            sample_list["targets"] = sample_list["targets"].repeat(2)
            if sample_list["task"] == "mlm":
                sample_list["input_ids_masked"] = sample_list[
                    "input_ids_masked"
                ].repeat(2, 1)
                sample_list["lm_label_ids"] = sample_list["lm_label_ids"].repeat(2, 1)

        if sample_list["task"] == "mvc":
            sample_list["visual_embeddings_type"] = torch.zeros(
                (b // 2, l), device=device
            ).long()
            sample_list["attention_mask"] = torch.cat(
                (
                    sample_list["input_mask"],
                    torch.ones((b // 2, l), device=device).long(),
                ),
                dim=-1,
            )

        if sample_list["task"] in ["itm", "mlm", "mpfr", "mpfc", "icc"]:
            sample_list["attention_mask"] = torch.cat(
                (
                    sample_list["input_mask"],
                    torch.ones((b, l), device=device).long(),
                ),
                dim=-1,
            )

        if sample_list["task"] in ["mpfr", "mpfc"]:
            if self.double_view:
                sample_list["image_masks"] = sample_list["image_masks"].repeat(2, 1)
            mask = sample_list["image_masks"] == 0
            mask = mask.float().unsqueeze(-1)
            sample_list["masked_image"] = sample_list["image"] * mask

        return sample_list

    def _forward_itc(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        visual_embeddings, _, _ = self.get_image_embedding(
            sample_list["image"], sample_list["visual_embeddings_type"]
        )
        visual_embeddings = visual_embeddings.mean(dim=1)
        visual_embeddings = self.contrastive_norm(visual_embeddings)

        text_embeddings, _, _ = self.get_text_embedding(
            sample_list["input_ids"],
            sample_list["segment_ids"],
            sample_list["input_mask"],
        )
        masks = sample_list["input_mask"]
        text_embeddings = text_embeddings * masks.unsqueeze(2)
        text_embeddings = torch.sum(text_embeddings, dim=1) / (
            torch.sum(masks, dim=1, keepdim=True)
        )
        text_embeddings = self.contrastive_norm(text_embeddings)

        output_dict = {
            "scores": visual_embeddings,
            "targets": text_embeddings,
        }

        loss = {}
        loss["itc_loss"] = self.loss_funcs["itc"](sample_list, output_dict)
        output_dict["losses"] = loss

        return output_dict

    def _forward_itm(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        sample_list = self.get_hard_pairs(sample_list)
        _, pooled_output, _ = self.get_joint_embedding(
            sample_list["input_ids"],
            sample_list["segment_ids"],
            sample_list["image"],
            sample_list["visual_embeddings_type"],
            sample_list["attention_mask"],
        )
        logits = self.heads["itm"](pooled_output)
        reshaped_logits = logits.contiguous().view(-1, 2)
        output_dict = {"scores": reshaped_logits}

        loss = {}
        loss["itm_loss"] = self.loss_funcs["itm"](sample_list, output_dict)
        output_dict["losses"] = loss

        return output_dict

    def _forward_mlm(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        sequence_output, _, _ = self.get_joint_embedding(
            sample_list["input_ids_masked"],
            sample_list["segment_ids"],
            sample_list["image"],
            sample_list["visual_embeddings_type"],
            sample_list["attention_mask"],
        )
        num_visual_tokens = sample_list["image"].shape[1]
        sequence_output = sequence_output[:, :-num_visual_tokens]
        logits = (
            self.heads["mlm"](sequence_output)
            .contiguous()
            .view(-1, self.bert.config.vocab_size)
        )
        labels = sample_list["lm_label_ids"].contiguous().view(-1)
        sample_list["targets"] = labels

        output_dict = {"scores": logits}

        loss = {}
        loss["mlm_loss"] = self.loss_funcs["mlm"](sample_list, output_dict)
        output_dict["losses"] = loss

        return output_dict

    def _forward_mpfr(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        hidden, _, _ = self.get_joint_embedding(
            sample_list["input_ids"],
            sample_list["segment_ids"],
            sample_list["masked_image"],
            sample_list["visual_embeddings_type"],
            sample_list["attention_mask"],
        )
        _, num_visual_tokens, visual_dim = sample_list["image"].shape

        mask = sample_list["image_masks"] == 1
        mask = mask.unsqueeze(-1)

        hidden = hidden[:, -num_visual_tokens:]
        hidden_masked = (
            hidden[mask.expand_as(hidden)].contiguous().view(-1, hidden.size(-1))
        )
        predict_feat = self.heads["mpfr"](hidden_masked)

        target = sample_list["image"]
        target_masked = target[mask.expand_as(target)].contiguous().view(-1, visual_dim)

        sample_list["targets"] = target_masked.detach()

        output_dict = {"scores": predict_feat}

        loss = {}
        loss["mpfr_loss"] = self.loss_funcs["mpfr"](sample_list, output_dict)
        output_dict["losses"] = loss

        return output_dict

    def _forward_mpfc(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        hidden, _, _ = self.get_joint_embedding(
            sample_list["input_ids"],
            sample_list["segment_ids"],
            sample_list["masked_image"],
            sample_list["visual_embeddings_type"],
            sample_list["attention_mask"],
        )
        _, num_visual_tokens, visual_dim = sample_list["image"].shape

        mask = sample_list["image_masks"] == 1

        hidden = hidden[:, -num_visual_tokens:]
        hidden_masked = (
            hidden[mask.unsqueeze(-1).expand_as(hidden)]
            .contiguous()
            .view(-1, hidden.size(-1))
        )
        logits = self.heads["mpfc"](hidden_masked)

        target = self.get_patch_labels(sample_list["original_image"])
        target_masked = target[mask].contiguous().view(-1)

        sample_list["targets"] = target_masked

        output_dict = {"scores": logits}

        loss = {}
        loss["mpfc_loss"] = self.loss_funcs["mpfc"](sample_list, output_dict)
        output_dict["losses"] = loss

        return output_dict

    def _forward_2wpa(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        sample_list = self.get_hard_pairs(sample_list)
        visual_embeddings, _, _ = self.get_image_embedding(
            sample_list["image"], sample_list["visual_embeddings_type"]
        )
        image_pad = torch.zeros_like(sample_list["visual_embeddings_type"]).bool()

        text_embeddings, _, _ = self.get_text_embedding(
            sample_list["input_ids"],
            sample_list["segment_ids"],
            sample_list["input_mask"],
        )
        text_pad = ~sample_list["input_mask"].bool()

        ot_dist = optimal_transport_dist(
            text_embeddings, visual_embeddings, text_pad, image_pad
        ).to(text_embeddings.device)

        itm_labels = sample_list["targets"]
        ot_pos = ot_dist.masked_select(itm_labels == 1)
        ot_neg = ot_dist.masked_select(itm_labels == 0)
        ot_loss = (ot_pos.sum() - ot_neg.sum()) / (ot_pos.size(0) + ot_neg.size(0))

        loss = {}
        loss["2wpa_loss"] = ot_loss
        output_dict = {}
        output_dict["losses"] = loss

        return output_dict

    def _forward_mvc(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        visual_embeddings, _, _ = self.get_image_embedding(
            sample_list["image_0"], sample_list["visual_embeddings_type"]
        )
        visual_embeddings = visual_embeddings.mean(dim=1)
        visual_embeddings = self.contrastive_norm(visual_embeddings)

        text_embeddings, _, _ = self.get_text_embedding(
            sample_list["input_ids"],
            sample_list["segment_ids"],
            sample_list["input_mask"],
        )
        masks = sample_list["input_mask"]
        text_embeddings = text_embeddings * masks.unsqueeze(2)
        text_embeddings = torch.sum(text_embeddings, dim=1) / (
            torch.sum(masks, dim=1, keepdim=True)
        )
        text_embeddings = self.contrastive_norm(text_embeddings)

        comp_embeddings, _, _ = self.get_joint_embedding(
            sample_list["input_ids"],
            sample_list["segment_ids"],
            sample_list["image_1"],
            sample_list["visual_embeddings_type"],
            sample_list["attention_mask"],
        )
        num_visual_tokens = sample_list["image_1"].shape[1]
        comp_embeddings = comp_embeddings[:, -num_visual_tokens:].mean(dim=1)
        comp_embeddings = self.contrastive_norm(comp_embeddings)

        output_dict = {
            "scores": torch.cat(
                (visual_embeddings, text_embeddings, comp_embeddings), dim=0
            ),
        }
        sample_list["targets"] = sample_list["ann_idx"].squeeze().repeat(3)

        loss = {}
        loss["mvc_loss"] = self.loss_funcs["mvc"](sample_list, output_dict)
        output_dict["losses"] = loss

        return output_dict

    def _forward_pac(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        visual_embeddings, _, _ = self.get_image_embedding(
            sample_list["image"], sample_list["visual_embeddings_type"]
        )
        visual_embeddings = visual_embeddings.mean(dim=1)

        text_embeddings, _, _ = self.get_text_embedding(
            sample_list["input_ids"],
            sample_list["segment_ids"],
            sample_list["input_mask"],
        )
        masks = sample_list["input_mask"]
        text_embeddings = text_embeddings * masks.unsqueeze(2)
        text_embeddings = torch.sum(text_embeddings, dim=1) / (
            torch.sum(masks, dim=1, keepdim=True)
        )

        visual_logits = self.heads["pac"](visual_embeddings)
        text_logits = self.heads["pac"](text_embeddings)
        sample_list.targets = sample_list.attribute_labels

        loss = {}
        output_dict = {"scores": visual_logits}
        loss["pac_loss"] = self.loss_funcs["pac"](sample_list, output_dict)
        output_dict = {"scores": text_logits}
        loss["pac_loss"] += self.loss_funcs["pac"](sample_list, output_dict)
        output_dict["losses"] = loss

        return output_dict

    def _forward_icc(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        visual_embeddings, _, _ = self.get_image_embedding(
            sample_list["image"], sample_list["visual_embeddings_type"]
        )
        visual_embeddings = visual_embeddings.mean(dim=1)
        visual_embeddings = self.contrastive_norm(visual_embeddings)

        num_visual_tokens = sample_list["dv_image"].shape[1]
        num_text_tokens = sample_list["input_ids"].shape[1]
        dropout_ratio = 0.25
        text_dropout_index = torch.randint(
            high=num_text_tokens, size=(int(dropout_ratio * num_text_tokens),)
        )
        pacth_dropout_index = (
            torch.randint(
                high=num_visual_tokens, size=(int(dropout_ratio * num_visual_tokens),)
            )
            + num_text_tokens
        )
        sample_list["attention_mask"][:, text_dropout_index] = 0
        sample_list["attention_mask"][:, pacth_dropout_index] = 0
        comp_embeddings, _, _ = self.get_joint_embedding(
            sample_list["input_ids"],
            sample_list["segment_ids"],
            sample_list["dv_image"],
            sample_list["visual_embeddings_type"],
            sample_list["attention_mask"],
        )
        comp_embeddings[:, pacth_dropout_index] = 0
        comp_embeddings = comp_embeddings[:, -num_visual_tokens:].sum(dim=1) / (
            num_visual_tokens - int(dropout_ratio * num_visual_tokens)
        )
        comp_embeddings = self.contrastive_norm(comp_embeddings)

        output_dict = {
            "scores": visual_embeddings,
            "targets": comp_embeddings,
        }

        loss = {}
        loss["icc_loss"] = self.loss_funcs["icc"](sample_list, output_dict)
        output_dict["losses"] = loss

        return output_dict

    def _forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if sample_list["task"] == "itm":
            ouput_dict = self._forward_itm(sample_list)
        elif sample_list["task"] == "mlm":
            ouput_dict = self._forward_mlm(sample_list)
        elif sample_list["task"] == "mpfr":
            ouput_dict = self._forward_mpfr(sample_list)
        elif sample_list["task"] == "mpfc":
            ouput_dict = self._forward_mpfc(sample_list)
        elif sample_list["task"] == "2wpa":
            ouput_dict = self._forward_2wpa(sample_list)
        elif sample_list["task"] == "mvc":
            ouput_dict = self._forward_mvc(sample_list)
        elif sample_list["task"] == "pac":
            ouput_dict = self._forward_pac(sample_list)
        elif sample_list["task"] == "icc":
            ouput_dict = self._forward_icc(sample_list)
        else:
            ouput_dict = self._forward_itc(sample_list)

        return ouput_dict
