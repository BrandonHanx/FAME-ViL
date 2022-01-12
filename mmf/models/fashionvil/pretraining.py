# Copyright (c) Facebook, Inc. and its affiliates.

import os
import random
from copy import deepcopy
from typing import Dict

import torch
from mmf.models.composition import NormalizationLayer
from mmf.models.fashionvil.base import FashionViLBaseModel
from mmf.modules.losses import ContrastiveLoss, CrossEntropyLoss, MSELoss
from mmf.utils.configuration import get_mmf_cache_dir
from torch import Tensor, nn
from transformers.modeling_bert import (
    BertOnlyNSPHead,
    BertForPreTraining,
    BertPredictionHeadTransform,
)


class FashionViLForPretraining(FashionViLBaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.task_for_inference = config.task_for_inference
        self.tasks = config.tasks
        self.task_probs = config.task_probs
        self.heads = nn.ModuleDict()
        self.loss_funcs = nn.ModuleDict()

        self.init_heads()
        self.init_losses()

    def init_heads(self):
        if "itm" in self.tasks:
            self.heads["itm"] = BertOnlyNSPHead(self.bert.config)
        if "itc" in self.tasks:
            self.heads["itc"] = NormalizationLayer()
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

    def init_losses(self):
        if "itm" in self.tasks:
            self.loss_funcs["itm"] = CrossEntropyLoss()
        if "itc" in self.tasks:
            self.loss_funcs["itc"] = ContrastiveLoss()
        if "mlm" in self.tasks:
            self.loss_funcs["mlm"] = CrossEntropyLoss(ignore_index=-1)
        if "mpfr" in self.tasks:
            self.loss_funcs["mpfr"] = MSELoss()

    def add_custom_params(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if self.training:
            sample_list["task"] = random.choices(self.tasks, weights=self.task_probs)[0]
        else:
            sample_list["task"] = self.task_for_inference
        return sample_list

    def flatten_for_bert(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        to_be_flattened = ["input_ids", "segment_ids"]
        to_be_flattened_dim = ["image"]
        if sample_list["task"] == "mlm":
            to_be_flattened += ["lm_label_ids", "input_ids_masked"]
        elif sample_list["task"] == "mpfr":
            to_be_flattened += ["image_masks"]
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
        if sample_list["task"] in ["itm", "mlm", "mpfr"]:
            sample_list["attention_mask"] = torch.cat(
                (
                    sample_list["input_mask"],
                    torch.ones((b, l), device=device).long(),
                ),
                dim=-1,
            )
        if sample_list["task"] == "mpfr":
            mask = sample_list["image_masks"] == 0
            mask = mask.float().unsqueeze(-1)
            sample_list["image"] = sample_list["image"] * mask
        return sample_list

    def _forward_itc(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        visual_embeddings, _, _ = self.bert.get_image_embedding(
            sample_list["image"], sample_list["visual_embeddings_type"]
        )
        visual_embeddings = visual_embeddings.mean(dim=1)
        visual_embeddings = self.heads["itc"](visual_embeddings)

        text_embeddings, _, _ = self.bert.get_text_embedding(
            sample_list["input_ids"],
            sample_list["segment_ids"],
            sample_list["input_mask"],
        )
        masks = sample_list["input_mask"]
        text_embeddings = text_embeddings * masks.unsqueeze(2)
        text_embeddings = torch.sum(text_embeddings, dim=1) / (
            torch.sum(masks, dim=1, keepdim=True)
        )
        text_embeddings = self.heads["itc"](text_embeddings)

        output_dict = {
            "scores": visual_embeddings,
            "targets": text_embeddings,
        }

        loss = {}
        loss["itc_loss"] = self.loss_funcs["itc"](sample_list, output_dict)
        output_dict["losses"] = loss

        return output_dict

    def _forward_itm(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        input_ids = sample_list["input_ids"]
        token_type_ids = sample_list["segment_ids"]
        visual_embeddings = sample_list["image"]
        visual_embeddings_type = sample_list["visual_embeddings_type"]
        attention_mask = sample_list["attention_mask"]

        # Hard negative pais mining
        # FIXME: not support multi-gpu
        self.eval()
        with torch.no_grad():
            itc_dict = self._forward_itc(sample_list)
            image_embeddings = itc_dict["scores"]
            text_embeddings = itc_dict["targets"]
            correlations = image_embeddings @ text_embeddings.t()
            batch_size = correlations.shape[0]
            diag = torch.eye(batch_size).bool()
            correlations[diag] = -1
            # FIXME: more complicated sampling strategy
            hard_text_index = torch.argmax(correlations, dim=1)
            combine_index = torch.arange(batch_size).to(image_embeddings.device)
            combine_index_index = torch.rand(batch_size) > 0.5
            combine_index[combine_index_index] = hard_text_index[combine_index_index]

        input_ids = input_ids[combine_index]
        token_type_ids = token_type_ids[combine_index]
        attention_mask = attention_mask[combine_index]
        sample_list["targets"][combine_index_index] = 0

        self.train()
        _, pooled_output, _ = self.bert.get_joint_embedding(
            input_ids,
            token_type_ids,
            visual_embeddings,
            visual_embeddings_type,
            attention_mask,
        )
        logits = self.heads["itm"](pooled_output)
        reshaped_logits = logits.contiguous().view(-1, 2)
        output_dict = {"scores": reshaped_logits}

        loss = {}
        loss["itm_loss"] = self.loss_funcs["itm"](sample_list, output_dict)
        output_dict["losses"] = loss

        return output_dict

    def _forward_mlm(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        sequence_output, _, _ = self.bert.get_joint_embedding(
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
        hidden, _, _ = self.bert.get_joint_embedding(
            sample_list["input_ids"],
            sample_list["segment_ids"],
            sample_list["image"],
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

        sample_list["targets"] = target_masked

        output_dict = {"scores": predict_feat}

        loss = {}
        loss["mpfr_loss"] = self.loss_funcs["mpfr"](sample_list, output_dict)
        output_dict["losses"] = loss

        return output_dict

    def _forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if sample_list["task"] == "itm":
            ouput_dict = self._forward_itm(sample_list)
        elif sample_list["task"] == "mlm":
            ouput_dict = self._forward_mlm(sample_list)
        elif sample_list["task"] == "mpfr":
            ouput_dict = self._forward_mpfr(sample_list)
        else:
            ouput_dict = self._forward_itc(sample_list)

        return ouput_dict
