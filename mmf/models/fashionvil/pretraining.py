# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict

import torch
from mmf.models.composition import NormalizationLayer
from mmf.models.fashionvil.base import FashionViLBaseModel
from mmf.modules.losses import ContrastiveLoss, CrossEntropyLoss
from torch import Tensor, nn
from transformers.modeling_bert import BertOnlyNSPHead


class FashionViLForPretraining(FashionViLBaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.tasks = config.tasks
        self.heads = nn.ModuleDict()
        self.loss_funcs = nn.ModuleDict()

        self.init_heads()
        self.init_losses()

    def init_heads(self):
        if "itm" in self.tasks:
            self.heads["itm"] = BertOnlyNSPHead(self.bert.config)
        if "itc" in self.tasks:
            self.heads["itc"] = NormalizationLayer()

    def init_losses(self):
        if "itm" in self.tasks:
            self.loss_funcs["itm"] = CrossEntropyLoss()
        if "itc" in self.tasks:
            self.loss_funcs["itc"] = ContrastiveLoss()

    def flatten_for_bert(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        to_be_flattened = ["input_ids", "segment_ids"]
        to_be_flattened_dim = ["image"]
        flattened = self.flatten(sample_list, to_be_flattened, to_be_flattened_dim)
        return flattened

    def add_post_flatten_params(
        self,
        sample_list: Dict[str, Tensor],
        task: str,
    ) -> Dict[str, Tensor]:
        b, l, _ = sample_list["image"].shape
        device = sample_list["image"].device
        sample_list["visual_embeddings_type"] = torch.zeros(
            (b, l), device=device
        ).long()
        if task == "itm":
            sample_list["attention_mask"] = torch.cat(
                (
                    sample_list["input_mask"],
                    torch.ones((b, l), device=device).long(),
                ),
                dim=-1,
            )
        return sample_list

    def forward_itc(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        input_ids = sample_list["input_ids"]
        token_type_ids = sample_list["segment_ids"]
        visual_embeddings = sample_list["image"]
        visual_embeddings_type = sample_list["visual_embeddings_type"]
        text_attention_mask = sample_list["input_mask"]

        visual_embeddings, _, _ = self.bert.get_image_embedding(
            visual_embeddings, visual_embeddings_type
        )
        visual_embeddings = visual_embeddings.mean(dim=1)
        visual_embeddings = self.heads["itc"](visual_embeddings)

        text_embeddings, _, _ = self.bert.get_text_embedding(
            input_ids,
            token_type_ids,
            text_attention_mask,
        )
        text_embeddings = text_embeddings[:, 0]
        text_embeddings = self.heads["itc"](text_embeddings)

        output_dict = {
            "scores": visual_embeddings,
            "targets": text_embeddings,
        }

        loss = {}
        loss["itc_loss"] = self.loss_funcs["itc"](sample_list, output_dict)
        output_dict["losses"] = loss

        return output_dict

    def forward_itm(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        input_ids = sample_list["input_ids"]
        token_type_ids = sample_list["segment_ids"]
        visual_embeddings = sample_list["image"]
        visual_embeddings_type = sample_list["visual_embeddings_type"]
        attention_mask = sample_list["attention_mask"]

        # Hard negative pais mining
        # FIXME: not support multi-gpu
        self.eval()
        with torch.no_grad():
            itc_dict = self.forward_itc(sample_list)
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

    def forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if torch.rand(1) > 0.3 and self.training:
            task = "itm"
        else:
            task = "itc"
        sample_list = self.flatten_for_bert(sample_list)
        sample_list = self.add_post_flatten_params(sample_list, task)
        if task == "itm":
            ouput_dict = self.forward_itm(sample_list)
        else:
            ouput_dict = self.forward_itc(sample_list)

        return ouput_dict
