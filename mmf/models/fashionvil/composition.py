# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict

import torch
from mmf.models.composition import NormalizationLayer
from mmf.models.fashionvil.base import FashionViLBaseModel
from torch import Tensor


class FashionViLForComposition(FashionViLBaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.comp_mode = config.get("comp_mode", "heavy")
        self.norm_layer = NormalizationLayer()

    def add_post_flatten_params(
        self, sample_list: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        b, l, _ = sample_list["ref_image"].shape
        device = sample_list["ref_image"].device

        sample_list["tar_visual_embeddings_type"] = torch.zeros(
            (b, l), device=device
        ).long()
        sample_list["ref_visual_embeddings_type"] = torch.zeros(
            (b, l), device=device
        ).long()
        sample_list["comp_attention_mask"] = torch.cat(
            (sample_list["input_mask"], torch.ones((b, l), device=device).long()),
            dim=-1,
        )
        sample_list["visual_attention_mask"] = torch.ones((b, l), device=device).long()
        return sample_list

    def flatten_for_bert(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        to_be_flattened = ["input_ids", "segment_ids"]
        to_be_flattened_dim = ["ref_image", "tar_image"]
        flattened = self.flatten(sample_list, to_be_flattened, to_be_flattened_dim)
        return flattened

    def _forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        tar_embeddings, _, _ = self.bert.get_image_embedding(
            sample_list["tar_image"],
            sample_list["tar_visual_embeddings_type"],
            sample_list["visual_attention_mask"],
        )
        tar_embeddings = tar_embeddings.mean(dim=1)
        tar_embeddings = self.norm_layer(tar_embeddings)

        if self.comp_mode == "heavy":
            comp_embeddings, _, _ = self.bert.get_joint_embedding(
                sample_list["input_ids"],
                sample_list["segment_ids"],
                sample_list["ref_image"],
                sample_list["ref_visual_embeddings_type"],
                sample_list["comp_attention_mask"],
            )
            num_visual_tokens = sample_list["tar_image"].shape[1]
            comp_embeddings = comp_embeddings[:, -num_visual_tokens:].mean(dim=1)
        elif self.comp_mode == "va":
            ref_embeddings, _, _ = self.bert.get_image_embedding(
                sample_list["ref_image"],
                sample_list["ref_visual_embeddings_type"],
                sample_list["visual_attention_mask"],
            )
            ref_embeddings = ref_embeddings.mean(dim=1)
            text_embeddings, _, _ = self.bert.get_text_embedding(
                sample_list["input_ids"],
                sample_list["segment_ids"],
                sample_list["input_mask"],
            )
            # text_embeddings = text_embeddings[:, 0]
            masks = sample_list["input_mask"]
            text_embeddings = text_embeddings * masks.unsqueeze(2)
            text_embeddings = torch.sum(text_embeddings, dim=1) / (
                torch.sum(masks, dim=1, keepdim=True)
            )
            comp_embeddings = text_embeddings + ref_embeddings
        else:
            raise NotImplementedError
        comp_embeddings = self.norm_layer(comp_embeddings)

        output_dict = {
            "comp_feats": comp_embeddings,
            "tar_feats": tar_embeddings,
        }
        return output_dict
