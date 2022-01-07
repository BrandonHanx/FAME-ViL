# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict, Optional

import torch
from mmf.models.composition import NormalizationLayer
from mmf.models.fashionvil.base import FashionViLBaseModel
from torch import Tensor


class FashionViLForComposition(FashionViLBaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.norm_layer = NormalizationLayer()

    def _forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        tar_visual_embeddings: Tensor,
        tar_visual_embeddings_type: Tensor,
        ref_visual_embeddings: Tensor,
        ref_visual_embeddings_type: Tensor,
        comp_attention_mask: Tensor,
        visual_attention_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        tar_embeddings, _, _ = self.bert.get_image_embedding(
            tar_visual_embeddings, tar_visual_embeddings_type, visual_attention_mask
        )
        tar_embeddings = tar_embeddings.mean(dim=1)
        tar_embeddings = self.norm_layer(tar_embeddings)

        comp_embeddings, _, _ = self.bert.get_joint_embedding(
            input_ids,
            token_type_ids,
            ref_visual_embeddings,
            ref_visual_embeddings_type,
            comp_attention_mask,
        )
        num_visual_tokens = tar_visual_embeddings.shape[1]
        comp_embeddings = comp_embeddings[:, -num_visual_tokens:].mean(dim=1)
        comp_embeddings = self.norm_layer(comp_embeddings)

        output_dict = {
            "scores": comp_embeddings,
            "targets": tar_embeddings,
        }
        return output_dict

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

    def forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        sample_list = self.flatten_for_bert(sample_list)
        sample_list = self.add_post_flatten_params(sample_list)

        output_dict = self._forward(
            input_ids=sample_list["input_ids"],
            token_type_ids=sample_list["segment_ids"],
            tar_visual_embeddings=sample_list["tar_image"],
            tar_visual_embeddings_type=sample_list["tar_visual_embeddings_type"],
            ref_visual_embeddings=sample_list["ref_image"],
            ref_visual_embeddings_type=sample_list["ref_visual_embeddings_type"],
            comp_attention_mask=sample_list["comp_attention_mask"],
            visual_attention_mask=sample_list["visual_attention_mask"],
        )
        return output_dict
