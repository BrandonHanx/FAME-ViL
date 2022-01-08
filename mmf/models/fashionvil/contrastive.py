# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict

import torch
from mmf.models.composition import NormalizationLayer
from mmf.models.fashionvil.base import FashionViLBaseModel
from torch import Tensor


class FashionViLForContrastive(FashionViLBaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.norm_layer = NormalizationLayer()

    def flatten_for_bert(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        to_be_flattened = ["input_ids", "segment_ids"]
        to_be_flattened_dim = ["image"]
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
        return sample_list

    def _forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        visual_embeddings, _, _ = self.bert.get_image_embedding(
            sample_list["image"],
            sample_list["visual_embeddings_type"],
        )
        visual_embeddings = visual_embeddings.mean(dim=1)
        visual_embeddings = self.norm_layer(visual_embeddings)

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
        text_embeddings = self.norm_layer(text_embeddings)

        output_dict = {
            "scores": visual_embeddings,
            "targets": text_embeddings,
        }
        return output_dict
