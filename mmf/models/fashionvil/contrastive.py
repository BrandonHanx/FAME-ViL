# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict, Optional

import torch
from mmf.models.composition import NormalizationLayer
from mmf.models.fashionvil.base import FashionViLBaseModel
from torch import Tensor


class FashionViLForContrastive(FashionViLBaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.norm_layer = NormalizationLayer()

    def _forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        visual_embeddings: Tensor,
        visual_embeddings_type: Tensor,
        text_attention_mask: Tensor,
        visual_attention_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        visual_embeddings, _, _ = self.bert.get_image_embedding(
            visual_embeddings, visual_embeddings_type, visual_attention_mask
        )
        visual_embeddings = visual_embeddings.mean(dim=1)
        visual_embeddings = self.norm_layer(visual_embeddings)

        text_embeddings, _, _ = self.bert.get_text_embedding(
            input_ids,
            token_type_ids,
            text_attention_mask,
        )
        text_embeddings = text_embeddings[:, 0]
        text_embeddings = self.norm_layer(text_embeddings)

        output_dict = {
            "scores": visual_embeddings,
            "targets": text_embeddings,
        }
        return output_dict

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

    def forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        sample_list = self.flatten_for_bert(sample_list)
        sample_list = self.add_post_flatten_params(sample_list)

        output_dict = self._forward(
            input_ids=sample_list["input_ids"],
            token_type_ids=sample_list["segment_ids"],
            visual_embeddings=sample_list["image"],
            visual_embeddings_type=sample_list["visual_embeddings_type"],
            text_attention_mask=sample_list["input_mask"],
        )

        return output_dict
