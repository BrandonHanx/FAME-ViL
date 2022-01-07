# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict

import torch
from mmf.models.fashionvil.base import FashionViLBaseModel
from torch import Tensor, nn


class FashionViLForClassification(FashionViLBaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(self.bert.config.hidden_size, config.num_labels)

    def _forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        visual_embeddings: Tensor,
        visual_embeddings_type: Tensor,
        attention_mask: Tensor,
    ) -> Dict[str, Tensor]:
        _, pooled_output, _ = self.bert.get_joint_embedding(
            input_ids,
            token_type_ids,
            visual_embeddings,
            visual_embeddings_type,
            attention_mask,
        )
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.contiguous().view(-1, self.num_labels)
        output_dict = {"scores": reshaped_logits}
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
        sample_list["attention_mask"] = torch.cat(
            (
                sample_list["input_mask"],
                torch.ones((b, l), device=device).long(),
            ),
            dim=-1,
        )
        return sample_list

    def forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        sample_list = self.flatten_for_bert(sample_list)
        sample_list = self.add_post_flatten_params(sample_list)

        output_dict = self._forward(
            input_ids=sample_list["input_ids"],
            token_type_ids=sample_list["segment_ids"],
            visual_embeddings=sample_list["image"],
            visual_embeddings_type=sample_list["visual_embeddings_type"],
            attention_mask=sample_list["attention_mask"],
        )

        return output_dict
