# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict

from mmf.models.composition import NormalizationLayer, VectorAddition
from torch import Tensor

from .base import FashionCLIPBaseModel


class FashionCLIPForComposition(FashionCLIPBaseModel):
    def __init__(self, config, adapter_config):
        super().__init__(config, adapter_config)
        self.comp_mode = config.get("comp_mode", "va")
        self.norm_layer = NormalizationLayer()
        if self.comp_mode == "va":
            self.compositor = VectorAddition()
        else:
            raise NotImplementedError

    def flatten_for_clip(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        to_be_flattened = ["input_ids", "attention_mask"]
        to_be_flattened_dim = []
        flattened = self.flatten(sample_list, to_be_flattened, to_be_flattened_dim)
        return flattened

    def _forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        tar_embeddings = self.clip.get_image_features(sample_list.tar_image)
        tar_embeddings = self.norm_layer(tar_embeddings)

        ref_embeddings = self.clip.get_image_features(sample_list.ref_image)
        text_embeddings = self.clip.get_text_features(
            sample_list.input_ids, sample_list.attention_mask
        )
        comp_embeddings = self.compositor(ref_embeddings, text_embeddings)
        comp_embeddings = self.norm_layer(comp_embeddings)

        output_dict = {
            "comp_feats": comp_embeddings,
            "tar_feats": tar_embeddings,
        }
        return output_dict
