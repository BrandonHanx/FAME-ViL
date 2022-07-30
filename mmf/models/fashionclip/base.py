# Copyright (c) Facebook, Inc. and its affiliates.

import os
from typing import Dict, List

from mmf.utils.configuration import get_mmf_cache_dir
from mmf.utils.transform import (
    transform_to_batch_sequence,
    transform_to_batch_sequence_dim,
)
from omegaconf import OmegaConf
from torch import Tensor, nn

from .modelling_clip import CLIPModelWithAdapter, CLIPAdapterConfig


class FashionCLIPBaseModel(nn.Module):
    def __init__(self, config, adapter_config=None):
        super().__init__()
        clip_model_name = getattr(
            config, "clip_model_name", "openai/clip-vit-base-patch16"
        )
        if adapter_config is not None:
            adapter_config = CLIPAdapterConfig(
                **OmegaConf.to_container(adapter_config, resolve=True)
            )

        self.config = config
        self.clip = CLIPModelWithAdapter.from_pretrained(
            clip_model_name,
            adapter_config=adapter_config,
            cache_dir=os.path.join(get_mmf_cache_dir(), "distributed_{}".format(-1)),
        )

    @staticmethod
    def flatten(
        sample_list: Dict[str, Tensor],
        to_be_flattened: List[str],
        to_be_flattened_dim: List[str],
    ) -> Dict[str, Tensor]:
        for key in to_be_flattened:
            # Make sure these keys are present or otherwise set these keys to None
            sample_list[key] = transform_to_batch_sequence(sample_list[key])
        for key in to_be_flattened_dim:
            sample_list[key] = transform_to_batch_sequence_dim(sample_list[key])
        return sample_list

    def add_custom_params(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return sample_list

    def flatten_for_clip(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return sample_list

    def add_post_flatten_params(
        self, sample_list: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        return sample_list

    def _forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        raise NotImplementedError

    def forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        sample_list = self.add_custom_params(sample_list)
        sample_list = self.flatten_for_clip(sample_list)
        sample_list = self.add_post_flatten_params(sample_list)
        return self._forward(sample_list)
