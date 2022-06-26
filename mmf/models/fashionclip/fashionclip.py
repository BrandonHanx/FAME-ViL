# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict

from mmf.common.registry import registry
from mmf.models import BaseModel
from torch import Tensor

from .composition import FashionCLIPForComposition


@registry.register_model("fashionclip")
class FashionCLIP(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.training_head_type = config.training_head_type

    @classmethod
    def config_path(cls):
        return "configs/models/fashionclip/defaults.yaml"

    def build(self):
        if self.training_head_type == "composition":
            self.model = FashionCLIPForComposition(self.config)
        else:
            raise NotImplementedError

    def forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.model(sample_list)
