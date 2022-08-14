# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict

from mmf.common.registry import registry
from mmf.models import BaseModel
from mmf.utils.modeling import get_fashionvil_configured_parameters
from torch import Tensor

from .composition import FashionCLIPForComposition
from .mtl import FashionCLIPForMTL


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
        elif self.training_head_type == "mtl":
            self.model = FashionCLIPForMTL(self.config)
        else:
            raise NotImplementedError

    def get_optimizer_parameters(self, config):
        base_lr = config.optimizer.params.lr
        weight_decay = config.optimizer.params.weight_decay
        lr_multiplier = self.config.lr_multiplier

        print(self.model.state_dict().keys())

        lr_filter = ["adapt_mlp", "heads"]
        params = get_fashionvil_configured_parameters(
            self.model,
            base_lr,
            weight_decay,
            lr_filter,
            lr_multiplier,
        )
        return params

    def forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.model(sample_list)
