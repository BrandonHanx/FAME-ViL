# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict

from mmf.common.registry import registry
from mmf.models import BaseModel
from mmf.models.fashionvil.classification import FashionViLForClassification
from mmf.models.fashionvil.composition import FashionViLForComposition
from mmf.models.fashionvil.contrastive import FashionViLForContrastive
from mmf.models.fashionvil.pretraining import FashionViLForPretraining
from torch import Tensor


@registry.register_model("fashionvil")
class FashionViL(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.training_head_type = config.training_head_type

    @classmethod
    def config_path(cls):
        return "configs/models/fashionvil/defaults.yaml"

    def build(self):
        if self.training_head_type == "pretraining":
            self.model = FashionViLForPretraining(self.config)
        elif self.training_head_type == "classification":
            self.model = FashionViLForClassification(self.config)
        elif self.training_head_type == "composition":
            self.model = FashionViLForComposition(self.config)
        elif self.training_head_type == "contrastive":
            self.model = FashionViLForContrastive(self.config)
        else:
            raise NotImplementedError

        if self.config.special_visual_initialize:
            self.model.bert.embeddings.initialize_visual_from_pretrained()

        if getattr(self.config, "freeze_base", False):
            for p in self.model.bert.parameters():
                p.requires_grad = False

    def forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.model(sample_list)
