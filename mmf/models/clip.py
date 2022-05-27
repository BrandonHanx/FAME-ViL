# Copyright (c) Facebook, Inc. and its affiliates.

import clip
import torch
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel


@registry.register_model("clip")
class CLIP(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    @classmethod
    def config_path(cls):
        return "configs/models/clip/defaults.yaml"

    def build(self):
        self.model, _ = clip.load(self.config.name)
        self.model = self.model.float()

    def forward(self, sample_list):
        if sample_list.image.dim() > 4:
            sample_list.image = torch.flatten(sample_list.image, end_dim=-4)
        image_feats = self.model.encode_image(sample_list.image)
        text = clip.tokenize(sample_list.text, truncate=True).to(image_feats.device)
        text_feats = self.model.encode_text(text)

        output = {
            "scores": image_feats,
            "targets": text_feats,
        }

        return output
