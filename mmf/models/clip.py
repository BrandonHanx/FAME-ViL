# Copyright (c) Facebook, Inc. and its affiliates.

import clip
import torch
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.models.composition import NormalizationLayer
from mmf.utils.modeling import get_clip_text_encoder_configured_parameters


@registry.register_model("clip")
class CLIP(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.norm_layer = NormalizationLayer()

    @classmethod
    def config_path(cls):
        return "configs/models/clip/defaults.yaml"

    def build(self):
        self.model, _ = clip.load(self.config.name)
        self.model = self.model.float()

    def get_optimizer_parameters(self, config):
        base_lr = config.optimizer.params.lr
        weight_decay = config.optimizer.params.weight_decay
        lr_multiplier = self.config.lr_multiplier

        text_encoder_params = get_clip_text_encoder_configured_parameters(
            self.model,
            base_lr,
            weight_decay,
        )
        image_encoder_params = [
            {
                "params": self.model.visual.parameters(),
                "lr": base_lr * lr_multiplier,
            }
        ]
        rest_params = [
            {
                "params": self.norm_layer.parameters(),
                "lr": base_lr,
            }
        ]
        return image_encoder_params + text_encoder_params + rest_params

    def forward(self, sample_list):
        if sample_list.image.dim() > 4:
            sample_list.image = torch.flatten(sample_list.image, end_dim=-4)
        image_feats = self.model.encode_image(sample_list.image)
        text = clip.tokenize(sample_list.text, truncate=True).to(image_feats.device)
        text_feats = self.model.encode_text(text)

        output = {
            "scores": self.norm_layer(image_feats),
            "targets": self.norm_layer(text_feats),
        }

        return output
