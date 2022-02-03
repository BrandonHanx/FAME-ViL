# Copyright (c) Facebook, Inc. and its affiliates.

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Tuple

import torch
import torch.nn as nn
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.encoders import IdentityEncoder
from mmf.modules.layers import AttnPool1d
from mmf.utils.build import (
    build_classifier_layer,
    build_image_encoder,
    build_text_encoder,
)
from mmf.utils.general import filter_grads
from mmf.utils.modeling import get_bert_configured_parameters
from mmf.utils.transform import transform_to_batch_sequence
from omegaconf import MISSING


class TIRG(nn.Module):
    """The TIGR model.
    The method is described in
    Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays.
    "Composing Text and Image for Image Retrieval - An Empirical Odyssey"
    CVPR 2019. arXiv:1812.07119
    """

    def __init__(self, text_channel, img_channel):
        super().__init__()

        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0]))
        self.gated_feature_composer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(img_channel + text_channel),
            torch.nn.ReLU(),
            torch.nn.Linear(img_channel + text_channel, img_channel),
        )
        self.res_info_composer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(img_channel + text_channel),
            torch.nn.ReLU(),
            torch.nn.Linear(img_channel + text_channel, img_channel + text_channel),
            torch.nn.ReLU(),
            torch.nn.Linear(img_channel + text_channel, img_channel),
        )

    def forward(self, img_features, text_features):
        x = torch.cat((img_features, text_features), dim=1)
        f1 = self.gated_feature_composer(x)
        f2 = self.res_info_composer(x)
        f = torch.sigmoid(f1) * img_features * self.a[0] + f2 * self.a[1]
        return f


class NormalizationLayer(nn.Module):
    """Class for normalization layer."""

    def __init__(self, normalize_scale=4.0, learn_scale=True):
        super().__init__()
        self.norm_s = float(normalize_scale)
        if learn_scale:
            self.norm_s = nn.Parameter(torch.FloatTensor((self.norm_s,)))

    def forward(self, x, dim=-1):
        features = self.norm_s * nn.functional.normalize(x, dim=dim)
        return features


class BaseComposition(BaseModel):
    @dataclass
    class Config(BaseModel.Config):
        direct_features_input: bool = False
        image_encoder: Any = MISSING
        text_encoder: Any = MISSING
        compositor: Any = MISSING
        norm_layer: Any = MISSING
        image_projection: Any = IdentityEncoder.Config()
        text_projection: Any = IdentityEncoder.Config()

    def __init__(self, config: Config):
        """Initialize the config which is the model configuration."""
        super().__init__(config)
        self.config = config

    def preprocess_text(self, sample_list) -> Tuple:
        raise NotImplementedError("Text processing not implemented")

    def preprocess_image(self, sample_list) -> Tuple:
        raise NotImplementedError("Image processing not implemented")

    def get_ref_image_embedding(self, sample_list) -> torch.Tensor:
        raise NotImplementedError("Image Encoder not implemented")

    def get_tar_image_embedding(self, sample_list) -> torch.Tensor:
        raise NotImplementedError("Image Encoder not implemented")

    def get_text_embedding(self, sample_list) -> torch.Tensor:
        raise NotImplementedError("Text Encoder not implemented")

    def get_comp_embedding(self, sample_list) -> torch.Tensor:
        raise NotImplementedError("Compositor not implemented")


@registry.register_model("simple_composition")
class SimpleComposition(BaseComposition):
    def __init__(self, config: BaseComposition.Config):
        """Initialize the config which is the model configuration."""
        super().__init__(config)
        self.config = config
        self.build()

    @classmethod
    def config_path(cls):
        return "configs/models/composition/defaults.yaml"

    def build(self):
        self._is_direct_features_input = self.config.direct_features_input
        # Encoders
        self.text_encoder = build_text_encoder(self.config.text_encoder)
        self.image_encoder = build_image_encoder(
            self.config.image_encoder, self._is_direct_features_input
        )

        # Projectors
        image_proj_config = deepcopy(self.config.image_projection)
        self.image_proj = build_classifier_layer(image_proj_config)

        text_proj_config = deepcopy(self.config.text_projection)
        self.text_proj = build_classifier_layer(text_proj_config)

        # Aggregators
        self.image_pool = AttnPool1d(self.config.final_hidden_size, 1)
        self.text_pool = AttnPool1d(self.config.final_hidden_size, 1)

        if self.config.compositor.type == "tirg":
            self.compositor = TIRG(**self.config.compositor.params)
        else:
            raise NotImplementedError("Compositor not implemented")

        self.norm_layer = NormalizationLayer(**self.config.norm_layer)

    def get_optimizer_parameters(self, config):
        base_lr = config.optimizer.params.lr
        bert_params = get_bert_configured_parameters(self.text_encoder, base_lr * 0.05)
        backbone_params = [
            {
                "params": filter_grads(self.image_encoder.parameters()),
                "lr": base_lr,
            }
        ]
        rest_params = [
            {"params": filter_grads(self.image_proj.parameters()), "lr": base_lr},
            {"params": filter_grads(self.text_proj.parameters()), "lr": base_lr},
            {"params": filter_grads(self.image_pool.parameters()), "lr": base_lr},
            {"params": filter_grads(self.text_pool.parameters()), "lr": base_lr},
            {"params": filter_grads(self.compositor.parameters()), "lr": base_lr},
            {"params": filter_grads(self.norm_layer.parameters()), "lr": base_lr},
        ]
        training_parameters = bert_params + backbone_params + rest_params

        return training_parameters

    def preprocess_text(self, sample_list) -> Tuple:
        text = transform_to_batch_sequence(sample_list.input_ids)
        mask = transform_to_batch_sequence(sample_list.input_mask)
        segment = transform_to_batch_sequence(sample_list.segment_ids)

        return (text, mask, segment)

    def _get_image_embedding(self, image_data) -> Tuple[torch.Tensor, torch.Tensor]:
        # image_data shape B x 3 x 224 x 224
        src = self.image_encoder(image_data)
        # src shape B x 49 x 2048
        if isinstance(src, dict):
            src = src[0]
        # Image embedding
        image_proj = self.image_proj(src)
        # image_proj shape B x 49 x 512
        # image_pool = self.image_pool(image_proj, image_proj).squeeze(1)
        image_pool = image_proj.mean(dim=1)
        return image_pool

    def get_ref_image_embedding(self, sample_list) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._get_image_embedding(sample_list.ref_image)

    def get_tar_image_embedding(self, sample_list) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.norm_layer(self._get_image_embedding(sample_list.tar_image))

    def get_text_embedding(self, sample_list) -> Tuple[torch.Tensor, torch.Tensor]:
        text, mask, segment = self.preprocess_text(sample_list)

        text_enc = self.text_encoder(text, mask, segment)

        # Text embedding
        text_proj = self.text_proj(text_enc[0])
        # text_pool = self.text_pool(text_proj, text_proj, mask.eq(0)).squeeze(1)
        masks = sample_list["input_mask"]
        text_proj = text_proj * masks.unsqueeze(2)
        text_pool = torch.sum(text_proj, dim=1) / (
            torch.sum(masks, dim=1, keepdim=True)
        )

        return text_pool

    def get_comp_embedding(self, sample_list) -> Tuple[torch.Tensor, torch.Tensor]:
        ref_img_ebd = self.get_ref_image_embedding(sample_list)
        text_ebd = self.get_text_embedding(sample_list)
        comp_ebd = self.compositor(ref_img_ebd, text_ebd)
        return self.norm_layer(comp_ebd)

    def forward(self, sample_list):
        comp_ebd = self.get_comp_embedding(sample_list)
        tar_ebd = self.get_tar_image_embedding(sample_list)

        output = {
            "scores": comp_ebd,
            "targets": tar_ebd,
        }

        return output
