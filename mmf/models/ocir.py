# Copyright (c) Facebook, Inc. and its affiliates.

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.models.composition import NormalizationLayer
from mmf.utils.build import build_image_encoder
from mmf.utils.general import filter_grads
from omegaconf import MISSING
from torch.nn.functional import one_hot


class CSAFusion(nn.Module):
    def __init__(self, n_conditions, onehot_channel, img_channel):
        super().__init__()
        self.num_conditions = n_conditions
        self.weight_classifier = nn.Sequential(
            nn.Linear(onehot_channel, 64),
            nn.ReLU(),
            nn.Linear(64, n_conditions),
            nn.Softmax(dim=-1),
        )
        self.masks = torch.nn.Embedding(n_conditions, img_channel)

    def forward(self, x, c):
        weight = self.weight_classifier(c)  # B x 5
        x = x.unsqueeze(1).expand(-1, 5, -1)  # B x 5 x D
        x = x * self.masks.weight.unsqueeze(0)  # B x 5 x D
        x = x * weight.unsqueeze(-1)  # B x 5 x D
        return x.mean(dim=1)


@registry.register_model("csa_net")
class CSANet(BaseModel):
    @dataclass
    class Config(BaseModel.Config):
        image_encoder: Any = MISSING
        image_channel: int = 512
        feature_dim: int = 64
        n_categories: int = 153
        n_conditions: int = 5

    def __init__(self, config: BaseModel.Config):
        """Initialize the config which is the model configuration."""
        super().__init__(config)
        self.config = config
        self.build()

    @classmethod
    def config_path(cls):
        return "configs/models/csa_net/defaults.yaml"

    @staticmethod
    def set_bn_eval(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    @staticmethod
    def set_bn_no_grad(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            for p in module.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.image_encoder.apply(self.set_bn_eval)

    def build(self):
        self.image_encoder = build_image_encoder(self.config.image_encoder)
        # Set block fixed
        for p in self.image_encoder.model[-1].parameters():
            p.requires_grad = False
        self.image_encoder.apply(self.set_bn_eval)
        self.image_encoder.apply(self.set_bn_no_grad)
        self.proj_layer = nn.Linear(self.config.image_channel, self.config.feature_dim)
        self.fusion_module = CSAFusion(
            self.config.n_conditions,
            self.config.n_categories * 2,
            self.config.feature_dim,
        )
        self.norm_layer = NormalizationLayer(normalize_scale=4.0, learn_scale=True)

    def get_optimizer_parameters(self, config):
        base_lr = config.optimizer.params.lr
        backbone_params = [
            {
                "params": filter_grads(self.image_encoder.parameters()),
                "lr": base_lr * 1,
            }
        ]
        rest_params = [
            {"params": filter_grads(self.proj_layer.parameters()), "lr": base_lr},
            {"params": filter_grads(self.fusion_module.parameters()), "lr": base_lr},
            {"params": filter_grads(self.norm_layer.parameters()), "lr": base_lr},
        ]
        training_parameters = backbone_params + rest_params

        return training_parameters

    def forward(self, sample_list):
        if sample_list.question_image.dim() > 4:
            sample_list.question_image = torch.flatten(
                sample_list.question_image, end_dim=1
            )
            sample_list.combine_cat_id = torch.flatten(
                sample_list.combine_cat_id, end_dim=1
            )
            sample_list.ann_idx = torch.flatten(sample_list.ann_idx, end_dim=1)

        question_onehot = one_hot(
            sample_list.combine_cat_id[:, 0], num_classes=self.config.n_categories
        )
        blank_onehot = one_hot(
            sample_list.combine_cat_id[:, 1], num_classes=self.config.n_categories
        )
        twohot = (
            torch.cat([question_onehot, blank_onehot], dim=1)
            .float()
            .to(sample_list.combine_cat_id.device)
        )

        question_feat = self.image_encoder(sample_list.question_image).squeeze()
        question_feat = self.proj_layer(question_feat)
        question_feat = self.fusion_module(question_feat, twohot)
        question_feat = self.norm_layer(question_feat)
        if self.training:
            pass
        #     negative_feat = self.image_encoder(sample_list.negative_image).squeeze()
        #     negative_feat = self.proj_layer(negative_feat)
        #     negative_feat = self.norm_layer(negative_feat)
        else:
            diff_idx = torch.diff(sample_list.ann_idx)
            feat_pool = []
            feat = question_feat[0]
            n_feat = 1
            for i, di in enumerate(diff_idx):
                if di == 0:
                    n_feat += 1
                    feat += question_feat[i + 1]
                else:
                    feat_pool.append(feat / n_feat)
                    n_feat = 1
                    feat = question_feat[i + 1]
            feat_pool.append(feat / n_feat)
            question_feat = torch.stack(feat_pool)

        blank_feat = self.image_encoder(sample_list.blank_image).squeeze()
        blank_feat = self.proj_layer(blank_feat)
        blank_feat = self.norm_layer(blank_feat)

        output = {
            "scores": question_feat,
            "targets": blank_feat,
            # "negative": negative_feat if self.training else None,
        }

        return output
