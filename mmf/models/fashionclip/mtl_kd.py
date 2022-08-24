# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from typing import Dict

import torch
from mmf.modules.losses import (
    BatchBasedClassificationLossKD,
    ContrastiveLossKD,
    SoftLabelCrossEntropyLoss,
)
from torch import Tensor, nn

from .mtl import FashionCLIPForMTL


logger = logging.getLogger(__name__)


class FashionCLIPForMTLwithKD(FashionCLIPForMTL):
    def __init__(self, config):
        super().__init__(config)
        self.teachers = nn.ModuleDict()
        self.kd_loss_funcs = nn.ModuleDict()
        self.init_teachers()
        self.init_kd_losses()

    @staticmethod
    def _rename_state_dict(state_dict):
        new_state_dict = dict()
        for k, v in state_dict.items():
            if k.startswith("module.model."):
                new_state_dict[k[13:]] = v
            elif k.startswith("model."):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    def init_teachers(self):
        if self.config.get("itc_teacher_config", None) is not None:
            config = self.config.itc_teacher_config
            self.teachers["itc"] = FashionCLIPForMTL(config)
            state_dict = torch.load(
                config.pretrained_path,
                map_location=torch.device("cpu"),
            )
            state_dict = self._rename_state_dict(state_dict)
            self.teachers["itc"].load_state_dict(state_dict)
            logger.info(
                f"Successfully loaded ITC teacher from {config.pretrained_path}"
            )
        if self.config.get("tgir_teacher_config", None) is not None:
            config = self.config.tgir_teacher_config
            self.teachers["tgir"] = FashionCLIPForMTL(config)
            state_dict = torch.load(
                config.pretrained_path,
                map_location=torch.device("cpu"),
            )
            state_dict = self._rename_state_dict(state_dict)
            self.teachers["tgir"].load_state_dict(state_dict)
            logger.info(
                f"Successfully loaded TGIR teacher from {config.pretrained_path}"
            )
        if self.config.get("scr_teacher_config", None) is not None:
            config = self.config.scr_teacher_config
            self.teachers["scr"] = FashionCLIPForMTL(config)
            state_dict = torch.load(
                config.pretrained_path,
                map_location=torch.device("cpu"),
            )
            state_dict = self._rename_state_dict(state_dict)
            self.teachers["scr"].load_state_dict(state_dict)
            logger.info(
                f"Successfully loaded SCR teacher from {config.pretrained_path}"
            )
        for p in self.teachers.parameters():
            p.requires_grad = False

    def init_kd_losses(self):
        if "itc" in self.tasks:
            self.kd_loss_funcs["itc"] = ContrastiveLossKD()
        if "tgir" in self.tasks:
            self.kd_loss_funcs["tgir"] = BatchBasedClassificationLossKD()
        if "scr" in self.tasks:
            self.kd_loss_funcs["scr"] = SoftLabelCrossEntropyLoss()

    def _forward_itc(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        output_dict = super()._forward_itc(sample_list)
        if self.training:
            teacher_output_dict = self.teachers["itc"]._forward_itc(sample_list)
            output_dict["teacher_scores"] = teacher_output_dict["scores"]
            output_dict["teacher_targets"] = teacher_output_dict["targets"]
            output_dict["losses"]["kd_itc_loss"] = self.kd_loss_funcs["itc"](
                sample_list, output_dict
            )
        return output_dict

    def _forward_tgir(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        output_dict = super()._forward_tgir(sample_list)
        if self.training:
            teacher_output_dict = self.teachers["tgir"]._forward_tgir(sample_list)
            output_dict["teacher_comp_feats"] = teacher_output_dict["comp_feats"]
            output_dict["teacher_tar_feats"] = teacher_output_dict["tar_feats"]
            output_dict["losses"]["kd_tgir_loss"] = self.kd_loss_funcs["tgir"](
                sample_list, output_dict
            )
        return output_dict

    def _forward_scr(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        output_dict = super()._forward_scr(sample_list)
        if self.training:
            teacher_output_dict = self.teachers["scr"]._forward_scr(sample_list)
            sample_list["targets"] = teacher_output_dict["scores"]
            output_dict["losses"]["kd_scr_loss"] = self.kd_loss_funcs["scr"](
                sample_list, output_dict
            )
        return output_dict
