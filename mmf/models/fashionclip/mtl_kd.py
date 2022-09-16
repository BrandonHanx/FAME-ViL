# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from typing import Dict

import torch
import torch.nn.functional as F
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
        self.balances = nn.ParameterDict()
        self.init_teachers()
        self.init_kd_losses()
        self.init_balances()

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
        if self.config.get("cap_teacher_config", None) is not None:
            config = self.config.cap_teacher_config
            self.teachers["cap"] = FashionCLIPForMTL(config)
            state_dict = torch.load(
                config.pretrained_path,
                map_location=torch.device("cpu"),
            )
            state_dict = self._rename_state_dict(state_dict)
            self.teachers["cap"].load_state_dict(state_dict)
            logger.info(
                f"Successfully loaded CAP teacher from {config.pretrained_path}"
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
        if "cap" in self.tasks:
            self.kd_loss_funcs["cap"] = SoftLabelCrossEntropyLoss()

    def init_balances(self):
        learnable_balances = self.config.get("learnable_balances", False)
        if "itc" in self.tasks:
            self.balances["itc"] = nn.Parameter(
                torch.ones(1), requires_grad=learnable_balances
            )
        if "tgir" in self.tasks:
            self.balances["tgir"] = nn.Parameter(
                torch.ones(1), requires_grad=learnable_balances
            )
        if "scr" in self.tasks:
            self.balances["scr"] = nn.Parameter(
                torch.ones(1), requires_grad=learnable_balances
            )
        if "cap" in self.tasks:
            self.balances["cap"] = nn.Parameter(
                torch.ones(1), requires_grad=learnable_balances
            )

    def _forward_itc(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        output_dict = super()._forward_itc(sample_list)
        if self.training:
            teacher_output_dict = self.teachers["itc"]._forward_itc(sample_list)
            output_dict["teacher_scores"] = teacher_output_dict["scores"]
            output_dict["teacher_targets"] = teacher_output_dict["targets"]
            output_dict["losses"]["kd_itc_loss"] = (
                self.kd_loss_funcs["itc"](sample_list, output_dict)
                * self.balances["itc"]
            )
            output_dict["losses"]["itc_loss"] = output_dict["losses"]["itc_loss"] * (
                2 - self.balances["itc"]
            )
        return output_dict

    def _forward_tgir(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        output_dict = super()._forward_tgir(sample_list)
        if self.training:
            teacher_output_dict = self.teachers["tgir"]._forward_tgir(sample_list)
            output_dict["teacher_comp_feats"] = teacher_output_dict["comp_feats"]
            output_dict["teacher_tar_feats"] = teacher_output_dict["tar_feats"]
            output_dict["losses"]["kd_tgir_loss"] = (
                self.kd_loss_funcs["tgir"](sample_list, output_dict)
                * self.balances["tgir"]
            )
            output_dict["losses"]["tgir_loss"] = output_dict["losses"]["tgir_loss"] * (
                2 - self.balances["tgir"]
            )
        return output_dict

    def _forward_scr(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        output_dict = super()._forward_scr(sample_list)
        if self.training:
            teacher_output_dict = self.teachers["scr"]._forward_scr(sample_list)
            sample_list["targets"] = F.softmax(teacher_output_dict["scores"], dim=-1)
            output_dict["losses"]["kd_scr_loss"] = (
                self.kd_loss_funcs["scr"](sample_list, output_dict)
                * self.balances["scr"]
            )
            output_dict["losses"]["scr_loss"] = output_dict["losses"]["scr_loss"] * (
                2 - self.balances["scr"]
            )
        return output_dict

    def _forward_cap(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        output_dict = super()._forward_cap(sample_list)
        if self.training:
            teacher_output_dict = self.teachers["cap"]._forward_cap(sample_list)
            sample_list["targets"] = F.softmax(teacher_output_dict["scores"], dim=-1)
            output_dict["losses"]["kd_cap_loss"] = (
                self.kd_loss_funcs["cap"](sample_list, output_dict)
                * self.balances["cap"]
            )
            output_dict["losses"]["cap_loss"] = output_dict["losses"]["cap_loss"] * (
                2 - self.balances["cap"]
            )
        return output_dict
