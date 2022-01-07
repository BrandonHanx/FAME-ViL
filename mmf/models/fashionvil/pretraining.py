# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict

from mmf.models.fashionvil.base import FashionViLBaseModel
from mmf.modules.losses import ContrastiveLoss, CrossEntropyLoss
from torch import Tensor, nn
from transformers.modeling_bert import (
    BertOnlyNSPHead,
    BertPredictionHeadTransform,
)


class FashionViLForPretraining(FashionViLBaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.tasks = config.tasks
        self.losses = config.losses
        self.heads = nn.ModuleDict()
        self.loss_funcs = nn.ModuleDict()

        self.init_heads()
        self.init_losses()

    def init_heads(self):
        if "itm" in self.tasks:
            self.heads["itm"] = BertOnlyNSPHead(self.bert.config)
        if "itc" in self.tasks:
            self.heads["itc"] = BertPredictionHeadTransform(self.bert.config)

    def init_losses(self):
        if "itm" in self.tasks:
            self.loss_funcs["itm"] = CrossEntropyLoss()
        if "itc" in self.tasks:
            self.loss_funcs["itc"] = ContrastiveLoss()

    def forward_itm(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        visual_embeddings: Tensor,
        visual_embeddings_type: Tensor,
        attention_mask: Tensor,
    ) -> Dict[str, Tensor]:
        _, pooled_output, _ = self.bert.get_joint_embedding(
            input_ids,
            token_type_ids,
            visual_embeddings,
            visual_embeddings_type,
            attention_mask,
        )
        logits = self.heads["itm"](pooled_output)
        reshaped_logits = logits.contiguous().view(-1, self.num_labels)
        output_dict = {"scores": reshaped_logits}
        return output_dict
