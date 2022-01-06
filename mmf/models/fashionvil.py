# Copyright (c) Facebook, Inc. and its affiliates.

import os
from typing import Dict, List, Optional, Tuple

import torch
from mmf.common.registry import registry
from mmf.models import BaseModel
from mmf.models.composition import NormalizationLayer
from mmf.modules.embeddings import BertVisioLinguisticEmbeddings
from mmf.modules.hf_layers import BertEncoderJit
from mmf.utils.configuration import get_mmf_cache_dir
from mmf.utils.modeling import get_optimizer_parameters_for_bert
from mmf.utils.transform import (
    transform_to_batch_sequence,
    transform_to_batch_sequence_dim,
)
from omegaconf import OmegaConf
from torch import Tensor, nn
from transformers.modeling_bert import (
    BertConfig,
    BertPooler,
    BertPreTrainedModel,
    BertPredictionHeadTransform,
)


class FashionViLBase(BertPreTrainedModel):
    def __init__(
        self,
        config,
        visual_embedding_dim=2048,
        output_attentions=False,
        output_hidden_states=False,
    ):
        super().__init__(config)
        self.config = config

        config.visual_embedding_dim = visual_embedding_dim
        config.output_attentions = output_attentions
        config.output_hidden_states = output_hidden_states

        self.embeddings = BertVisioLinguisticEmbeddings(config)
        self.encoder = BertEncoderJit(config)
        self.pooler = BertPooler(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.bypass_transformer = config.bypass_transformer

        self.init_weights()

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        visual_embeddings: Optional[Tensor] = None,
        visual_embeddings_type: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:

        extended_attention_mask = None
        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            if not torch.jit.is_scripting():
                extended_attention_mask = extended_attention_mask.to(
                    dtype=next(self.parameters()).dtype
                )  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(
            input_ids,
            token_type_ids,
            visual_embeddings=visual_embeddings,
            visual_embeddings_type=visual_embeddings_type,
        )

        encoded_layers = self.encoder(embedding_output, extended_attention_mask)
        sequence_output = encoded_layers[0]
        pooled_output = self.pooler(sequence_output)
        attn_data_list = []

        if not torch.jit.is_scripting():
            if self.output_attentions:
                attn_data_list = encoded_layers[1:]
        else:
            assert (
                not self.output_attentions
            ), "output_attentions not supported in script mode"

        return sequence_output, pooled_output, attn_data_list

    def get_joint_embedding(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        visual_embeddings: Tensor,
        visual_embeddings_type: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        return self.forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            visual_embeddings=visual_embeddings,
            visual_embeddings_type=visual_embeddings_type,
            attention_mask=attention_mask,
        )

    def get_image_embedding(
        self,
        visual_embeddings: Tensor,
        visual_embeddings_type: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        if self.bypass_transformer:
            return self.embeddings.projection(visual_embeddings)
        else:
            return self.forward(
                visual_embeddings=visual_embeddings,
                visual_embeddings_type=visual_embeddings_type,
                attention_mask=attention_mask,
            )

    def get_text_embedding(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        return self.forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )


class FashionViLForPretraining(nn.Module):
    pass


class FashionViLForClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        bert_model_name = getattr(config, "bert_model_name", "bert-base-uncased")
        bert_config = BertConfig.from_dict(OmegaConf.to_container(config, resolve=True))

        self.config = config
        self.num_labels = config.num_labels
        self.bert = FashionViLBase.from_pretrained(
            bert_model_name,
            config=bert_config,
            cache_dir=os.path.join(get_mmf_cache_dir(), "distributed_{}".format(-1)),
            visual_embedding_dim=config.visual_embedding_dim,
            output_attentions=config.output_attentions,
            output_hidden_states=config.output_hidden_states,
        )
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Sequential(
            BertPredictionHeadTransform(self.bert.config),
            nn.Linear(self.bert.config.hidden_size, config.num_labels),
        )

    def forward(
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
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.contiguous().view(-1, self.num_labels)
        output_dict = {"scores": reshaped_logits}
        return output_dict


class FashionViLForContrastive(nn.Module):
    def __init__(self, config):
        super().__init__()
        bert_model_name = getattr(config, "bert_model_name", "bert-base-uncased")
        bert_config = BertConfig.from_dict(OmegaConf.to_container(config, resolve=True))

        self.config = config
        self.bert = FashionViLBase.from_pretrained(
            bert_model_name,
            config=bert_config,
            cache_dir=os.path.join(get_mmf_cache_dir(), "distributed_{}".format(-1)),
            visual_embedding_dim=config.visual_embedding_dim,
            output_attentions=config.output_attentions,
            output_hidden_states=config.output_hidden_states,
        )
        self.norm_layer = NormalizationLayer()

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        visual_embeddings: Tensor,
        visual_embeddings_type: Tensor,
        text_attention_mask: Tensor,
        visual_attention_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        visual_embeddings, _, _ = self.bert.get_image_embedding(
            visual_embeddings, visual_embeddings_type, visual_attention_mask
        )
        visual_embeddings = visual_embeddings.mean(dim=1)
        visual_embeddings = self.norm_layer(visual_embeddings)

        text_embeddings, _, _ = self.bert.get_text_embedding(
            input_ids,
            token_type_ids,
            text_attention_mask,
        )
        text_embeddings = text_embeddings[:, 0]
        text_embeddings = self.norm_layer(text_embeddings)

        output_dict = {
            "scores": visual_embeddings,
            "targets": text_embeddings,
        }
        return output_dict


class FashionViLForComposition(nn.Module):
    def __init__(self, config):
        super().__init__()
        bert_model_name = getattr(config, "bert_model_name", "bert-base-uncased")
        bert_config = BertConfig.from_dict(OmegaConf.to_container(config, resolve=True))

        self.config = config
        self.bert = FashionViLBase.from_pretrained(
            bert_model_name,
            config=bert_config,
            cache_dir=os.path.join(get_mmf_cache_dir(), "distributed_{}".format(-1)),
            visual_embedding_dim=config.visual_embedding_dim,
            output_attentions=config.output_attentions,
            output_hidden_states=config.output_hidden_states,
        )
        self.norm_layer = NormalizationLayer()

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        tar_visual_embeddings: Tensor,
        tar_visual_embeddings_type: Tensor,
        ref_visual_embeddings: Tensor,
        ref_visual_embeddings_type: Tensor,
        comp_attention_mask: Tensor,
        visual_attention_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        tar_embeddings, _, _ = self.bert.get_image_embedding(
            tar_visual_embeddings, tar_visual_embeddings_type, visual_attention_mask
        )
        tar_embeddings = tar_embeddings.mean(dim=1)
        tar_embeddings = self.norm_layer(tar_embeddings)

        comp_embeddings, _, _ = self.bert.get_joint_embedding(
            input_ids,
            token_type_ids,
            ref_visual_embeddings,
            ref_visual_embeddings_type,
            comp_attention_mask,
        )
        num_visual_tokens = tar_visual_embeddings.shape[1]
        comp_embeddings = comp_embeddings[:, -num_visual_tokens:].mean(dim=1)
        comp_embeddings = self.norm_layer(comp_embeddings)

        output_dict = {
            "scores": comp_embeddings,
            "targets": tar_embeddings,
        }
        return output_dict


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

    def flatten(
        self,
        sample_list: Dict[str, Tensor],
        to_be_flattened: List[str],
        to_be_flattened_dim: List[str],
    ) -> Dict[str, Tensor]:
        for key in to_be_flattened:
            # Make sure these keys are present or otherwise set these keys to None
            sample_list[key] = transform_to_batch_sequence(sample_list[key])
        for key in to_be_flattened_dim:
            sample_list[key] = transform_to_batch_sequence_dim(sample_list[key])
        return sample_list

    def add_post_flatten_params(
        self, sample_list: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        if self.training_head_type == "composition":
            b, l, _ = sample_list["ref_image"].shape
            device = sample_list["ref_image"].device

            sample_list["tar_visual_embeddings_type"] = torch.zeros(
                (b, l), device=device
            ).long()
            sample_list["ref_visual_embeddings_type"] = torch.zeros(
                (b, l), device=device
            ).long()
            sample_list["comp_attention_mask"] = torch.cat(
                (sample_list["input_mask"], torch.ones((b, l), device=device).long()),
                dim=-1,
            )
            sample_list["visual_attention_mask"] = torch.ones(
                (b, l), device=device
            ).long()
        else:
            b, l, _ = sample_list["image"].shape
            device = sample_list["image"].device
            sample_list["visual_embeddings_type"] = torch.zeros(
                (b, l), device=device
            ).long()
            if self.training_head_type == "classification":
                sample_list["attention_mask"] = torch.cat(
                    (
                        sample_list["input_mask"],
                        torch.ones((b, l), device=device).long(),
                    ),
                    dim=-1,
                )
        return sample_list

    def get_optimizer_parameters(self, config):
        return get_optimizer_parameters_for_bert(self.model, config)

    def flatten_for_bert(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        to_be_flattened = ["input_ids", "segment_ids"]
        if self.training_head_type == "composition":
            to_be_flattened_dim = ["ref_image", "tar_image"]
        else:
            to_be_flattened_dim = ["image"]

        # We want to convert everything into: batch x sequence_length x (dim).
        flattened = self.flatten(sample_list, to_be_flattened, to_be_flattened_dim)
        return flattened

    def update_sample_list_based_on_head(
        self, sample_list: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        # We don't need image mask here coz we are using grid features
        return sample_list

    def add_custom_params(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return sample_list

    def forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # sample_list = self.update_sample_list_based_on_head(sample_list)
        # sample_list = self.add_custom_params(sample_list)
        sample_list = self.flatten_for_bert(sample_list)
        sample_list = self.add_post_flatten_params(sample_list)

        if self.training_head_type == "composition":
            output_dict = self.model(
                input_ids=sample_list["input_ids"],
                token_type_ids=sample_list["segment_ids"],
                tar_visual_embeddings=sample_list["tar_image"],
                tar_visual_embeddings_type=sample_list["tar_visual_embeddings_type"],
                ref_visual_embeddings=sample_list["ref_image"],
                ref_visual_embeddings_type=sample_list["ref_visual_embeddings_type"],
                comp_attention_mask=sample_list["comp_attention_mask"],
                visual_attention_mask=sample_list["visual_attention_mask"],
            )
        elif self.training_head_type == "contrastive":
            output_dict = self.model(
                input_ids=sample_list["input_ids"],
                token_type_ids=sample_list["segment_ids"],
                visual_embeddings=sample_list["image"],
                visual_embeddings_type=sample_list["visual_embeddings_type"],
                text_attention_mask=sample_list["input_mask"],
            )
        else:
            output_dict = self.model(
                input_ids=sample_list["input_ids"],
                token_type_ids=sample_list["segment_ids"],
                visual_embeddings=sample_list["image"],
                visual_embeddings_type=sample_list["visual_embeddings_type"],
                attention_mask=sample_list["attention_mask"],
            )

        return output_dict
