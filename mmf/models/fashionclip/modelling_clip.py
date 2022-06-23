# Copyright (c) Facebook, Inc. and its affiliates.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextConfig, CLIPVisionConfig
from transformers.modeling_clip import (
    CLIPConfig,
    CLIPAttention,
    CLIPMLP,
    CLIPEncoder,
    CLIPTextTransformer,
    CLIPVisionTransformer,
    CLIPVisionEmbeddings,
    CLIPTextEmbeddings,
    CLIPModel,
)


@dataclass
class CLIPAdapterConfig:
    def __init__(
        self,
        freeze: bool = False,
        adapter_name: str = None,
        bottleneck: int = 64,
        dropout: float = 0.0,
    ):
        self.freeze = freeze
        self.adapter_name = adapter_name
        self.bottleneck = bottleneck
        self.dropout = dropout


class Adapter(nn.Module):
    def __init__(
        self,
        d_model,
        bottleneck,
        dropout=0.0,
        adapter_scalar="learnable_scalar",
        adapter_layernorm_option="in",
    ):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        # _before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == "in":
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = F.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == "out":
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


class CLIPEncoderLayerWithAdapter(nn.Module):
    def __init__(self, config: CLIPConfig, adapter_config: CLIPAdapterConfig = None):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

        self.adapter_config = adapter_config
        if self.adapter_config is not None:
            if self.adapter_config.freeze:
                self.freeze()
            if self.adapter_config.adapter_name == "scaled_pa":
                self.adapt_mlp = Adapter(
                    self.embed_dim,
                    self.adapter_config.bottleneck,
                    self.adapter_config.dropout,
                )

    @staticmethod
    def _freeze(module):
        for p in module.parameters():
            p.requires_grad = False

    def freeze(self):
        self._freeze(self.self_attn)
        self._freeze(self.layer_norm1)
        self._freeze(self.mlp)
        self._freeze(self.layer_norm2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:

        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        if self.adapter_config is None:
            hidden_states = self.layer_norm2(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
        else:
            adapt_hidden_states = self.adapt_mlp(hidden_states, add_residual=False)
            hidden_states = self.layer_norm2(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states + adapt_hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class CLIPEncoderWithAdapter(CLIPEncoder):
    def __init__(self, config: CLIPConfig, adapter_config: CLIPAdapterConfig = None):
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [
                CLIPEncoderLayerWithAdapter(config, adapter_config)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False


class CLIPTextTransformerWithAdapter(CLIPTextTransformer):
    def __init__(
        self, config: CLIPTextConfig, adapter_config: CLIPAdapterConfig = None
    ):
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoderWithAdapter(config, adapter_config)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

        if adapter_config is not None:
            if adapter_config.freeze:
                self.freeze()

    @staticmethod
    def _freeze(module):
        for p in module.parameters():
            p.requires_grad = False

    def freeze(self):
        self._freeze(self.embeddings)
        self._freeze(self.final_layer_norm)


class CLIPVisionTransformerWithAdapter(CLIPVisionTransformer):
    def __init__(
        self, config: CLIPVisionConfig, adapter_config: CLIPAdapterConfig = None
    ):
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = CLIPVisionEmbeddings(config)
        # Typo of huggingface
        self.pre_layrnorm = nn.LayerNorm(embed_dim)
        self.encoder = CLIPEncoderWithAdapter(config, adapter_config)
        self.post_layernorm = nn.LayerNorm(embed_dim)

        if adapter_config is not None:
            if adapter_config.freeze:
                self.freeze()

    @staticmethod
    def _freeze(module):
        for p in module.parameters():
            p.requires_grad = False

    def freeze(self):
        self._freeze(self.embeddings)
        self._freeze(self.pre_layrnorm)
        self._freeze(self.post_layernorm)


class CLIPModelWithAdapter(CLIPModel):
    config_class = CLIPConfig

    def __init__(self, config: CLIPConfig, adapter_config: CLIPAdapterConfig = None):
        super().__init__(config)

        assert isinstance(config.text_config, CLIPTextConfig)
        assert isinstance(config.vision_config, CLIPVisionConfig)

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = CLIPTextTransformerWithAdapter(text_config, adapter_config)
        self.vision_model = CLIPVisionTransformerWithAdapter(
            vision_config, adapter_config
        )

        self.visual_projection = nn.Linear(
            self.vision_embed_dim, self.projection_dim, bias=False
        )
        self.text_projection = nn.Linear(
            self.text_embed_dim, self.projection_dim, bias=False
        )
        self.logit_scale = nn.Parameter(
            torch.ones([]) * self.config.logit_scale_init_value
        )

        # Initialize weights and apply final processing
        self.post_init()

        if adapter_config is not None:
            if adapter_config.freeze:
                self.freeze()

    @staticmethod
    def _freeze(module):
        for p in module.parameters():
            p.requires_grad = False

    def freeze(self):
        self._freeze(self.visual_projection)
        self._freeze(self.text_projection)
        self.logit_scale.requires_grad = False
