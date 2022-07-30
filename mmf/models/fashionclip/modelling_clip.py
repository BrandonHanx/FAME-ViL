# Copyright (c) Facebook, Inc. and its affiliates.

import math
import random
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
    _expand_mask,
)


@dataclass
class CLIPAdapterConfig:
    def __init__(
        self,
        freeze: bool = False,
        adapter_name: str = None,
        bottleneck: int = 64,
        dropout: float = 0.0,
        enable_xattn: bool = False,
        cross_dropout: float = 0.0,
        share_cross: bool = False,
        share_adapter: bool = False,
    ):
        self.freeze = freeze
        self.adapter_name = adapter_name
        self.bottleneck = bottleneck
        self.dropout = dropout
        self.enable_xattn = enable_xattn
        self.cross_dropout = cross_dropout
        self.share_cross = share_cross
        self.share_adapter = share_adapter


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


class ConvPass(nn.Module):
    def __init__(
        self,
        d_model,
        bottleneck,
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

        self.down_conv = nn.Conv1d(self.n_embd, self.down_size, 1, stride=1, padding=0)
        self.non_linear_func = nn.GELU()
        self.in_conv = nn.Conv2d(self.down_size, self.down_size, 3, stride=1, padding=1)
        self.up_conv = nn.Conv1d(self.down_size, self.n_embd, 1, stride=1, padding=0)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == "in":
            x = self.adapter_layer_norm_before(x)

        b, _, d = x.shape
        down = self.down_conv(x.reshape(b, d, -1))
        down = self.non_linear_func(down)
        down_cls = self.in_conv(down[:, :, 0].reshape(b, self.down_size, 1, 1)).squeeze(
            -1
        )
        down_ctx = self.in_conv(
            down[:, :, 1:].reshape(b, self.down_size, 14, 14)
        ).reshape(b, self.down_size, -1)
        down = torch.cat([down_cls, down_ctx], dim=2)
        down = self.non_linear_func(down)
        up = self.up_conv(down).reshape(b, -1, d)

        up = up * self.scale

        if self.adapter_layernorm_option == "out":
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


class CLIPCrossAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim = config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.head_size)
        self.key = nn.Linear(ctx_dim, self.head_size)
        self.value = nn.Linear(ctx_dim, self.head_size)

        self.dropout = nn.Dropout(config.attention_dropout)
        self.scale = nn.Parameter(torch.zeros(1))
        self.query_layernorm = nn.LayerNorm(config.hidden_size)
        self.context_layernorm = nn.LayerNorm(ctx_dim)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, context, attention_mask=None, output_attentions=False
    ):
        hidden_states = self.query_layernorm(hidden_states)
        context = self.context_layernorm(context)
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key"
        # to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is
        # (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.head_size,)
        context_layer = context_layer.view(new_context_layer_shape) * self.scale

        outputs = (
            (context_layer, attention_probs)
            if output_attentions
            else (context_layer, None)
        )
        return outputs


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
            if (
                self.adapter_config.adapter_name == "convpass"
                and config.hidden_size == 768
            ):
                self.adapt_mlp = ConvPass(
                    self.embed_dim,
                    self.adapter_config.bottleneck,
                )
            else:
                self.adapt_mlp = Adapter(
                    self.embed_dim,
                    self.adapter_config.bottleneck,
                    self.adapter_config.dropout,
                )
            if self.adapter_config.enable_xattn:
                # FIXME: monkey patching
                self.cross_attn = CLIPCrossAttention(
                    config, 768 if config.hidden_size == 512 else 512
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

    def forward_self_attn(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        return hidden_states, attn_weights

    def forward_cross_attn(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        hidden_states, attn_weights = self.cross_attn(
            hidden_states=hidden_states,
            context=context,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        return hidden_states, attn_weights

    def forward_originmlp(self, hidden_states: torch.Tensor) -> torch.FloatTensor:
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return hidden_states

    def forward_adaptmlp(self, hidden_states: torch.Tensor) -> torch.FloatTensor:
        adapt_hidden_states = self.adapt_mlp(hidden_states, add_residual=False)
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return hidden_states + adapt_hidden_states

    def forward_mlp(self, hidden_states: torch.Tensor) -> torch.FloatTensor:
        if self.adapter_config is None:
            hidden_states = self.forward_originmlp(hidden_states)
        else:
            hidden_states = self.forward_adaptmlp(hidden_states)
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:

        residual = hidden_states

        hidden_states, attn_weights = self.forward_self_attn(
            hidden_states, attention_mask, causal_attention_mask, output_attentions
        )

        hidden_states = hidden_states + residual
        residual = hidden_states

        hidden_states = self.forward_mlp(hidden_states)

        hidden_states = hidden_states + residual

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
        if adapter_config is not None:
            if adapter_config.share_adapter:
                for i in range(config.num_hidden_layers):
                    self.layers[i].adapt_mlp = self.layers[0].adapt_mlp
            if adapter_config.share_cross:
                for i in range(config.num_hidden_layers):
                    self.layers[i].cross_attn = self.layers[0].cross_attn
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
        self.adapter_config = adapter_config

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

    def get_cross_attn_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # output_attentions: Optional[bool] = None,
    ) -> torch.FloatTensor:

        v_hidden_states = self.vision_model.embeddings(pixel_values)
        v_hidden_states = self.vision_model.pre_layrnorm(v_hidden_states)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        t_hidden_states = self.text_model.embeddings(
            input_ids=input_ids, position_ids=position_ids
        )
        bsz, seq_len = input_shape
        causal_attention_mask = self.text_model._build_causal_attention_mask(
            bsz, seq_len
        ).to(t_hidden_states.device)
        if attention_mask is not None:
            self_attn_mask = _expand_mask(attention_mask, t_hidden_states.dtype)
            cross_attn_mask = _expand_mask(
                attention_mask, t_hidden_states.dtype, v_hidden_states.shape[1]
            )

        for v_layer, t_layer in zip(
            self.vision_model.encoder.layers, self.text_model.encoder.layers
        ):
            v_residual = v_hidden_states
            t_residual = t_hidden_states
            v_hidden_states, _ = v_layer.forward_self_attn(v_hidden_states)
            t_hidden_states, _ = t_layer.forward_self_attn(
                t_hidden_states, self_attn_mask, causal_attention_mask
            )

            if random.random() < self.adapter_config.cross_dropout and self.training:
                pass
            else:
                vt_hidden_states, _ = v_layer.forward_cross_attn(
                    v_hidden_states, t_hidden_states, cross_attn_mask
                )
                tv_hidden_states, _ = t_layer.forward_cross_attn(
                    t_hidden_states, v_hidden_states
                )
                v_hidden_states = v_hidden_states + vt_hidden_states
                t_hidden_states = t_hidden_states + tv_hidden_states

            v_hidden_states = v_residual + v_hidden_states
            t_hidden_states = t_residual + t_hidden_states
            v_residual = v_hidden_states
            t_residual = t_hidden_states

            v_hidden_states = v_layer.forward_mlp(v_hidden_states)
            t_hidden_states = t_layer.forward_mlp(t_hidden_states)
            v_hidden_states = v_residual + v_hidden_states
            t_hidden_states = t_residual + t_hidden_states

        v_features = v_hidden_states[:, 0, :]
        v_features = self.vision_model.post_layernorm(v_features)
        v_features = self.visual_projection(v_features)

        t_hidden_states = self.text_model.final_layer_norm(t_hidden_states)
        t_features = t_hidden_states[
            torch.arange(t_hidden_states.shape[0]), input_ids.argmax(dim=-1)
        ]
        t_features = self.text_projection(t_features)

        return v_features, t_features
