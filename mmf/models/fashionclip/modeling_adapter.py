# Copyright (c) Facebook, Inc. and its affiliates.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x, add_residual=False, residual=None):
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


class NASAdapter(nn.Module):
    def __init__(
        self,
        d_model,
        bottleneck,
        dropout=0.0,
        num_adapters=2,
        adapter_name_list=None,
        initial_tau=1.0,
        gamma=0.9999,
        hard=False,
    ):
        super().__init__()
        self.adapters = nn.ModuleList(
            [Adapter(d_model, bottleneck, dropout) for _ in range(num_adapters)]
        )
        if adapter_name_list is None:
            self.controller = nn.Parameter(torch.rand(num_adapters))
        else:
            self.controller = nn.ParameterDict()
            for name in adapter_name_list:
                self.controller[name] = nn.Parameter(torch.rand(num_adapters))
        self.tau = initial_tau
        self.gamma = gamma
        self.hard = hard

    def get_controller_entropy(self, task_name=None):
        if isinstance(self.controller, nn.ParameterDict) and task_name is not None:
            entropy = torch.special.entr(F.softmax(self.controller[task_name]))
        else:
            entropy = torch.special.entr(F.softmax(self.controller))
        return torch.mean(entropy)

    def forward(self, x, task_name=None):
        if self.training:
            if isinstance(self.controller, nn.ParameterDict) and task_name is not None:
                probs = F.gumbel_softmax(
                    self.controller[task_name], tau=self.tau, hard=self.hard
                )
            else:
                probs = F.gumbel_softmax(self.controller, tau=self.tau, hard=self.hard)
            if self.hard:
                index = torch.argmax(probs)
                x = self.adapters[index](x) * probs[index]
            else:
                x = torch.stack(
                    [a(x) * probs[i] for i, a in enumerate(self.adapters)]
                )  # soft version
                x = torch.sum(x, dim=0)
            self.tau = max(self.tau * self.gamma, 0.01)  # exponential decay
            return x
        if isinstance(self.controller, nn.ParameterDict) and task_name is not None:
            index = torch.argmax(self.controller[task_name])
        else:
            index = torch.argmax(self.controller)
        return self.adapters[index](x)


class NASAdapterPool(NASAdapter):
    def __init__(self, *args, **kwargs):
        d_model, _, dropout = args
        kwargs["num_adapters"] = 36
        super().__init__(*args, **kwargs)
        adapters = []
        # Manually design candidate pool
        for i in range(12):
            adapters.append(Adapter(d_model, 16, dropout))
            adapters.append(Adapter(d_model, 32, dropout))
            adapters.append(Adapter(d_model, 64, dropout))
        self.adapters = nn.ModuleList(adapters)


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
