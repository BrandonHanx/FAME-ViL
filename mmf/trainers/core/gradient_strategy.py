# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import torch.nn.functional as F


def sum(p_grad, name, gradient_list):
    for grad in gradient_list:
        p_grad = (
            p_grad + grad.get(name, 0)
            if p_grad is not None and grad.get(name, 0) is not None
            else p_grad
        )
    return p_grad


def imtlg(p_grad, name, gradient_list):
    raw_grad = []
    raw_grad_norm = []
    for grad in gradient_list:
        if name in grad and grad[name] is not None:
            flatten_grad = torch.flatten(grad[name])
            raw_grad.append(flatten_grad)
            raw_grad_norm.append(F.normalize(flatten_grad, p=2, dim=-1))
    if len(raw_grad) == 0:
        return p_grad
    if len(raw_grad) == 1:
        return raw_grad[0]

    G = torch.stack(raw_grad)
    D = G[0] - G[1:]
    U = torch.stack(raw_grad_norm)
    U = U[0] - U[1:]
    first_element = torch.matmul(G[0], U.t())
    try:
        second_element = torch.inverse(torch.matmul(D, U.t()))
    except RuntimeError:
        # workaround for cases where matrix is singular
        second_element = torch.inverse(
            torch.eye(len(raw_grad) - 1, device=G.device) * 1e-8
            + torch.matmul(D, U.t())
        )

    alpha_ = torch.matmul(first_element, second_element)
    alpha = torch.stack(((1 - alpha_.sum()).unsqueeze(-1), alpha_))
    return torch.sum(G * alpha, dim=0).reshape(p_grad.shape)
