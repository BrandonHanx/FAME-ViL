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


def ogd(p_grad, name, gradient_dict):
    if p_grad is None or not torch.any(p_grad):
        return p_grad
    operate_task = gradient_dict["operate_task"]
    regular_grads = []
    for k, v in gradient_dict.items():
        if k == operate_task or k == "operate_task":
            continue
        regular_grad = v.get(name, None)
        if regular_grad is not None and torch.any(regular_grad):
            regular_grads.append(v[name])
    for regular_grad in regular_grads:
        regular_grad_norm = regular_grad / torch.linalg.norm(regular_grad)
        p_grad_norm = p_grad / torch.linalg.norm(p_grad)
        if torch.dot(p_grad_norm.flatten(), regular_grad_norm.flatten()) < 0:
            p_grad = p_grad - regular_grad_norm * torch.dot(
                p_grad.flatten(), regular_grad_norm.flatten()
            )
    gradient_dict[operate_task][name] = p_grad
    return p_grad
