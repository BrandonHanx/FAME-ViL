# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import torch.nn.functional as F


def simple_sum(p_grad, name, gradient_list):
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
    operate_task = gradient_dict["operate_task"]
    p_grad = gradient_dict[operate_task][name]
    if p_grad is not None and torch.any(p_grad):
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
        gradient_dict[operate_task][name] = (
            gradient_dict[operate_task][name] * 0.9 + p_grad.clone() * 0.1
        )
    return p_grad


def get_gradient_scales(val_scores, gamma=0.5, alpha=16, beta=0.7):
    baseline_scores = dict(itc=0.7, tgir=0.6, scr=0.9, cap=0.4)
    current_scores = dict(
        itc=val_scores["val/fashiongen/r@k_general/avg"],
        tgir=val_scores["val/fashioniq/r@k_fashioniq/avg"],
        scr=val_scores["val/fashiongen_cls/macro_f1"],
        cap=val_scores["val/fashiongen_cap/bleu4"],
    )
    relative_scores = dict()
    for k in baseline_scores.keys():
        relative_scores[k] = current_scores[k].get_latest() / baseline_scores[k]
    average_relative_score = sum(relative_scores.values()) / len(relative_scores)
    max_relative_score = max(relative_scores.values())
    gradient_scales = dict()
    for k in relative_scores.keys():
        bias_score = average_relative_score - relative_scores[k]
        sign = 1 if bias_score > 0 else -1
        gradient_scales[k] = 1 + sign * min(
            gamma, (max_relative_score ** alpha) * (abs(bias_score) ** beta)
        )
    return gradient_scales


def implicit(p_grad, operate_task, gradient_scales):
    if len(gradient_scales) == 0 or p_grad is None:
        return p_grad
    if operate_task == "fashiongen":
        p_grad = p_grad.clone() * gradient_scales["itc"]
    elif operate_task == "fashioniq":
        p_grad = p_grad.clone() * gradient_scales["tgir"]
    elif operate_task == "fashiongen_cls":
        p_grad = p_grad.clone() * gradient_scales["scr"]
    elif operate_task == "fashiongen_cap":
        p_grad = p_grad.clone() * gradient_scales["cap"]
    return p_grad
