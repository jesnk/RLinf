# Copyright 2026 The RLinf Authors / sigma project.
# SPDX-License-Identifier: Apache-2.0
"""Adjoint Matching loss for QAM (Q-learning with Adjoint Matching).

Reference: Q-learning with Adjoint Matching, Li & Levine, arXiv:2601.14234.
Concrete eq. references taken from LWD (arXiv 2605.00416, Sec. 4.2):

    L_QAM(theta) = E_{(s, a^w, w) ~ D, eps}[
        || 2 * f_delta(s, a^w, w) / sigma_w
           + sigma_w * tilde_g_w ||_2^2
    ]                                                          (Eq. 9, LWD)

    where the adjoint state is initialized from the critic gradient:
        tilde_g_1 = - grad_a [ Q_phi(s, a^1) / lambda ]         (Eq. 10, LWD)

The adjoint state at intermediate flow time w in [0, 1) is propagated by the
backward adjoint ODE -- but in QAM the *trick* is that we DO NOT differentiate
through the denoising trajectory.  Instead, the adjoint is treated as a
*regression target* that the velocity field f_theta(=f_delta + base) must
match in L2.  This makes QAM a step-wise supervised objective free from
backprop-through-denoising, which is the polar opposite of sigma's pathwise
SAC-Flow update (sigma differentiates Q gradient through the entire
denoising chain; QAM regresses a single-step target).

This module implements `compute_adjoint_matching_loss`. The full adjoint
ODE solver is left as a TODO (Phase 2). For Phase 1 we use the *terminal*
adjoint (w=1) directly as the regression target, matching ALOE / advantage-
weighted score matching surrogates that we already discuss in the sigma
related work.
"""
from __future__ import annotations

from typing import Optional

import torch


def compute_adjoint_matching_loss(
    velocity_pred: torch.Tensor,    # [B, D] f_theta(s, a^w, w) -- predicted flow velocity
    a_w: torch.Tensor,              # [B, D] noisy action a^w along the flow trajectory
    sigma_w: torch.Tensor,          # [B, 1] flow-matching schedule sigma_w
    q_action_grad: torch.Tensor,    # [B, D] grad_a Q_phi(s, a)  (computed with stop-grad on critic params)
    advantage: Optional[torch.Tensor] = None,  # [B, 1] optional advantage weighting (sigma extension)
    lam: float = 1.0,               # lambda temperature in QAM (Eq. 10, LWD)
    reduction: str = "mean",
) -> torch.Tensor:
    """QAM adjoint matching loss (Eq. 9, LWD; QAM Eq. (4)-(7) Levine 2026).

    Args:
        velocity_pred:   f_delta(s, a^w, w) -- the *residual* velocity that the
                         policy is learning. Caller is responsible for setting
                         requires_grad correctly so that loss.backward() touches
                         only theta (not the critic parameters phi).
        a_w:             noisy action sample at flow-time w.
        sigma_w:         flow-matching noise scale at w.
        q_action_grad:   action gradient of the *target* Q-critic, computed
                         under torch.no_grad() w.r.t. critic parameters.
                         Shape must match velocity_pred and a_w.
        advantage:       optional per-sample weight (sigma's chunked advantage).
        lam:             temperature lambda from QAM Eq. (10).
        reduction:       'mean' | 'sum' | 'none'.

    Returns:
        scalar loss (or [B] if reduction == 'none').

    Reference:
      L_QAM = E[ || 2 * f_delta / sigma_w  +  sigma_w * tilde_g_w ||^2 ]
      tilde_g_1 = - grad_a Q_phi(s, a^1) / lam
    """
    assert velocity_pred.shape == a_w.shape == q_action_grad.shape, (
        f"shape mismatch: f={velocity_pred.shape} a_w={a_w.shape} "
        f"qg={q_action_grad.shape}"
    )
    if sigma_w.ndim == 1:
        sigma_w = sigma_w.unsqueeze(-1)

    # Adjoint state initialization (Eq. 10): tilde_g_1 = -grad_a Q / lambda.
    # TODO(sigma-phase2): integrate the backward adjoint ODE to obtain
    # tilde_g_w for intermediate w; for now we use the terminal adjoint
    # which is the dominant contribution in practice (cf. QAM Sec 4.1).
    tilde_g_w = -q_action_grad / lam

    # Eq. 9: residual = 2 f_delta / sigma_w + sigma_w * tilde_g_w
    residual = 2.0 * velocity_pred / sigma_w + sigma_w * tilde_g_w
    per_sample = residual.pow(2).sum(dim=-1, keepdim=True)  # [B, 1]

    if advantage is not None:
        # sigma extension: chunked advantage weighting (cf. ALOE advantage-
        # weighted CFM; reduces variance in heterogeneous fleet data).
        per_sample = per_sample * advantage.detach()

    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    return per_sample.squeeze(-1)


def compute_critic_action_grad(
    critic_fn,
    obs,
    action: torch.Tensor,
) -> torch.Tensor:
    """Compute grad_a Q_phi(s, a) with stop-grad on critic parameters.

    QAM uses the *target* critic to provide a stable adjoint signal (cf.
    LWD Algorithm 2 line 8). Critic params are frozen during the actor
    pass; only the action tensor receives gradients.

    Args:
        critic_fn: callable returning Q-values [B, K] or [B, 1].
        obs:       observation (forwarded as-is).
        action:    [B, D] action tensor (a clone with requires_grad=True is made).

    Returns:
        grad_a Q  with shape [B, D].
    """
    a = action.detach().clone().requires_grad_(True)
    q = critic_fn(obs, a)
    if q.ndim > 1 and q.shape[-1] > 1:
        q = q.min(dim=-1, keepdim=True).values  # pessimistic LCB
    grad = torch.autograd.grad(q.sum(), a, create_graph=False, retain_graph=False)[0]
    return grad.detach()
