# Copyright 2026 The RLinf Authors / sigma project.
# SPDX-License-Identifier: Apache-2.0
"""LWD = QAM + DIVL + chunked Q (Fleet RL).

Reference: arXiv:2605.00416 ("Learning While Deploying").
Concrete equation references (LWD paper):

  Eq. 9   L_QAM(theta)  -- adjoint matching policy extraction
  Eq. 10  tilde_g_1     -- adjoint init from -grad_a Q / lambda
  Eq. 12  L_V(psi)      -- distributional value learning (DIVL)
                          L_V = E[-log p_psi(Q_phi_bar(s, a) | s)]
  Eq. 15  L_Q(phi)      -- critic TD loss with quantile target
                          y_Q = r_t + gamma^H * Quant_tau(V_psi(s_{t+H}))
  Eq. 19  chunked y_Q   -- multi-step (n=10) target for sparse rewards
                          y_Q = sum_{i=0..n-1} gamma^{iH} r_{t+iH}
                                + gamma^{nH} Quant_tau(V_psi(s_{t+nH}))

This module provides loss helpers; the full update loop lives in
fsdp_qam_policy_worker.py. We deliberately keep the *interface* identical
to rlinf.algorithms.chunked_q so the LWD worker can swap the TD target.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from rlinf.algorithms.adjoint_matching import compute_adjoint_matching_loss


# ---------------------------------------------------------------------------
# DIVL: Distributional Implicit Value Learning  (Eq. 12)
# ---------------------------------------------------------------------------
def compute_divl_value_loss(
    value_logits: torch.Tensor,   # [B, K] categorical logits over K atoms (C51-style)
    q_target: torch.Tensor,       # [B, 1] target Q from frozen target critic
    v_min: float,
    v_max: float,
    num_atoms: int,
) -> torch.Tensor:
    """Categorical cross-entropy DIVL loss (Eq. 12, LWD).

    L_V(psi) = E[-log p_psi(Q_phi_bar(s, a) | s)]

    Implementation: project scalar q_target onto K equally-spaced atoms in
    [v_min, v_max] (C51-style), then standard CE. This is the *implicit*
    distributional analog of expectile regression in IQL (cf. LWD Sec 4.1).

    Args:
        value_logits: raw logits [B, K] from V_psi(s).
        q_target:     scalar TD target [B, 1] (use target critic, no_grad).
        v_min, v_max: support range.
        num_atoms:    K.

    Returns:
        scalar CE loss.
    """
    assert value_logits.shape[-1] == num_atoms
    delta_z = (v_max - v_min) / max(num_atoms - 1, 1)
    support = torch.linspace(v_min, v_max, num_atoms, device=value_logits.device)

    q_clamped = q_target.clamp(v_min, v_max)
    b = (q_clamped - v_min) / delta_z       # [B, 1] continuous index
    lo = b.floor().long().clamp(0, num_atoms - 1)
    hi = b.ceil().long().clamp(0, num_atoms - 1)
    target_dist = torch.zeros_like(value_logits)
    target_dist.scatter_add_(1, lo, hi.float() + 1.0 - b)  # interpolate
    target_dist.scatter_add_(1, hi, b - lo.float())
    # Edge case: lo == hi (q_target is on grid).
    same = (lo == hi).float()
    target_dist.scatter_add_(1, lo, same * (1.0 - target_dist.gather(1, lo)))

    log_p = F.log_softmax(value_logits, dim=-1)
    return -(target_dist.detach() * log_p).sum(dim=-1).mean()


def quantile_value(value_logits: torch.Tensor, tau: float, v_min: float, v_max: float) -> torch.Tensor:
    """Read tau-quantile out of categorical V distribution (LWD Eq. 15)."""
    num_atoms = value_logits.shape[-1]
    support = torch.linspace(v_min, v_max, num_atoms, device=value_logits.device)
    p = F.softmax(value_logits, dim=-1)
    cdf = p.cumsum(dim=-1)
    idx = (cdf >= tau).float().argmax(dim=-1, keepdim=True)
    return support[idx.squeeze(-1)].unsqueeze(-1)


# ---------------------------------------------------------------------------
# Chunked Q target  (Eq. 15 / Eq. 19, LWD)
# ---------------------------------------------------------------------------
def compute_lwd_chunked_q_target(
    rewards: torch.Tensor,          # [B, n, H] rewards within n consecutive H-chunks
    v_quantile_terminal: torch.Tensor,  # [B, 1] Quant_tau V_psi(s_{t+nH})
    gamma: float,
    H: int,
    n: int = 10,
    dones: Optional[torch.Tensor] = None,  # [B, n, H] terminal mask
) -> torch.Tensor:
    """Multi-step chunked TD target (Eq. 19, LWD).

    y_Q = sum_{i=0..n-1} gamma^{iH} r_{t+iH}  +  gamma^{nH} * Quant_tau(V(s_{t+nH}))

    NOTE: rewards in chunk i are summed (chunk-level reward) before the
    gamma^{iH} discount is applied -- matches LWD's chunk-level reward
    aggregation.
    """
    B = rewards.shape[0]
    chunk_rewards = rewards.sum(dim=-1)  # [B, n]
    if dones is not None:
        # zero out chunks after the first terminal
        chunk_done = dones.any(dim=-1).cumsum(dim=-1).clamp_max(1)
        chunk_rewards = chunk_rewards * (1.0 - chunk_done.float())
    discounts = gamma ** (H * torch.arange(n, device=rewards.device, dtype=rewards.dtype))
    discounted_sum = (chunk_rewards * discounts.unsqueeze(0)).sum(dim=-1, keepdim=True)
    bootstrap = (gamma ** (H * n)) * v_quantile_terminal
    if dones is not None:
        any_terminal = dones.any(dim=(-1, -2), keepdim=True).squeeze(-1).float()
        bootstrap = bootstrap * (1.0 - any_terminal)
    return discounted_sum + bootstrap


# ---------------------------------------------------------------------------
# Combined LWD learner step (Algorithm 2, LWD)
# ---------------------------------------------------------------------------
def compute_lwd_losses(
    *,
    # DIVL inputs
    value_logits: torch.Tensor,
    q_target_for_value: torch.Tensor,
    # Critic TD inputs
    q_pred: torch.Tensor,
    q_target: torch.Tensor,
    # QAM inputs
    velocity_pred: torch.Tensor,
    a_w: torch.Tensor,
    sigma_w: torch.Tensor,
    q_action_grad: torch.Tensor,
    # config
    v_min: float = -1.0,
    v_max: float = 1.0,
    num_atoms: int = 51,
    qam_lambda: float = 1.0,
    qam_weight: float = 1.0,
    divl_weight: float = 1.0,
    critic_weight: float = 1.0,
):
    """One Learner step (LWD Algorithm 2) -- returns dict of losses."""
    l_divl = compute_divl_value_loss(value_logits, q_target_for_value, v_min, v_max, num_atoms)
    l_q = F.mse_loss(q_pred, q_target.detach())
    l_qam = compute_adjoint_matching_loss(
        velocity_pred=velocity_pred,
        a_w=a_w,
        sigma_w=sigma_w,
        q_action_grad=q_action_grad,
        lam=qam_lambda,
    )
    total = critic_weight * l_q + divl_weight * l_divl + qam_weight * l_qam
    return {
        "loss/total": total,
        "loss/critic_q": l_q.detach(),
        "loss/divl": l_divl.detach(),
        "loss/qam": l_qam.detach(),
    }
