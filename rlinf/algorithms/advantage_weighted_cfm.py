# Copyright 2026 The RLinf Authors / sigma project.
# SPDX-License-Identifier: Apache-2.0
"""Advantage-weighted Conditional Flow Matching loss for ALOE baseline.

Reference: ALOE (arXiv:2602.12691, AgiBot/HKUST/Fudan, 2026-02).
  pi0.5 (3B) flow VLA actor + Q-chunking critic + K-ensemble pessimistic LCB.

Concrete equations (paper Sec. 4):

  L_actor(theta) = E_D[ w(s_t, a_{t:t+h}, l) *
                        || eps - a_{t:t+h} - f_theta(a_tilde_{t:t+h}, s_t, l) ||_2^2 ]   (Eq. ALOE-actor)

  a_tilde_{t:t+h} = eta * a_{t:t+h} + (1 - eta) * eps,       eta ~ U[0, 1]   (Eq. ALOE-noisy)

  w(s_t, a_{t:t+h}, l) = exp( clip( A^pi(s_t, a_{t:t+h}, l) / beta,
                                    -eps_clip, +eps_clip ) )                 (Eq. ALOE-weight)

  A^pi(s_t, a_{t:t+h}, l) = Q_pess(s_t, a_{t:t+h}, l) - sg( V^pi(s_t, l) )   (Eq. ALOE-adv)

  Q_pess(s_t, a_{t:t+h}, l) = min_i Q_{phi_i}(s_t, a_{t:t+h}, l)             (K-ensemble LCB)

Stop-gradient discipline (paper Sec. 4): the advantage A^pi enters L_actor
ONLY through the scalar weight w (which is detached). The critic gradient
NEVER propagates pathwise into f_theta. This is the *direct anti-thesis*
of sigma's pathwise SAC-Flow update:

  - sigma:  grad_theta J_sigma = grad_theta Q( s, denoise_chain_theta(eps) )
            ==> backprops through the FULL flow ODE (multi-step BPTT).
  - ALOE:   grad_theta L_actor regresses f_theta to a *score-matching*
            target (eps - a) weighted by detached advantage
            ==> NO denoising backprop, single-step CFM surrogate.

This module mirrors the interface shape of `rlinf.algorithms.adjoint_matching`
and `rlinf.algorithms.fleet_rl_qam` for symmetry across the
sigma-phase1 baseline pool {QAM, LWD, ALOE}.
"""
from __future__ import annotations

from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Advantage -> weight transform  (Eq. ALOE-weight)
# ---------------------------------------------------------------------------
def compute_advantage_weight(
    advantage: torch.Tensor,    # [B, 1]  A^pi(s, a, l), already with sg on V
    beta: float = 0.5,          # advantage temperature; paper estimate ~0.5
    eps_clip: float = 5.0,      # symmetric clip on advantage / beta (pre-exp)
    w_max: Optional[float] = None,  # optional hard clip on the weight (post-exp)
    detach: bool = True,        # ensure no gradient flows into the critic
) -> torch.Tensor:
    """ALOE advantage-to-weight transform (clipped exponential).

        w = exp( clip( A / beta, -eps_clip, +eps_clip ) )

    Optionally clamp the resulting weight at w_max (a numerical safeguard;
    paper does not specify a value -- leaving None replicates the paper).

    Args:
        advantage: [B, 1] advantage tensor. The caller is expected to have
                   already applied stop-gradient on V_pi (Eq. ALOE-adv).
        beta:      advantage temperature.
        eps_clip:  symmetric clip on the *log-weight* before exponentiation.
        w_max:     optional hard upper bound on the weight (e.g. 20.0).
        detach:    if True (default), .detach() the weight so backprop into
                   advantage / critic is impossible. ALOE *requires* this.

    Returns:
        [B, 1] non-negative weight tensor (no_grad if detach=True).
    """
    a = advantage
    if a.ndim == 1:
        a = a.unsqueeze(-1)
    log_w = (a / max(beta, 1e-8)).clamp(-eps_clip, eps_clip)
    w = log_w.exp()
    if w_max is not None:
        w = w.clamp_max(w_max)
    if detach:
        w = w.detach()
    return w


# ---------------------------------------------------------------------------
# Advantage-weighted CFM actor loss  (Eq. ALOE-actor)
# ---------------------------------------------------------------------------
def compute_advantage_weighted_cfm_loss(
    epsilon: torch.Tensor,              # [B, D]  eps ~ N(0, I), the same noise used for a_tilde
    action: torch.Tensor,               # [B, D]  ground-truth action chunk a_{t:t+h} (flattened)
    predicted_velocity: torch.Tensor,   # [B, D]  f_theta(a_tilde, s, l)
    advantage: torch.Tensor,            # [B, 1]  A^pi (caller applied sg on V)
    beta: float = 0.5,
    eps_clip: float = 5.0,
    w_max: Optional[float] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """ALOE advantage-weighted CFM loss (Eq. ALOE-actor).

        L_actor = w * || eps - a - f_theta(a_tilde, s, l) ||^2

    Sign convention follows the user's spec:  the regression target is
    `eps - a` (NOT the standard CFM target `a - eps` of rectified flow).
    This matches the user-supplied paper formulation and is consistent with
    a flow that maps a -> eps in the forward direction.

    Args:
        epsilon, action, predicted_velocity: as above. All shape [B, D].
        advantage: [B, 1] advantage; caller is responsible for stop-grad on
                   the value baseline V (Eq. ALOE-adv).
        beta:      advantage temperature for the weight transform.
        eps_clip:  symmetric clip in log-weight space.
        w_max:     optional weight ceiling.
        reduction: 'mean' | 'sum' | 'none'.

    Returns:
        scalar loss (or [B] if reduction == 'none').
    """
    assert epsilon.shape == action.shape == predicted_velocity.shape, (
        f"shape mismatch: eps={epsilon.shape} a={action.shape} "
        f"f={predicted_velocity.shape}"
    )

    # Eq. ALOE-actor: residual = eps - a - f_theta
    target = epsilon - action
    residual = target - predicted_velocity
    per_sample = residual.pow(2).sum(dim=-1, keepdim=True)  # [B, 1]

    # Eq. ALOE-weight: clipped exp, with hard stop-gradient on advantage.
    w = compute_advantage_weight(
        advantage, beta=beta, eps_clip=eps_clip, w_max=w_max, detach=True
    )

    weighted = w * per_sample  # [B, 1]
    if reduction == "mean":
        return weighted.mean()
    if reduction == "sum":
        return weighted.sum()
    return weighted.squeeze(-1)


# ---------------------------------------------------------------------------
# Build a_tilde and a noise sample together  (Eq. ALOE-noisy)
# ---------------------------------------------------------------------------
def sample_aloe_flow_inputs(
    action: torch.Tensor,
    eta: Optional[torch.Tensor] = None,
):
    """Return (eps, eta, a_tilde) following ALOE Eq. ALOE-noisy.

        eta ~ U[0, 1],  eps ~ N(0, I)
        a_tilde = eta * a + (1 - eta) * eps

    Convenience helper so the worker can keep the noisy-action construction
    in a single line and avoid inconsistencies between actor and CFM loss.
    """
    B = action.shape[0]
    device, dtype = action.device, action.dtype
    if eta is None:
        eta = torch.rand((B, 1), device=device, dtype=dtype)
    eps = torch.randn_like(action)
    a_tilde = eta * action + (1.0 - eta) * eps
    return eps, eta, a_tilde


# ---------------------------------------------------------------------------
# Q-chunking-compatible advantage-weighted target
# ---------------------------------------------------------------------------
def compute_advantage_weighted_target(
    rewards: torch.Tensor,         # [B, H] per-step rewards in chunk
    next_q_pess: torch.Tensor,     # [B, 1] pessimistic LCB Q at s_{t+H}
    v_baseline: torch.Tensor,      # [B, 1] V^pi(s_t, l), stop-grad applied by caller
    gamma: float,
    H: int,
    dones: Optional[torch.Tensor] = None,
):
    """Build (Q-target, advantage) compatible with rlinf.algorithms.chunked_q.

    Returns:
        q_target: [B, 1] = sum_h gamma^h r_{t+h} + gamma^H * next_q_pess * (~done)
                  -- identical to compute_chunked_td_target so the LWD/sigma N1
                     critic can be reused unchanged.
        advantage: [B, 1] = q_target - sg(v_baseline)
                  -- ALOE Eq. ALOE-adv. v_baseline is detached defensively here
                     even if caller forgot.

    This helper exists so the ALOE worker can call ONE function and get both
    the chunked TD target (for critic) and the advantage (for actor). It
    mirrors the interface in `rlinf.algorithms.chunked_q.compute_chunked_td_target`
    so the sigma N1 stub stays compatible.
    """
    from rlinf.algorithms.chunked_q import compute_chunked_td_target

    if dones is None:
        dones = torch.zeros_like(rewards, dtype=torch.bool)
    q_target = compute_chunked_td_target(
        rewards=rewards,
        next_q=next_q_pess,
        gamma=gamma,
        dones=dones,
    )
    advantage = q_target - v_baseline.detach()
    return q_target, advantage
