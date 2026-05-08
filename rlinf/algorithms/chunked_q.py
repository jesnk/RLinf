"""Chunked Q (Q-chunking unbiased H-step backup) for σ N1.

Reference: Q-chunking paper (arXiv 2507.07969).
σ가 추가하는 것: chunk 내부 per-step credit assignment.
RLinf의 fsdp_sac_policy_worker.py:350 default 는 `target = r_0 + γ^H * next_q`
(or `target = r.sum() + γ * next_q`) 인데, σ는 chunk 내부 H-step TD를
Σ γ^h r_{t+h} + γ^H * next_q * (~done_anywhere_in_chunk) 형태로 unbiased
H-step backup으로 교체한다.

핵심 식:
    G^H_t = Σ_{h=0}^{H-1} γ^h r_{t+h}  +  γ^H * Q(s_{t+H}, a_{t+H}) * (1 - done)

terminal masking: chunk 내부 어디든 done=True 가 발생하면 bootstrap 항을 0.

추가 옵션:
- per-step termination mask
- n-step return
- GAE-style λ-return
"""

from __future__ import annotations

from typing import Optional

import torch


def compute_chunked_td_target(
    rewards: torch.Tensor,
    next_q: torch.Tensor,
    gamma: float,
    dones: torch.Tensor,
    *,
    per_step_termination: bool = True,
) -> torch.Tensor:
    """Chunked TD target = Σ_h γ^h r_{t+h} + γ^H * next_q * (~done_in_chunk).

    Args:
        rewards:               [B, H] per-step rewards in the action chunk.
        next_q:                [B, 1] Q(s_{t+H}, a_{t+H}) — bootstrap target.
        gamma:                 discount factor.
        dones:                 [B, H] per-step terminal mask (bool/float).
        per_step_termination:  If True, truncate reward accumulation at the first
                               terminal step. If False, accumulate all H rewards
                               but still mask bootstrap if any done.

    Returns:
        target Q values, shape [B, 1].
    """
    if rewards.ndim != 2:
        raise ValueError(f"rewards must be [B, H], got {rewards.shape}")
    B, H = rewards.shape
    if next_q.shape != (B, 1):
        raise ValueError(f"next_q must be [B, 1], got {next_q.shape}")
    if dones.shape != (B, H):
        raise ValueError(f"dones must be [B, H], got {dones.shape}")

    dtype = rewards.dtype
    device = rewards.device

    discounts = gamma ** torch.arange(H, device=device, dtype=dtype)  # [H]

    if per_step_termination:
        not_done = (1.0 - dones.to(dtype))  # [B, H]
        cum_not_done = torch.cumprod(not_done, dim=1)
        ones = torch.ones((B, 1), dtype=dtype, device=device)
        mask = torch.cat([ones, cum_not_done[:, :-1]], dim=1)  # [B, H]
        discounted_rewards = (rewards * discounts.unsqueeze(0) * mask).sum(
            dim=1, keepdim=True
        )
    else:
        discounted_rewards = (rewards * discounts.unsqueeze(0)).sum(
            dim=1, keepdim=True
        )

    any_done = dones.to(dtype).clamp(max=1.0).any(dim=1, keepdim=True).to(dtype)
    bootstrap = (gamma ** H) * next_q * (1.0 - any_done)
    return discounted_rewards + bootstrap


def compute_chunked_n_step_return(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    dones: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """N-step return for chunked rollout. n <= H.

    G^n_t = Σ_{h=0}^{n-1} γ^h r_{t+h} + γ^n V(s_{t+n}) * (~done_within_n)

    Args:
        rewards:    [B, H]
        values:     [B, H+1] bootstrap values along the chunk
        gamma:      discount
        dones:      [B, H]
        n:          n-step horizon (1 <= n <= H)

    Returns:
        [B, 1] n-step return.
    """
    if not (1 <= n <= rewards.shape[1]):
        raise ValueError(f"n={n} must be in [1, H={rewards.shape[1]}]")
    if values.shape[1] != rewards.shape[1] + 1:
        raise ValueError(
            f"values must have shape [B, H+1], got {values.shape}"
        )

    dtype = rewards.dtype
    device = rewards.device
    B = rewards.shape[0]

    discounts = gamma ** torch.arange(n, device=device, dtype=dtype)
    not_done = (1.0 - dones[:, :n].to(dtype))
    cum_not_done = torch.cumprod(not_done, dim=1)
    ones = torch.ones((B, 1), dtype=dtype, device=device)
    mask = torch.cat([ones, cum_not_done[:, :-1]], dim=1)

    discounted_rewards = (rewards[:, :n] * discounts.unsqueeze(0) * mask).sum(
        dim=1, keepdim=True
    )

    any_done_within = dones[:, :n].to(dtype).clamp(max=1.0).any(dim=1, keepdim=True).to(dtype)
    bootstrap = (gamma ** n) * values[:, n:n + 1] * (1.0 - any_done_within)
    return discounted_rewards + bootstrap


def compute_chunked_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lam: float,
    dones: torch.Tensor,
) -> torch.Tensor:
    """GAE λ-return inside an action chunk.

    A^λ_t = Σ_{h=0}^{H-1} (γλ)^h * δ_{t+h}
    δ_h = r_h + γ V(s_{h+1}) (1 - done_h) - V(s_h)

    Returns: [B, 1] estimated λ-advantage.
    """
    if values.shape[1] != rewards.shape[1] + 1:
        raise ValueError(
            f"values must be [B, H+1], got {values.shape}"
        )
    dtype = rewards.dtype
    device = rewards.device
    B, H = rewards.shape

    not_done = (1.0 - dones.to(dtype))
    deltas = (
        rewards
        + gamma * values[:, 1:] * not_done
        - values[:, :-1]
    )

    adv = torch.zeros((B, H + 1), dtype=dtype, device=device)
    for h in reversed(range(H)):
        adv[:, h] = deltas[:, h] + gamma * lam * not_done[:, h] * adv[:, h + 1]
    return adv[:, :1]


def chunked_advantage_lcb(
    q_ensemble: torch.Tensor,
    log_pi: torch.Tensor,
    alpha: float,
    *,
    pessimism: float = 1.0,
) -> torch.Tensor:
    """LCB-style pessimistic Q for σ — used for actor target in CSF.

    Q_LCB = mean(Q) - pessimism * std(Q) - α * log_π

    Args:
        q_ensemble: [B, K]
        log_pi:    [B, 1]
        alpha:      entropy temperature
        pessimism:  LCB std multiplier (0 = mean only, 1 = mean - 1σ)

    Returns: [B, 1] pessimistic Q
    """
    if q_ensemble.ndim != 2:
        raise ValueError(f"q_ensemble must be [B, K], got {q_ensemble.shape}")
    q_mean = q_ensemble.mean(dim=1, keepdim=True)
    q_std = q_ensemble.std(dim=1, keepdim=True, unbiased=False)
    return q_mean - pessimism * q_std - alpha * log_pi


def truncate_chunk_to_first_done(
    rewards: torch.Tensor,
    dones: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Helper: zero out rewards/dones after the first terminal step.

    Useful for off-policy correction when chunk replay buffer stores variable
    effective lengths.
    """
    if rewards.shape != dones.shape:
        raise ValueError("rewards and dones must have same shape")
    dtype = rewards.dtype
    not_done = (1.0 - dones.to(dtype))
    cum_not_done = torch.cumprod(not_done, dim=1)
    ones = torch.ones_like(cum_not_done[:, :1])
    mask = torch.cat([ones, cum_not_done[:, :-1]], dim=1)
    return rewards * mask, dones * mask
