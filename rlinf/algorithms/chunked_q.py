"""Chunked Q (Q-chunking unbiased H-step backup) for σ N1.

Reference: Q-chunking paper (arXiv 2507.07969).
σ가 추가하는 것: chunk 내부 per-step credit assignment (RLinf의 γ^H discount는 이미 있음).
"""
import torch


def compute_chunked_td_target(
    rewards: torch.Tensor,  # [B, H] per-step rewards in chunk
    next_q: torch.Tensor,    # [B, 1] Q-value of next state (after chunk)
    gamma: float,
    dones: torch.Tensor,     # [B, H] terminal mask per step
) -> torch.Tensor:
    """Chunked TD target = Σ_h γ^h r_{t+h} + γ^H * next_q * (~done).

    Replaces RLinf's `target = r_0 + γ^H * next_q * (~done)` in fsdp_sac_policy_worker
    with per-step accumulation that uses ALL rewards in chunk.

    Returns: [B, 1] target Q values.
    """
    B, H = rewards.shape
    assert next_q.shape == (B, 1)
    assert dones.shape == (B, H)

    # Cumulative discount per step
    discounts = gamma ** torch.arange(H, device=rewards.device, dtype=rewards.dtype)  # [H]
    discounted_rewards = (rewards * discounts.unsqueeze(0)).sum(dim=1, keepdim=True)  # [B, 1]

    # Bootstrap term (only if no terminal in chunk)
    any_done = dones.any(dim=1, keepdim=True).float()  # [B, 1]
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

    Used for advantage computation in σ paper.
    """
    raise NotImplementedError("σ Phase 3: implement when needed")
