# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for σ-QRT IQL variant losses.

Reference: Kostrikov et al. 2021 (Implicit Q-Learning). The three new losses
are designed to avoid querying the critic on out-of-distribution actions —
the failure mode that drove the σ-QRT β sweep Phase 1 collapse (q_mean → −∞).
"""

import torch


def test_qrt_iql_v_loss_shape_and_finite():
    from rlinf.algorithms.losses import qrt_iql_v_loss

    B = 16
    q_target = torch.randn(B)
    v_pred = torch.randn(B, requires_grad=True)
    loss = qrt_iql_v_loss(q_target, v_pred, tau=0.7)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    # Backward should flow to v_pred but NOT to q_target.
    loss.backward()
    assert v_pred.grad is not None
    assert torch.isfinite(v_pred.grad).all()


def test_qrt_iql_v_loss_tau_half_recovers_mse():
    """τ = 0.5 → weight = 0.5 everywhere → loss = 0.5 · MSE."""
    from rlinf.algorithms.losses import qrt_iql_v_loss

    torch.manual_seed(0)
    q_target = torch.randn(32)
    v_pred = torch.randn(32)
    loss = qrt_iql_v_loss(q_target, v_pred, tau=0.5)
    expected = 0.5 * (q_target - v_pred).pow(2).mean()
    assert torch.allclose(loss, expected, atol=1e-6)


def test_qrt_iql_v_loss_expectile_property():
    """With τ → 1, V is pushed upward to approximate max Q.

    Test by minimizing the loss via grad descent and confirming the
    converged V is closer to max(q) than to mean(q).
    """
    from rlinf.algorithms.losses import qrt_iql_v_loss

    torch.manual_seed(0)
    # Single state: pretend we sample many actions, q_target represents
    # different (s,a) pairs. The expectile-V on the same s should approach
    # the upper tail of q_target as τ → 1.
    q_samples = torch.tensor([0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0])
    v = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([v], lr=0.1)
    for _ in range(2000):
        opt.zero_grad()
        # Broadcast v across all q samples (same state).
        v_pred = v.expand(len(q_samples))
        loss = qrt_iql_v_loss(q_samples, v_pred, tau=0.9)
        loss.backward()
        opt.step()
    v_final = v.detach().item()
    q_mean = q_samples.mean().item()
    q_max = q_samples.max().item()
    # With τ=0.9, V should be much closer to q_max than q_mean.
    assert v_final > q_mean, f"τ=0.9 V should exceed mean: V={v_final} vs mean={q_mean}"
    assert v_final > 1.5, f"τ=0.9 V should reach upper tail, got {v_final}"
    assert v_final < q_max + 0.01, f"V cannot exceed max materially, got {v_final}"


def test_qrt_iql_q_loss_shape_and_finite():
    from rlinf.algorithms.losses import qrt_iql_q_loss

    B, C = 8, 10
    q1 = torch.randn(B, requires_grad=True)
    q2 = torch.randn(B, requires_grad=True)
    rewards = torch.randn(B, C)
    v_next = torch.randn(B)
    dones = torch.zeros(B)
    loss = qrt_iql_q_loss(q1, q2, rewards, v_next, dones, gamma=0.95, chunk_len=C)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert q1.grad is not None and q2.grad is not None


def test_qrt_iql_q_loss_zero_when_target_matches():
    """When q1 = q2 = target (computed manually), loss should be ~0."""
    from rlinf.algorithms.losses import qrt_iql_q_loss

    B, C = 4, 5
    gamma = 0.9
    rewards = torch.zeros(B, C)
    rewards[:, -1] = 1.0  # terminal reward
    v_next = torch.tensor([0.5, 0.25, 0.0, -0.3])
    dones = torch.tensor([0.0, 0.0, 1.0, 0.0])
    # Hand-compute target.
    discounted_r = sum((gamma**t) * rewards[:, t] for t in range(C))
    bootstrap = (gamma**C) * (1.0 - dones) * v_next
    target = discounted_r + bootstrap
    loss = qrt_iql_q_loss(
        target.clone(), target.clone(), rewards, v_next, dones, gamma, C
    )
    assert loss.item() < 1e-9, f"expected ~0, got {loss.item()}"


def test_qrt_iql_q_loss_v_next_is_detached():
    """V_next gradient must not flow through the Q loss (bootstrap stop-grad)."""
    from rlinf.algorithms.losses import qrt_iql_q_loss

    B, C = 4, 3
    q1 = torch.randn(B, requires_grad=True)
    q2 = torch.randn(B, requires_grad=True)
    rewards = torch.zeros(B, C)
    v_next = torch.randn(B, requires_grad=True)
    dones = torch.zeros(B)
    loss = qrt_iql_q_loss(q1, q2, rewards, v_next, dones, gamma=0.9, chunk_len=C)
    loss.backward()
    # q1/q2 should have gradients; v_next should NOT (bootstrap is detached).
    assert q1.grad is not None
    assert v_next.grad is None, "v_next must be detached inside qrt_iql_q_loss"


def test_qrt_iql_actor_loss_shape_and_finite():
    from rlinf.algorithms.losses import qrt_iql_actor_loss

    B, C, d_act = 8, 10, 7
    actions = torch.randn(B, C, d_act)
    mu = torch.randn(B, C, d_act, requires_grad=True)
    q_target = torch.randn(B)
    v_pred = torch.randn(B)
    loss = qrt_iql_actor_loss(actions, mu, q_target, v_pred, beta_iql=3.0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert mu.grad is not None


def test_qrt_iql_actor_loss_zero_when_actions_match():
    """When μ_θ == a_data, BC error is 0 → loss is 0 regardless of advantage."""
    from rlinf.algorithms.losses import qrt_iql_actor_loss

    B, C, d_act = 4, 5, 3
    actions = torch.randn(B, C, d_act)
    mu = actions.clone().requires_grad_(True)
    q_target = torch.randn(B) * 100  # huge advantage shouldn't matter
    v_pred = torch.zeros(B)
    loss = qrt_iql_actor_loss(actions, mu, q_target, v_pred, beta_iql=3.0)
    assert loss.item() < 1e-9, f"expected ~0, got {loss.item()}"


def test_qrt_iql_actor_loss_weight_clip_respected():
    """Advantage weight clipped to weight_clip; huge advantages don't blow up."""
    from rlinf.algorithms.losses import qrt_iql_actor_loss

    B, C, d_act = 4, 5, 3
    torch.manual_seed(0)
    actions = torch.zeros(B, C, d_act)
    mu = torch.ones(B, C, d_act)  # constant BC error per element = 1
    # Huge advantage that would otherwise produce exp(β · adv) → ∞.
    q_target = torch.full((B,), 1000.0)
    v_pred = torch.zeros(B)
    weight_clip = 50.0
    loss = qrt_iql_actor_loss(
        actions, mu, q_target, v_pred, beta_iql=10.0, weight_clip=weight_clip
    )
    # Per-sample BC error = sum over (C, d_act) of 1² = C * d_act = 15.
    # Weight clipped to 50. Expected = 50 * 15 = 750.
    expected = weight_clip * (C * d_act)
    assert torch.isfinite(loss), "weight_clip must prevent overflow"
    assert abs(loss.item() - expected) < 1e-3, (
        f"weight_clip not respected: loss={loss.item()} vs expected={expected}"
    )


def test_qrt_iql_actor_loss_positive_advantage_weights_more():
    """Higher advantage → larger weight → BC term dominated by that sample.

    Verify by constructing two batches with identical BC error but different
    advantages and confirming the high-advantage batch has higher loss.
    """
    from rlinf.algorithms.losses import qrt_iql_actor_loss

    B, C, d_act = 4, 3, 2
    actions = torch.zeros(B, C, d_act)
    mu = torch.ones(B, C, d_act)  # identical BC err per sample
    # Low advantage: β · 0 = 0 → weight 1.
    q_low = torch.zeros(B)
    v_zero = torch.zeros(B)
    loss_low = qrt_iql_actor_loss(actions, mu, q_low, v_zero, beta_iql=3.0)
    # Higher advantage: β · 1 = 3 → weight ≈ e^3.
    q_high = torch.ones(B)
    loss_high = qrt_iql_actor_loss(actions, mu, q_high, v_zero, beta_iql=3.0)
    assert loss_high.item() > loss_low.item()
    # Specifically, ratio should ≈ e^3 ≈ 20.
    ratio = loss_high.item() / loss_low.item()
    assert 15.0 < ratio < 25.0, f"weight ratio outside expected range: {ratio}"


def test_qrt_iql_actor_loss_q_target_and_v_pred_detached():
    """Gradient should NOT flow through q_target or v_pred (they're advantage)."""
    from rlinf.algorithms.losses import qrt_iql_actor_loss

    B, C, d_act = 4, 3, 2
    actions = torch.zeros(B, C, d_act)
    mu = torch.ones(B, C, d_act, requires_grad=True)
    q_target = torch.randn(B, requires_grad=True)
    v_pred = torch.randn(B, requires_grad=True)
    loss = qrt_iql_actor_loss(actions, mu, q_target, v_pred, beta_iql=1.0)
    loss.backward()
    assert mu.grad is not None
    assert q_target.grad is None, "q_target must be detached in actor loss"
    assert v_pred.grad is None, "v_pred must be detached in actor loss"
