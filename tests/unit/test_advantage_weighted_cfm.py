# Copyright 2026 The RLinf Authors / sigma project.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ALOE advantage-weighted CFM loss (sigma-phase1 baseline).

Reference: ALOE (arXiv:2602.12691, AgiBot/HKUST/Fudan, 2026-02).

These tests verify ONLY the algorithmic primitives in
rlinf.algorithms.advantage_weighted_cfm -- they intentionally do NOT
exercise the FSDP worker / replay buffer integration (covered in
integration tests). They also assert sigma-N1 chunked-Q interface
compatibility.
"""
from __future__ import annotations

import math

import pytest
import torch

from rlinf.algorithms.advantage_weighted_cfm import (
    compute_advantage_weight,
    compute_advantage_weighted_cfm_loss,
    compute_advantage_weighted_target,
    sample_aloe_flow_inputs,
)
from rlinf.algorithms.chunked_q import compute_chunked_td_target


# ---------------------------------------------------------------------------
# 1) Advantage weight transform
# ---------------------------------------------------------------------------
class TestAdvantageWeight:

    def test_zero_advantage_gives_unit_weight(self):
        """A = 0 ==> w = exp(0) = 1."""
        w = compute_advantage_weight(
            advantage=torch.zeros(8, 1), beta=0.5, eps_clip=5.0
        )
        assert torch.allclose(w, torch.ones_like(w), atol=1e-6)

    def test_positive_advantage_increases_weight(self):
        """Larger A ==> larger w (within clip)."""
        a_low = torch.full((4, 1), 0.1)
        a_high = torch.full((4, 1), 0.5)
        w_low = compute_advantage_weight(a_low, beta=0.5, eps_clip=5.0)
        w_high = compute_advantage_weight(a_high, beta=0.5, eps_clip=5.0)
        assert (w_high > w_low).all()

    def test_eps_clip_bounds_log_weight(self):
        """w must lie in [exp(-eps_clip), exp(eps_clip)] regardless of A."""
        eps_clip = 3.0
        a = torch.tensor([[-100.0], [100.0]])
        w = compute_advantage_weight(a, beta=0.5, eps_clip=eps_clip)
        assert w.min().item() >= math.exp(-eps_clip) - 1e-5
        assert w.max().item() <= math.exp(eps_clip) + 1e-5

    def test_w_max_caps_post_exp(self):
        """w_max provides a hard ceiling on the weight."""
        a = torch.full((4, 1), 100.0)
        w = compute_advantage_weight(
            a, beta=0.5, eps_clip=5.0, w_max=2.0
        )
        assert w.max().item() == pytest.approx(2.0, abs=1e-6)

    def test_weight_is_detached_by_default(self):
        """Weight must NOT carry gradients (stop-grad on critic)."""
        a = torch.randn(4, 1, requires_grad=True)
        w = compute_advantage_weight(a, beta=0.5, eps_clip=5.0)
        assert w.requires_grad is False


# ---------------------------------------------------------------------------
# 2) Advantage-weighted CFM loss (Eq. ALOE-actor)
# ---------------------------------------------------------------------------
class TestAdvantageWeightedCFMLoss:

    def test_zero_when_velocity_matches_target(self):
        """If f_theta = eps - a then residual = 0 => loss = 0."""
        torch.manual_seed(0)
        B, D = 8, 7
        a = torch.randn(B, D)
        eps = torch.randn(B, D)
        f_pred = eps - a   # exact match: residual = (eps - a) - f = 0
        adv = torch.randn(B, 1)
        loss = compute_advantage_weighted_cfm_loss(
            epsilon=eps,
            action=a,
            predicted_velocity=f_pred,
            advantage=adv,
            beta=0.5,
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_loss_positive_for_random_velocity(self):
        torch.manual_seed(1)
        B, D = 4, 6
        loss = compute_advantage_weighted_cfm_loss(
            epsilon=torch.randn(B, D),
            action=torch.randn(B, D),
            predicted_velocity=torch.randn(B, D),
            advantage=torch.randn(B, 1),
        )
        assert loss.item() > 0.0

    def test_gradient_flows_to_velocity_only(self):
        """Critical: gradient must reach f_theta but NOT advantage
        (since ALOE detaches the weight w)."""
        torch.manual_seed(2)
        B, D = 4, 5
        f = torch.randn(B, D, requires_grad=True)
        adv = torch.randn(B, 1, requires_grad=True)
        loss = compute_advantage_weighted_cfm_loss(
            epsilon=torch.randn(B, D),
            action=torch.randn(B, D),
            predicted_velocity=f,
            advantage=adv,
            beta=0.5,
        )
        loss.backward()
        assert f.grad is not None and f.grad.abs().sum() > 0.0
        # weight is detached, so adv must NOT receive any grad.
        assert adv.grad is None or adv.grad.abs().sum().item() == 0.0

    def test_higher_advantage_amplifies_loss(self):
        """Same residual, larger A ==> larger w ==> larger loss."""
        torch.manual_seed(3)
        B, D = 4, 3
        eps = torch.randn(B, D)
        a = torch.randn(B, D)
        f = torch.zeros(B, D)  # nonzero residual = eps - a
        adv_low = torch.zeros(B, 1)
        adv_high = torch.full((B, 1), 1.0)
        l_low = compute_advantage_weighted_cfm_loss(eps, a, f, adv_low, beta=0.5)
        l_high = compute_advantage_weighted_cfm_loss(eps, a, f, adv_high, beta=0.5)
        # exp(0)=1 vs exp(2) ~ 7.389
        ratio = l_high.item() / max(l_low.item(), 1e-8)
        assert ratio == pytest.approx(math.exp(1.0 / 0.5), rel=5e-2)

    def test_target_sign_matches_paper(self):
        """Regression target is (eps - a), NOT (a - eps).

        Set f = a - eps (the *opposite* sign convention from ours);
        residual = (eps - a) - (a - eps) = 2*(eps - a) ==> loss = 4 * ||eps-a||^2.
        """
        torch.manual_seed(4)
        B, D = 4, 5
        a = torch.randn(B, D)
        eps = torch.randn(B, D)
        f_wrong_sign = a - eps
        adv = torch.zeros(B, 1)  # w = 1
        loss = compute_advantage_weighted_cfm_loss(
            eps, a, f_wrong_sign, adv, beta=0.5
        )
        expected = 4.0 * (eps - a).pow(2).sum(dim=-1).mean().item()
        assert loss.item() == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# 3) sample_aloe_flow_inputs (Eq. ALOE-noisy)
# ---------------------------------------------------------------------------
class TestSampleAloeFlowInputs:

    def test_a_tilde_is_convex_combo(self):
        """a_tilde = eta * a + (1 - eta) * eps."""
        torch.manual_seed(5)
        B, D = 4, 3
        a = torch.randn(B, D)
        eps, eta, a_tilde = sample_aloe_flow_inputs(a)
        recon = eta * a + (1.0 - eta) * eps
        assert torch.allclose(a_tilde, recon, atol=1e-6)

    def test_eta_in_unit_interval(self):
        torch.manual_seed(6)
        B, D = 32, 4
        a = torch.randn(B, D)
        _, eta, _ = sample_aloe_flow_inputs(a)
        assert eta.shape == (B, 1)
        assert (eta >= 0.0).all() and (eta <= 1.0).all()

    def test_eps_unit_variance(self):
        torch.manual_seed(7)
        B, D = 4096, 2
        a = torch.zeros(B, D)
        eps, _, _ = sample_aloe_flow_inputs(a)
        # Std should be roughly 1 (large B).
        assert eps.std().item() == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# 4) Q-chunking compatibility (sigma N1 stub interface)
# ---------------------------------------------------------------------------
class TestChunkedQCompatibility:

    def test_target_matches_chunked_td_target(self):
        """`compute_advantage_weighted_target` must return the SAME
        q_target as `chunked_q.compute_chunked_td_target` so the LWD
        / sigma-N1 critic can swap in unchanged."""
        torch.manual_seed(8)
        B, H = 4, 5
        rewards = torch.randn(B, H)
        next_q = torch.randn(B, 1)
        v = torch.randn(B, 1)
        dones = torch.zeros(B, H, dtype=torch.bool)

        q_target_aloe, adv = compute_advantage_weighted_target(
            rewards=rewards,
            next_q_pess=next_q,
            v_baseline=v,
            gamma=0.99,
            H=H,
            dones=dones,
        )
        q_target_chunked = compute_chunked_td_target(
            rewards=rewards,
            next_q=next_q,
            gamma=0.99,
            dones=dones,
        )
        assert torch.allclose(q_target_aloe, q_target_chunked, atol=1e-6)
        # Advantage = q_target - sg(v_baseline)
        assert torch.allclose(adv, q_target_aloe - v.detach(), atol=1e-6)

    def test_advantage_does_not_carry_v_grad(self):
        """v_baseline gradient must NOT leak into advantage.

        Stop-grad is enforced inside `compute_advantage_weighted_target`
        via `v_baseline.detach()`. Result: the returned `adv` tensor must
        either (a) be a fresh leaf with no grad_fn, or (b) have a grad_fn
        that does NOT touch `v` on backward. We verify the stronger form:
        `adv` carries no autograd dependency on `v`.
        """
        B, H = 2, 3
        rewards = torch.zeros(B, H)
        next_q = torch.zeros(B, 1)
        v = torch.zeros(B, 1, requires_grad=True)
        _, adv = compute_advantage_weighted_target(
            rewards=rewards,
            next_q_pess=next_q,
            v_baseline=v,
            gamma=0.99,
            H=H,
        )
        # The detach() call inside the helper means `adv` must NOT trace
        # back to v. PyTorch represents this as either no grad_fn at all
        # OR a grad_fn that ignores v. Easiest invariant: adv.requires_grad
        # is False (no leaf needs grad in the computation graph here).
        assert adv.requires_grad is False, (
            "advantage should carry no autograd dependency on v_baseline; "
            f"requires_grad={adv.requires_grad}"
        )


# ---------------------------------------------------------------------------
# 5) End-to-end sigma vs ALOE differential (sanity)
# ---------------------------------------------------------------------------
class TestSigmaVsALOEGradientSemantics:

    def test_aloe_critic_grad_blocked(self):
        """In ALOE, the critic gradient must NEVER reach the actor params.

        We simulate this by routing critic params through the advantage
        weight w (detached) and confirming actor params receive grad
        ONLY via the squared residual, not via critic params.
        """
        torch.manual_seed(9)
        B, D = 4, 3
        # actor param theta enters f_theta directly
        theta = torch.zeros(B, D, requires_grad=True)
        # critic param phi enters the advantage; ALOE must detach this
        phi = torch.randn(B, 1, requires_grad=True)

        eps = torch.randn(B, D)
        a = torch.randn(B, D)
        f = theta  # toy: f_theta = theta
        loss = compute_advantage_weighted_cfm_loss(
            epsilon=eps, action=a, predicted_velocity=f,
            advantage=phi, beta=0.5,
        )
        loss.backward()
        assert theta.grad is not None and theta.grad.abs().sum() > 0.0
        # phi (critic-side) MUST be unchanged
        assert phi.grad is None or phi.grad.abs().sum().item() == 0.0
