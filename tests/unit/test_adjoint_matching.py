# Copyright 2026 The RLinf Authors / sigma project.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for QAM adjoint matching loss (sigma-phase1 baseline).

Reference equations: LWD Eq. 9 / Eq. 10 (= QAM Eq. 4-7 in Levine 2026).

These tests verify ONLY the algorithmic primitives in
rlinf.algorithms.adjoint_matching -- they intentionally do NOT exercise
the FSDP worker / replay buffer integration (covered in integration tests).
"""
from __future__ import annotations

import math

import pytest
import torch

from rlinf.algorithms.adjoint_matching import (
    compute_adjoint_matching_loss,
    compute_critic_action_grad,
)
from rlinf.algorithms.fleet_rl_qam import (
    compute_divl_value_loss,
    compute_lwd_chunked_q_target,
    quantile_value,
)


# ---------------------------------------------------------------------------
# QAM core loss
# ---------------------------------------------------------------------------
class TestAdjointMatchingLoss:

    def test_zero_when_velocity_matches_target(self):
        """If 2 f / sigma + sigma * tilde_g = 0 then L_QAM = 0 (Eq. 9)."""
        torch.manual_seed(0)
        B, D = 8, 7
        sigma_w = torch.full((B, 1), 0.5)
        a_w = torch.randn(B, D)
        q_grad = torch.randn(B, D)
        # tilde_g_w = -q_grad / lam, lam=1
        tilde_g = -q_grad
        # Choose f_delta so that 2 f / sigma + sigma * tilde_g = 0
        # =>  f_delta = -sigma^2 / 2 * tilde_g
        f_pred = -(sigma_w ** 2) / 2.0 * tilde_g

        loss = compute_adjoint_matching_loss(
            velocity_pred=f_pred,
            a_w=a_w,
            sigma_w=sigma_w,
            q_action_grad=q_grad,
            lam=1.0,
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_loss_positive_for_random_velocity(self):
        """Random f_delta should produce strictly positive loss."""
        torch.manual_seed(1)
        B, D = 4, 6
        loss = compute_adjoint_matching_loss(
            velocity_pred=torch.randn(B, D),
            a_w=torch.randn(B, D),
            sigma_w=torch.rand(B, 1) + 0.1,
            q_action_grad=torch.randn(B, D),
        )
        assert loss.item() > 0.0

    def test_gradient_flows_to_velocity_only(self):
        """Critical: gradient must reach f_delta but NOT q_grad
        (since QAM uses target-critic action grad with stop-grad)."""
        B, D = 4, 5
        f = torch.randn(B, D, requires_grad=True)
        qg = torch.randn(B, D, requires_grad=True)
        loss = compute_adjoint_matching_loss(
            velocity_pred=f,
            a_w=torch.randn(B, D),
            sigma_w=torch.full((B, 1), 0.7),
            q_action_grad=qg.detach(),  # stop-grad on critic-side
        )
        loss.backward()
        assert f.grad is not None and f.grad.abs().sum() > 0.0
        assert qg.grad is None  # we passed detached qg into loss

    def test_advantage_weighting_scales_linearly(self):
        """Advantage acts as a per-sample multiplier (sigma extension)."""
        torch.manual_seed(2)
        B, D = 4, 3
        f = torch.randn(B, D)
        a_w = torch.randn(B, D)
        sw = torch.full((B, 1), 0.5)
        qg = torch.randn(B, D)
        adv = torch.full((B, 1), 2.0)
        l_no = compute_adjoint_matching_loss(f, a_w, sw, qg)
        l_w  = compute_adjoint_matching_loss(f, a_w, sw, qg, advantage=adv)
        assert l_w.item() == pytest.approx(2.0 * l_no.item(), rel=1e-5)


# ---------------------------------------------------------------------------
# Critic action gradient helper
# ---------------------------------------------------------------------------
class TestCriticActionGrad:

    def test_action_grad_shape_matches(self):
        B, D = 3, 4
        action = torch.randn(B, D)
        # toy critic: Q = ||a||^2 -> grad_a Q = 2a
        def toy_critic(obs, a):
            return (a * a).sum(dim=-1, keepdim=True)
        grad = compute_critic_action_grad(toy_critic, obs=None, action=action)
        assert grad.shape == (B, D)
        assert torch.allclose(grad, 2.0 * action, atol=1e-5)

    def test_grad_is_detached(self):
        action = torch.randn(2, 3)
        def toy_critic(obs, a):
            return (a * a).sum(dim=-1, keepdim=True)
        grad = compute_critic_action_grad(toy_critic, obs=None, action=action)
        assert grad.requires_grad is False


# ---------------------------------------------------------------------------
# DIVL distributional value loss (LWD Eq. 12)
# ---------------------------------------------------------------------------
class TestDIVL:

    def test_divl_shapes(self):
        B, K = 5, 51
        logits = torch.randn(B, K)
        q_target = torch.zeros(B, 1)
        loss = compute_divl_value_loss(logits, q_target, v_min=-1.0, v_max=1.0, num_atoms=K)
        assert loss.dim() == 0  # scalar

    def test_divl_clipping(self):
        """Out-of-range Q targets should be clipped, not crash."""
        B, K = 3, 11
        logits = torch.randn(B, K)
        q_target = torch.tensor([[10.0], [-10.0], [0.0]])
        loss = compute_divl_value_loss(logits, q_target, -1.0, 1.0, K)
        assert torch.isfinite(loss)

    def test_quantile_value_in_support(self):
        B, K = 4, 11
        logits = torch.randn(B, K)
        q = quantile_value(logits, tau=0.5, v_min=-1.0, v_max=1.0)
        assert q.shape == (B, 1)
        assert (q >= -1.0).all() and (q <= 1.0).all()


# ---------------------------------------------------------------------------
# Chunked Q target (LWD Eq. 19)
# ---------------------------------------------------------------------------
class TestChunkedQTarget:

    def test_terminal_state_zero_bootstrap(self):
        B, n, H = 2, 3, 4
        rewards = torch.zeros(B, n, H)
        rewards[:, 0, 0] = 1.0
        v = torch.full((B, 1), 5.0)
        dones = torch.zeros(B, n, H, dtype=torch.bool)
        dones[:, 0, 0] = True  # terminal in first chunk
        y = compute_lwd_chunked_q_target(rewards, v, gamma=0.99, H=H, n=n, dones=dones)
        # bootstrap should be killed because there's a terminal
        assert (y < 5.0).all()

    def test_no_terminal_full_bootstrap(self):
        B, n, H = 1, 2, 3
        rewards = torch.zeros(B, n, H)
        v = torch.full((B, 1), 1.0)
        y = compute_lwd_chunked_q_target(rewards, v, gamma=1.0, H=H, n=n, dones=None)
        # gamma = 1, all rewards 0, V = 1  =>  y = 1
        assert y.item() == pytest.approx(1.0, rel=1e-5)

    def test_discount_factor_applied(self):
        B, n, H = 1, 2, 1
        rewards = torch.zeros(B, n, H)
        rewards[0, 1, 0] = 1.0  # reward in second chunk
        v = torch.zeros(B, 1)
        y = compute_lwd_chunked_q_target(rewards, v, gamma=0.5, H=H, n=n, dones=None)
        # expected: gamma^H * 1.0 = 0.5
        assert y.item() == pytest.approx(0.5, rel=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
