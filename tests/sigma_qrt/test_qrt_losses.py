import pytest
import torch


def test_qrt_critic_loss_shape_and_finite():
    from rlinf.algorithms.losses import qrt_critic_loss
    B = 8
    q1 = torch.randn(B)
    q2 = torch.randn(B)
    q_target = torch.randn(B)
    loss = qrt_critic_loss(q1, q2, q_target)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    # Identical inputs to target -> zero loss
    q_target_eq = q1.clone()
    zero = qrt_critic_loss(q1, q1, q_target_eq)
    assert zero.item() < 1e-9, f'expected approx 0, got {zero.item()}'


def test_qrt_critic_target_terminal_only_reward():
    from rlinf.algorithms.losses import qrt_compute_target
    B, C = 4, 10
    rewards = torch.zeros(B, C)
    rewards[:, -1] = 1.0  # terminal reward at last chunk step only
    q_next_target_min = torch.full((B,), 0.5)
    dones = torch.tensor([1.0, 0.0, 0.0, 1.0])
    gamma = 0.95
    target = qrt_compute_target(rewards, q_next_target_min, dones, gamma=gamma, chunk_len=C)
    assert target.shape == (B,)
    # done=1 sample 0: discounted reward only (gamma^{C-1} * 1.0), no bootstrap
    expected_done = gamma ** (C - 1)
    assert torch.allclose(target[0], torch.tensor(expected_done), atol=1e-5)
    # done=0 sample 1: discounted reward + gamma^C * 0.5
    expected_nodone = gamma ** (C - 1) + (gamma ** C) * 0.5
    assert torch.allclose(target[1], torch.tensor(expected_nodone), atol=1e-5)


def test_qrt_compute_target_zero_reward_zero_done_pure_bootstrap():
    '''When all rewards 0 and done 0, target = gamma^C * q_next.'''
    from rlinf.algorithms.losses import qrt_compute_target
    B, C = 3, 5
    rewards = torch.zeros(B, C)
    q_next = torch.tensor([1.0, -1.0, 0.5])
    dones = torch.zeros(B)
    gamma = 0.9
    target = qrt_compute_target(rewards, q_next, dones, gamma=gamma, chunk_len=C)
    expected = (gamma ** C) * q_next
    assert torch.allclose(target, expected, atol=1e-5)


def test_qrt_actor_loss_with_q_norm_and_bc():
    from rlinf.algorithms.losses import qrt_actor_loss
    B, C, d_act = 8, 10, 7
    q = torch.randn(B) * 2.0 + 3.0  # nonzero mean for Q-norm stability
    actions = torch.randn(B, C, d_act)
    ref_actions = actions.clone()  # zero BC contribution
    beta = 0.3
    loss = qrt_actor_loss(q, actions, ref_actions, beta=beta)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    # When actions == ref_actions, BC term is 0; loss = -Q / |mean(Q)|
    q_norm = q.abs().mean().clamp_min(1e-6)
    expected = (-q / q_norm).mean()
    assert torch.allclose(loss, expected, atol=1e-5), f'expected {expected.item()}, got {loss.item()}'


def test_qrt_actor_loss_bc_term_nonnegative():
    from rlinf.algorithms.losses import qrt_actor_loss
    B, C, d_act = 4, 5, 3
    q = torch.zeros(B)  # cancel Q term, keep only BC
    actions = torch.randn(B, C, d_act)
    ref_actions = torch.randn(B, C, d_act)
    loss = qrt_actor_loss(q, actions, ref_actions, beta=1.0)
    # Q is 0 -> loss = beta * mean(||a - tilde a||^2) >= 0
    assert loss.item() >= 0


def test_rlt_recon_loss_zero_when_equal():
    from rlinf.algorithms.losses import rlt_recon_loss
    B, M, d = 4, 16, 32
    z_target = torch.randn(B, M, d)
    loss = rlt_recon_loss(z_target.clone(), z_target)
    assert loss.dim() == 0
    assert loss.item() < 1e-9


def test_rlt_recon_loss_positive_when_diff():
    from rlinf.algorithms.losses import rlt_recon_loss
    B, M, d = 4, 16, 32
    z_hat = torch.randn(B, M, d)
    z_target = torch.randn(B, M, d)
    loss = rlt_recon_loss(z_hat, z_target)
    assert loss.item() > 0
