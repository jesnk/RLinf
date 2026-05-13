import pytest
import torch
from omegaconf import OmegaConf


def _cfg(token_dim=128, m_tokens=16, chunk_len=10, action_dim=7, proprio_dim=8, batch_size=4):
    return OmegaConf.create({
        "model": {
            "token_dim": token_dim,
            "encoder_layers": 2,
            "encoder_heads": 4,
            "encoder_ffn": 256,
            "decoder_layers": 2,
            "decoder_heads": 4,
            "decoder_ffn": 256,
            "decoder_max_len": m_tokens + 8,
            "actor_hidden": 64,
            "actor_layers": 2,
            "critic_hidden": 64,
            "critic_layers": 2,
            "action_dim": action_dim,
            "proprio_dim": proprio_dim,
            "chunk_len": chunk_len,
        },
        "training": {
            "lr_actor": 1e-4,
            "lr_critic": 1e-4,
            "lr_encoder": 1e-4,
            "gamma": 0.95,
            "beta_bc": 0.3,
            "alpha_recon": 0.5,
            "tau_target": 0.005,
            "actor_update_freq": 2,
            "action_std": 0.05,
            "target_noise_std": 0.2,
            "target_noise_clip": 0.5,
            "ref_action_dropout": 0.5,
            "stop_grad_z_rl_next": True,
            "grad_clip_norm": 1.0,
        },
    })


def _dummy_batch(B=4, M=16, token_dim=128, chunk_len=10, action_dim=7, proprio_dim=8):
    return {
        "z_obs": torch.randn(B, M, token_dim),
        "next_z_obs": torch.randn(B, M, token_dim),
        "s_p": torch.randn(B, proprio_dim),
        "next_s_p": torch.randn(B, proprio_dim),
        "action": torch.randn(B, chunk_len, action_dim),
        "ref_action": torch.randn(B, chunk_len, action_dim),
        "next_ref_action": torch.randn(B, chunk_len, action_dim),
        "reward": torch.zeros(B, chunk_len),
        "done": torch.zeros(B),
    }


def test_qrt_worker_one_step_runs_and_finite():
    from rlinf.workers.actor.fsdp_qrt_offline_policy_worker import QRTOfflinePolicyWorker
    cfg = _cfg()
    worker = QRTOfflinePolicyWorker(cfg, device="cpu")
    worker.setup()
    batch = _dummy_batch()
    metrics = worker.train_step(batch, step=0)
    for k in ("loss_critic", "loss_actor", "loss_recon", "q_mean"):
        assert k in metrics, f"missing metric {k}"
        v = torch.tensor(metrics[k])
        assert torch.isfinite(v), f"non-finite {k}={metrics[k]}"


def test_qrt_worker_four_steps_no_nan():
    from rlinf.workers.actor.fsdp_qrt_offline_policy_worker import QRTOfflinePolicyWorker
    cfg = _cfg()
    worker = QRTOfflinePolicyWorker(cfg, device="cpu")
    worker.setup()
    for s in range(4):
        m = worker.train_step(_dummy_batch(), step=s)
        for k, v in m.items():
            assert torch.isfinite(torch.tensor(v)), f"NaN at step {s} key {k}"


def test_qrt_worker_encoder_receives_gradient():
    """Core σ-QRT novelty: encoder ϕ must receive critic gradient (Q-aware joint)."""
    from rlinf.workers.actor.fsdp_qrt_offline_policy_worker import QRTOfflinePolicyWorker
    cfg = _cfg()
    worker = QRTOfflinePolicyWorker(cfg, device="cpu")
    worker.setup()
    for p in worker.encoder.parameters():
        p.grad = None
    _ = worker.train_step(_dummy_batch(), step=0)
    enc_grad_norm = sum(
        p.grad.norm().item() if p.grad is not None else 0.0
        for p in worker.encoder.parameters()
    )
    assert enc_grad_norm > 1e-9, f"encoder received no gradient (Q-aware violated). norm={enc_grad_norm}"


def test_qrt_worker_target_net_soft_update():
    """After 2 step (actor_update_freq=2), target params should differ slightly from live."""
    from rlinf.workers.actor.fsdp_qrt_offline_policy_worker import QRTOfflinePolicyWorker
    cfg = _cfg()
    worker = QRTOfflinePolicyWorker(cfg, device="cpu")
    worker.setup()
    p0 = next(worker.critic_target.parameters()).clone()
    for s in range(4):
        worker.train_step(_dummy_batch(), step=s)
    p1 = next(worker.critic_target.parameters())
    assert not torch.equal(p0, p1), "target critic did not soft-update"
