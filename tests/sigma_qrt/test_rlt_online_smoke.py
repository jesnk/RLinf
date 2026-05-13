"""Tests for RLTOnlineSimWorker (Task 9, paper-faithful RLT-online sim port).

See work/hbz/sigma/sigma_pivot_v2_design.md section 3.6 (B2 baseline) for
the paper-faithful contract enforced here.
"""

import pytest
import torch
from omegaconf import OmegaConf


def _cfg(token_dim=128, m_tokens=16, chunk_len=10, action_dim=7, proprio_dim=8):
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
            "lr_actor": 1e-4, "lr_critic": 1e-4, "lr_encoder": 1e-4,
            "gamma": 0.95, "beta_bc": 0.3, "alpha_recon": 0.5,
            "tau_target": 0.005, "actor_update_freq": 2, "action_std": 0.05,
            "target_noise_std": 0.2, "target_noise_clip": 0.5,
            "ref_action_dropout": 0.5, "stop_grad_z_rl_next": True,
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


def test_rlt_online_stage1_step_updates_encoder_only():
    from rlinf.workers.actor.fsdp_rlt_online_sim_worker import RLTOnlineSimWorker
    cfg = _cfg()
    worker = RLTOnlineSimWorker(cfg, device="cpu")
    worker.setup()
    # Snapshot critic params (should NOT change during stage1)
    critic_p0 = next(worker.critic.parameters()).clone()
    actor_p0 = next(worker.actor.parameters()).clone()
    enc_p0 = next(worker.encoder.parameters()).clone()
    batch = {"z_obs": torch.randn(4, 16, 128)}
    metrics = worker.stage1_step(batch)
    assert "loss_recon" in metrics
    assert torch.isfinite(torch.tensor(metrics["loss_recon"]))
    # encoder should update
    enc_p1 = next(worker.encoder.parameters())
    assert not torch.equal(enc_p0, enc_p1), "encoder did not update in stage1"
    # critic / actor must NOT update in stage1
    assert torch.equal(critic_p0, next(worker.critic.parameters())), "critic updated in stage1 (should not)"
    assert torch.equal(actor_p0, next(worker.actor.parameters())), "actor updated in stage1 (should not)"


def test_rlt_online_freeze_encoder_disables_grad():
    from rlinf.workers.actor.fsdp_rlt_online_sim_worker import RLTOnlineSimWorker
    cfg = _cfg()
    worker = RLTOnlineSimWorker(cfg, device="cpu")
    worker.setup()
    worker.freeze_encoder()
    for p in worker.encoder.parameters():
        assert not p.requires_grad
    for p in worker.decoder.parameters():
        assert not p.requires_grad


def test_rlt_online_stage2_no_encoder_grad_after_freeze():
    """After freeze_encoder(), stage2 must not update encoder params."""
    from rlinf.workers.actor.fsdp_rlt_online_sim_worker import RLTOnlineSimWorker
    cfg = _cfg()
    worker = RLTOnlineSimWorker(cfg, device="cpu")
    worker.setup()
    # Pre-fill encoder with some signal by running stage1
    for _ in range(2):
        worker.stage1_step({"z_obs": torch.randn(4, 16, 128)})
    worker.freeze_encoder()
    enc_p0 = next(worker.encoder.parameters()).clone()
    # Run stage2 - encoder must remain unchanged
    for s in range(4):
        worker.stage2_step(_dummy_batch(), step=s)
    enc_p1 = next(worker.encoder.parameters())
    assert torch.equal(enc_p0, enc_p1), "encoder changed during stage2 (frozen contract violated)"
    # And encoder grads should be None / zero
    for p in worker.encoder.parameters():
        assert p.grad is None or p.grad.abs().max() < 1e-9


def test_rlt_online_stage2_critic_actor_still_train():
    """Stage2 must still update critic and actor (only encoder is frozen)."""
    from rlinf.workers.actor.fsdp_rlt_online_sim_worker import RLTOnlineSimWorker
    cfg = _cfg()
    worker = RLTOnlineSimWorker(cfg, device="cpu")
    worker.setup()
    worker.freeze_encoder()
    critic_p0 = next(worker.critic.parameters()).clone()
    actor_p0 = next(worker.actor.parameters()).clone()
    for s in range(4):
        m = worker.stage2_step(_dummy_batch(), step=s)
        for k, v in m.items():
            assert torch.isfinite(torch.tensor(v)), f"NaN at step {s} key {k}"
    critic_p1 = next(worker.critic.parameters())
    actor_p1 = next(worker.actor.parameters())
    assert not torch.equal(critic_p0, critic_p1), "critic did not update in stage2"
    assert not torch.equal(actor_p0, actor_p1), "actor did not update in stage2"
