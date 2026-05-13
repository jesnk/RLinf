import pytest
from pathlib import Path


CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / 'examples/embodiment/config/libero_long_qrt_openpi_pi05.yaml'
)


def test_qrt_long_config_exists():
    assert CONFIG_PATH.exists(), f'missing config: {CONFIG_PATH}'


def test_qrt_long_config_loads_with_omegaconf():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(CONFIG_PATH)
    # model section
    assert cfg.model.action_dim == 7
    assert cfg.model.chunk_len == 10
    assert cfg.model.token_dim == 2048
    assert cfg.model.encoder_layers == 4
    assert cfg.model.decoder_layers == 4
    assert cfg.model.encoder_heads == 8
    assert cfg.model.decoder_heads == 8
    assert cfg.model.encoder_ffn == 4096
    assert cfg.model.decoder_ffn == 4096
    assert cfg.model.actor_hidden == 256
    assert cfg.model.actor_layers == 2
    assert cfg.model.critic_hidden == 256
    assert cfg.model.critic_layers == 2
    assert cfg.model.proprio_dim == 8
    # training section
    assert cfg.training.lr_actor == 1e-4
    assert cfg.training.lr_critic == 1e-4
    assert cfg.training.lr_encoder == 1e-4
    assert cfg.training.batch_size == 256
    assert cfg.training.gamma == 0.95
    assert cfg.training.beta_bc == 0.3
    assert cfg.training.alpha_recon == 0.5
    assert cfg.training.tau_target == 0.005
    assert cfg.training.actor_update_freq == 2
    assert cfg.training.action_std == 0.05
    assert cfg.training.target_noise_std == 0.2
    assert cfg.training.target_noise_clip == 0.5
    assert cfg.training.ref_action_dropout == 0.5
    assert cfg.training.stop_grad_z_rl_next is True
    assert cfg.training.grad_clip_norm == 1.0
    assert cfg.training.warmup_steps == 2000
    assert cfg.training.q_clamp is None

    # env section
    assert cfg.env.name == 'libero_long'
    # data section
    assert 'offline_buffer_path' in cfg.data
    assert 'capacity' in cfg.data


def test_qrt_long_config_compatible_with_worker_setup():
    """The yaml fields must satisfy QRTOfflinePolicyWorker.setup() requirements."""
    from omegaconf import OmegaConf
    from rlinf.workers.actor.fsdp_qrt_offline_policy_worker import QRTOfflinePolicyWorker
    cfg = OmegaConf.load(CONFIG_PATH)
    worker = QRTOfflinePolicyWorker(cfg, device='cpu')
    worker.setup()
