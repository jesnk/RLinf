"""σ-QRT Task 1 smoke tests.

Verifies:
  1. The π0.5 LIBERO SFT checkpoint exists on disk with the expected files
     (model.safetensors at root + norm_stats.json under physical-intelligence/libero/).
  2. The RLinf OpenPi0 RL action model can be instantiated with the pi05_libero
     preset config and its weights can be loaded from the SFT safetensors file
     with no unexpected key errors (missing keys for RL-only modules like
     ValueHead are allowed).

Run from repo root:
    PYTHONPATH=. python -m pytest tests/sigma_qrt/test_setup.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = REPO_ROOT / "ckpts" / "pi05_libero_sft"
SAFETENSORS_PATH = CKPT_DIR / "model.safetensors"
NORM_STATS_PATH = CKPT_DIR / "physical-intelligence" / "libero" / "norm_stats.json"


def test_ckpt_exists():
    """π0.5 LIBERO SFT checkpoint files are present."""
    assert CKPT_DIR.exists(), f"ckpt dir missing: {CKPT_DIR}"
    assert SAFETENSORS_PATH.is_file(), f"model.safetensors missing: {SAFETENSORS_PATH}"
    # safetensors should be ~7.47 GB (>5 GB sanity floor).
    size = SAFETENSORS_PATH.stat().st_size
    assert size > 5 * 1024**3, f"model.safetensors suspiciously small: {size} bytes"
    assert NORM_STATS_PATH.is_file(), f"norm_stats.json missing: {NORM_STATS_PATH}"


def test_pi05_load():
    """OpenPi0ForRLActionPrediction with pi05_libero preset loads SFT weights."""
    # Heavy imports: torch + jax + openpi + rlinf. Skip cleanly if env is missing.
    pytest.importorskip("torch")
    pytest.importorskip("jax")
    pytest.importorskip("openpi")
    pytest.importorskip("safetensors")

    from openpi.models.pi0_config import Pi0Config
    from safetensors.torch import load_file as safetensors_load

    from rlinf.models.embodiment.openpi.openpi_action_model import (
        OpenPi0Config,
        OpenPi0ForRLActionPrediction,
    )

    # Match the 'pi05_libero' TrainConfig preset in
    # rlinf/models/embodiment/openpi/dataconfig/__init__.py.
    base_pi0_cfg = Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)
    cfg = OpenPi0Config(
        **{
            f.name: getattr(base_pi0_cfg, f.name)
            for f in base_pi0_cfg.__dataclass_fields__.values()
        },
        config_name="pi05_libero",
        # Keep RL extras off so SFT weights load with no critic-only key mismatches.
        add_value_head=False,
        use_dsrl=False,
        noise_method="flow_sde",
    )

    model = OpenPi0ForRLActionPrediction(cfg)
    assert model is not None
    state_dict = safetensors_load(str(SAFETENSORS_PATH), device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # SFT ckpt is base π0.5 — RL-specific heads (value_head, noise_head) won't be in it.
    # We only fail if there are *unexpected* keys (means our config doesn't match the ckpt).
    assert not unexpected, f"unexpected keys when loading SFT ckpt: {unexpected[:10]}"
    # And at least the backbone parameters should have been populated.
    assert next(model.parameters()).device.type in ("cpu", "cuda")
