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

"""σ-QRT Task 2: π0.5 backbone embedding extraction tests.

Verifies that `OpenPi0ForRLActionPrediction.extract_embeddings(env_obs)` returns
the final-layer VLM hidden states (SigLIP image tokens + Gemma language tokens)
as a [B, M, d] tensor, detached so gradient cannot flow back into the frozen VLA.

Goes through the same obs pipeline as `predict_action_batch()`:
    env_obs -> obs_processor -> input_transform -> precision_processor
            -> Observation.from_dict -> _preprocess_observation
            -> _build_prefix_cache -> prefix_output (the embedding z_{1:M})

Run from RLinf repo root:
    PYTHONPATH=. python -m pytest tests/sigma_qrt/test_extract_embeddings.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = REPO_ROOT / "ckpts" / "pi05_libero_sft"
SAFETENSORS_PATH = CKPT_DIR / "model.safetensors"


def _load_pi05_libero_sft():
    """Load π0.5 LIBERO SFT model with full wrappers (input/output transforms,
    norm_stats). Uses `get_model()` so we exercise the same load path the
    training/eval code uses, including LiberoInputs/Outputs."""
    import torch
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi import get_model

    # Build a minimal OmegaConf cfg matching examples/embodiment/config/model/pi0_5.yaml
    # restricted to what get_model() reads.
    cfg = OmegaConf.create(
        {
            "model_type": "openpi",
            "model_path": str(CKPT_DIR),
            "precision": None,
            "num_action_chunks": 10,
            "action_dim": 7,
            "is_lora": False,
            "lora_rank": 32,
            "use_proprio": True,
            "num_steps": 5,
            "add_value_head": False,
            "openpi": {
                "config_name": "pi05_libero",
                "num_images_in_input": 2,
                "noise_level": 0.5,
                "action_chunk": 10,
                "num_steps": 5,
                "train_expert_only": True,
                "action_env_dim": 7,
                "noise_method": "flow_sde",
                "add_value_head": False,
                "value_after_vlm": False,
                "value_vlm_mode": "mean_token",
                "detach_critic_input": None,
                "use_dsrl": False,
                "dsrl_state_dim": 8,
                "dsrl_action_noise_dim": 32,
                "dsrl_num_q_heads": 10,
                "dsrl_agg_q": "mean",
                "dsrl_image_latent_dim": 64,
                "dsrl_state_latent_dim": 64,
                "dsrl_hidden_dims": [128, 128, 128],
            },
        }
    )
    model = get_model(cfg)
    model.eval()
    # Move to GPU if available (π0.5 is ~3.8B params; bfloat16 ckpt needs ~7.5 GB).
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model


def _make_dummy_env_obs(batch_size: int = 2) -> dict:
    """Construct env_obs in the format expected by `obs_processor()` (see
    `predict_action_batch()` for the canonical entry point). Keys are:

      - main_images: [B, H, W, 3] uint8 (will be normalized by LiberoInputs)
      - wrist_images: [B, H, W, 3] uint8 or None
      - extra_view_images: [B, H, W, 3] or None  (None for libero)
      - states: [B, 8] float (proprio for libero, 8 dim per make_libero_example)
      - task_descriptions: list[str] of length B
    """
    rng = np.random.default_rng(0)
    return {
        "main_images": rng.integers(
            0, 256, size=(batch_size, 224, 224, 3), dtype=np.uint8
        ),
        "wrist_images": rng.integers(
            0, 256, size=(batch_size, 224, 224, 3), dtype=np.uint8
        ),
        "extra_view_images": None,
        "states": rng.standard_normal((batch_size, 8)).astype(np.float32),
        "task_descriptions": ["pick up the black bowl"] * batch_size,
    }


@pytest.fixture(scope="module")
def model():
    pytest.importorskip("torch")
    pytest.importorskip("jax")
    pytest.importorskip("openpi")
    pytest.importorskip("safetensors")
    return _load_pi05_libero_sft()


def test_extract_embeddings_shape(model):
    """extract_embeddings(env_obs) returns [B, M, d]."""
    import torch

    env_obs = _make_dummy_env_obs(batch_size=2)
    with torch.no_grad():
        z = model.extract_embeddings(env_obs)
    assert z.dim() == 3, f"expected [B, M, d], got shape {tuple(z.shape)}"
    assert z.shape[0] == 2, f"batch dim mismatch: {tuple(z.shape)}"
    # d should be a plausible transformer hidden dim (Gemma-2B for π0.5 is ~2048).
    assert 512 <= z.shape[2] <= 4096, f"unexpected hidden dim {z.shape[2]}"
    assert torch.isfinite(z.float()).all(), "non-finite values in VLM embeddings"


def test_extract_embeddings_detached(model):
    """VLA is frozen → output tensor must be detached (no backward through VLA)."""
    env_obs = _make_dummy_env_obs(batch_size=2)
    z = model.extract_embeddings(env_obs)
    assert not z.requires_grad, (
        "extract_embeddings output must be detached (frozen VLA)"
    )
