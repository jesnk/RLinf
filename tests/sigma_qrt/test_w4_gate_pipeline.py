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

"""σ-QRT Task 11: W4 gate-test pipeline smoke tests.

Two entry scripts are exercised end-to-end here:

  - examples/embodiment/run_qrt_offline.py — σ-QRT offline training
    (also drives A1/RLT-online-sim variants).
  - examples/embodiment/eval_libero_sr.py — LIBERO SR eval (B0 zero-shot
    or σ-QRT trained actor refinement when --ckpt is supplied).

The training smoke uses a tiny synthetic offline buffer (2 trajectories,
M=16 tokens, token_dim=128) so the test stays CPU-runnable and < 60 s
wall-clock. The eval smoke runs 1 LIBERO episode B0 (zero-shot π0.5),
which is slow (~70 s on B200) but gated on ckpt presence so it skips on
machines without the SFT weights.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CFG = REPO_ROOT / "examples/embodiment/config/libero_long_qrt_openpi_pi05.yaml"
CKPT = REPO_ROOT / "ckpts" / "pi05_libero_sft"


# ----------------------------------------------------------------------
# Tiny synthetic offline buffer (Trajectory schema).
# ----------------------------------------------------------------------
def _make_tiny_buffer(out_path: Path) -> None:
    """Build a 2-trajectory synthetic offline buffer the worker can consume.

    Matches the Trajectory schema from Task 8 (curr_obs / next_obs have
    keys z_obs / s_p / ref_action). Uses small M, chunk_len, token_dim to
    keep memory minimal so the test runs on CPU.
    """
    import torch

    from rlinf.data.embodied_io_struct import Trajectory

    T, B, M, d = 2, 1, 16, 128
    chunk_len, action_dim, proprio_dim = 10, 7, 8

    def _build() -> Trajectory:
        curr_obs = {
            "z_obs": torch.randn(T, B, M, d),
            "s_p": torch.randn(T, B, proprio_dim),
            "ref_action": torch.randn(T, B, chunk_len, action_dim),
        }
        next_obs = {
            "z_obs": torch.randn(T, B, M, d),
            "s_p": torch.randn(T, B, proprio_dim),
            "ref_action": torch.randn(T, B, chunk_len, action_dim),
        }
        return Trajectory(
            max_episode_length=T,
            model_weights_id="tiny_synth",
            actions=torch.randn(T, B, chunk_len, action_dim),
            rewards=torch.zeros(T, B, chunk_len),
            dones=torch.zeros(T, B),
            terminations=torch.zeros(T, B),
            truncations=torch.zeros(T, B),
            prev_logprobs=torch.zeros(T, B, action_dim),
            curr_obs=curr_obs,
            next_obs=next_obs,
        )

    trajs = [_build(), _build()]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(trajs, f)


@pytest.fixture
def tiny_buffer(tmp_path: Path) -> Path:
    buf = tmp_path / "tiny_buffer.pkl"
    _make_tiny_buffer(buf)
    return buf


def _run_script(script: str, args: list[str], timeout: int = 1500) -> subprocess.CompletedProcess:
    """Spawn an entry script with PYTHONPATH and GL env forwarded."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    # LIBERO headless rendering defaults (only matters for eval smoke).
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    env.setdefault("LIBERO_TYPE", "standard")
    env.setdefault("ROBOT_PLATFORM", "LIBERO")
    cmd = [sys.executable, str(REPO_ROOT / "examples/embodiment" / script)] + args
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
        env=env,
    )


# ----------------------------------------------------------------------
# 1. run_qrt_offline.py smoke (CPU-runnable σ-QRT 10 stage1 + 20 stage2).
# ----------------------------------------------------------------------
def test_run_qrt_offline_smoke(tiny_buffer: Path, tmp_path: Path) -> None:
    """σ-QRT training smoke: 10 stage1 + 20 stage2 steps with tiny tokens."""
    out_dir = tmp_path / "qrt_run"
    result = _run_script(
        "run_qrt_offline.py",
        [
            "--config", str(CFG),
            "--override", "data.offline_buffer_path=" + str(tiny_buffer),
            "--override", "data.capacity=8",
            "--override", "training.batch_size=2",
            "--override", "training.warmup_steps=10",
            "--override", "training.max_train_steps=20",
            # Tiny token modules so M=16 with token_dim=128 fits cleanly.
            "--override", "model.token_dim=128",
            "--override", "model.encoder_layers=1",
            "--override", "model.encoder_heads=2",
            "--override", "model.encoder_ffn=128",
            "--override", "model.decoder_layers=1",
            "--override", "model.decoder_heads=2",
            "--override", "model.decoder_ffn=128",
            "--override", "model.decoder_max_len=32",
            "--override", "model.actor_hidden=32",
            "--override", "model.critic_hidden=32",
            "--override", "logging.log_interval=2",
            "--variant", "qrt",
            "--output_dir", str(out_dir),
            "--no_wandb",
            "--device", "cpu",
        ],
        timeout=600,
    )
    assert result.returncode == 0, (
        "run_qrt_offline.py failed\n"
        f"stdout (tail):\n{result.stdout[-3000:]}\n"
        f"stderr (tail):\n{result.stderr[-3000:]}"
    )
    metrics_path = out_dir / "metrics.json"
    assert metrics_path.exists(), "metrics.json not written"
    data = json.loads(metrics_path.read_text())
    assert isinstance(data, list)
    assert len(data) > 0
    phases = {d.get("phase") for d in data}
    assert "stage1" in phases, f"no stage1 log: {phases}"
    assert "stage2" in phases, f"no stage2 log: {phases}"

    # Ckpt was saved.
    assert (out_dir / "ckpt.pt").exists(), "ckpt.pt not written"


# ----------------------------------------------------------------------
# 2. eval_libero_sr.py B0 zero-shot smoke. Slow; gated on ckpt presence.
# ----------------------------------------------------------------------
@pytest.mark.skipif(not CKPT.exists(), reason=f"π0.5 SFT ckpt not present at {CKPT}")
def test_eval_libero_sr_b0_smoke(tmp_path: Path) -> None:
    """B0 (π0.5 zero-shot) eval — 1 episode on libero_long for smoke."""
    out_path = tmp_path / "b0_eval.json"
    result = _run_script(
        "eval_libero_sr.py",
        [
            "--config", str(CFG),
            "--num_eval", "1",
            "--max_episode_len", "120",
            "--output", str(out_path),
            # No --ckpt → zero-shot base VLA (B0).
        ],
        timeout=1500,
    )
    assert result.returncode == 0, (
        "eval_libero_sr.py B0 failed\n"
        f"stdout (tail):\n{result.stdout[-3000:]}\n"
        f"stderr (tail):\n{result.stderr[-3000:]}"
    )
    assert out_path.exists()
    d = json.loads(out_path.read_text())
    for key in ("sr", "n_success", "n_eval"):
        assert key in d, f"missing eval key {key}"
    assert d["n_eval"] == 1
    assert d["ckpt"] is None  # B0 path.
