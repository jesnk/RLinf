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

"""σ-QRT periodic-eval feature smoke tests.

Two pieces are tested end-to-end without invoking the heavy
``eval_libero_sr.py`` (which loads π0.5 + LIBERO):

  1. ``run_qrt_offline.py --save_interval N`` writes
     ``ckpt_step{N}.pt`` + ``ckpt_latest.pt`` + ``.training_done``.
  2. ``periodic_eval_watcher.py`` polls a directory, processes
     pre-staged ``ckpt_step*.pt`` files via a stub ``eval_libero_sr.py``,
     appends one line per ckpt to ``learning_curve.jsonl``, and exits
     cleanly when ``.training_done`` appears.

The watcher's subprocess machinery is exercised by monkey-patching
``EVAL_SCRIPT`` to a stub that writes a deterministic SR json — this way
we don't need π0.5 / LIBERO loaded for the test.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CFG = REPO_ROOT / "examples/embodiment/config/libero_long_qrt_openpi_pi05.yaml"


# ----------------------------------------------------------------------
# Tiny offline buffer (same recipe as test_w4_gate_pipeline.py).
# ----------------------------------------------------------------------
def _make_tiny_buffer(out_path: Path) -> None:
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


# ----------------------------------------------------------------------
# 1. run_qrt_offline.py --save_interval N produces intermediate ckpts.
# ----------------------------------------------------------------------
def test_run_qrt_offline_save_interval(tiny_buffer: Path, tmp_path: Path) -> None:
    """With max_train_steps=10 + save_interval=4, expect ckpt_step4.pt + ckpt_step8.pt.

    The Stage-2 loop guards `(step + 1) < max_steps` so the final step does
    not duplicate the final ckpt.pt write. With max=10 + interval=4, the
    boundaries are at step indices 3 and 7 → produces ckpt_step4.pt and
    ckpt_step8.pt. Boundary at step 11 would have been skipped (>= max).
    """
    out_dir = tmp_path / "qrt_run_save"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("LIBERO_TYPE", "standard")
    env.setdefault("ROBOT_PLATFORM", "LIBERO")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "examples/embodiment/run_qrt_offline.py"),
        "--config",
        str(CFG),
        "--override",
        "data.offline_buffer_path=" + str(tiny_buffer),
        "--override",
        "data.capacity=8",
        "--override",
        "training.batch_size=2",
        "--override",
        "training.warmup_steps=4",
        "--override",
        "training.max_train_steps=10",
        "--override",
        "model.token_dim=128",
        "--override",
        "model.encoder_layers=1",
        "--override",
        "model.encoder_heads=2",
        "--override",
        "model.encoder_ffn=128",
        "--override",
        "model.decoder_layers=1",
        "--override",
        "model.decoder_heads=2",
        "--override",
        "model.decoder_ffn=128",
        "--override",
        "model.decoder_max_len=32",
        "--override",
        "model.actor_hidden=32",
        "--override",
        "model.critic_hidden=32",
        "--override",
        "logging.log_interval=2",
        "--variant",
        "qrt",
        "--output_dir",
        str(out_dir),
        "--no_wandb",
        "--device",
        "cpu",
        "--save_interval",
        "4",
    ]
    res = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600, env=env, cwd=str(REPO_ROOT)
    )
    assert res.returncode == 0, (
        f"run_qrt_offline failed\nstdout:\n{res.stdout[-2000:]}\nstderr:\n{res.stderr[-2000:]}"
    )

    # Intermediate ckpts.
    assert (out_dir / "ckpt_step4.pt").exists(), "ckpt_step4.pt not written"
    assert (out_dir / "ckpt_step8.pt").exists(), "ckpt_step8.pt not written"
    # No boundary at step 12 (max=10).
    assert not (out_dir / "ckpt_step12.pt").exists()
    # Latest copy + final ckpt + sentinel.
    assert (out_dir / "ckpt_latest.pt").exists()
    assert (out_dir / "ckpt.pt").exists()
    assert (out_dir / ".training_done").exists()
    # No stale .tmp files.
    leftover = list(out_dir.glob("*.tmp"))
    assert not leftover, f"leftover tmp files: {leftover}"


def test_run_qrt_offline_save_interval_disabled(
    tiny_buffer: Path, tmp_path: Path
) -> None:
    """--save_interval -1 (or 0) disables intermediate ckpts; existing smoke unchanged."""
    out_dir = tmp_path / "qrt_run_disabled"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("LIBERO_TYPE", "standard")
    env.setdefault("ROBOT_PLATFORM", "LIBERO")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "examples/embodiment/run_qrt_offline.py"),
        "--config",
        str(CFG),
        "--override",
        "data.offline_buffer_path=" + str(tiny_buffer),
        "--override",
        "data.capacity=8",
        "--override",
        "training.batch_size=2",
        "--override",
        "training.warmup_steps=4",
        "--override",
        "training.max_train_steps=10",
        "--override",
        "model.token_dim=128",
        "--override",
        "model.encoder_layers=1",
        "--override",
        "model.encoder_heads=2",
        "--override",
        "model.encoder_ffn=128",
        "--override",
        "model.decoder_layers=1",
        "--override",
        "model.decoder_heads=2",
        "--override",
        "model.decoder_ffn=128",
        "--override",
        "model.decoder_max_len=32",
        "--override",
        "model.actor_hidden=32",
        "--override",
        "model.critic_hidden=32",
        "--override",
        "logging.log_interval=2",
        "--variant",
        "qrt",
        "--output_dir",
        str(out_dir),
        "--no_wandb",
        "--device",
        "cpu",
        "--save_interval",
        "-1",
    ]
    res = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600, env=env, cwd=str(REPO_ROOT)
    )
    assert res.returncode == 0, (
        f"run_qrt_offline failed\nstdout:\n{res.stdout[-2000:]}\nstderr:\n{res.stderr[-2000:]}"
    )
    # No intermediate ckpts.
    inters = list(out_dir.glob("ckpt_step*.pt"))
    assert not inters, f"expected no intermediate ckpts, got: {inters}"
    # Final ckpt + sentinel still written.
    assert (out_dir / "ckpt.pt").exists()
    assert (out_dir / "ckpt_latest.pt").exists()
    assert (out_dir / ".training_done").exists()


# ----------------------------------------------------------------------
# 2. periodic_eval_watcher.py polls, fires stub eval, writes learning curve.
# ----------------------------------------------------------------------
def test_periodic_eval_watcher_drains_two_ckpts(tmp_path: Path) -> None:
    """Watcher picks up 2 staged ckpts + a stub eval, writes 2 JSONL rows, exits.

    The stub eval is a tiny python script that mirrors eval_libero_sr.py's
    output contract (--config, --ckpt, --num_eval, --output) but writes a
    deterministic SR file instead of running the real env. We point the
    watcher at the stub by overriding the EVAL_SCRIPT module-level path
    (cheapest possible test wiring — no eval_libero_sr import chain).
    """
    # 1) Stage a run directory with 2 dummy ckpts + .training_done so the
    # watcher sees a complete run on first poll.
    watch = tmp_path / "watch"
    watch.mkdir()
    (watch / "ckpt_step100.pt").write_text("dummy ckpt 100")
    (watch / "ckpt_step200.pt").write_text("dummy ckpt 200")
    (watch / ".training_done").touch()

    # 2) Write a stub eval_libero_sr.py-compatible CLI that writes
    # {"sr": ..., "n_success": ..., "n_eval": ...} JSON. The step is parsed
    # from the ckpt filename for determinism.
    stub = tmp_path / "stub_eval.py"
    stub.write_text(
        textwrap.dedent(
            """\
            import argparse
            import json
            import re
            from pathlib import Path

            p = argparse.ArgumentParser()
            p.add_argument("--config", required=True)
            p.add_argument("--ckpt", required=True)
            p.add_argument("--num_eval", type=int, required=True)
            p.add_argument("--output", required=True)
            a = p.parse_args()
            m = re.search(r"ckpt_step(\\d+)\\.pt", a.ckpt)
            step = int(m.group(1)) if m else 0
            sr = round(step / 1000.0, 4)
            out = {
                "sr": sr,
                "n_success": int(round(sr * a.num_eval)),
                "n_eval": a.num_eval,
                "per_task_sr": {},
                "ckpt": a.ckpt,
            }
            Path(a.output).write_text(json.dumps(out, indent=2))
            """
        )
    )

    # 3) Import the watcher module + patch EVAL_SCRIPT to our stub.
    sys.path.insert(0, str(REPO_ROOT / "examples/embodiment"))
    try:
        import importlib

        if "periodic_eval_watcher" in sys.modules:
            importlib.reload(sys.modules["periodic_eval_watcher"])
        import periodic_eval_watcher as pew

        pew.EVAL_SCRIPT = stub  # divert subprocess to the stub
        # Use a fake config path — stub doesn't read it.
        fake_cfg = tmp_path / "fake.yaml"
        fake_cfg.write_text("placeholder: true\n")

        rc = pew.main(
            [
                "--watch_dir",
                str(watch),
                "--config",
                str(fake_cfg),
                "--num_eval",
                "5",
                "--gpu",
                "0",  # CUDA_VISIBLE_DEVICES is set in env but stub ignores
                "--poll_interval",
                "1",
                "--max_wait",
                "60",
                "--max_iters",
                "5",
            ]
        )
    finally:
        sys.path.remove(str(REPO_ROOT / "examples/embodiment"))
    assert rc == 0, f"watcher returned non-zero rc={rc}"

    # 4) Verify learning_curve.jsonl has 2 rows in step order.
    curve = watch / "learning_curve.jsonl"
    assert curve.exists(), "learning_curve.jsonl not written"
    rows = [json.loads(l) for l in curve.read_text().splitlines() if l.strip()]
    assert len(rows) == 2, f"expected 2 rows, got {len(rows)}: {rows}"
    rows.sort(key=lambda r: r["step"])
    assert rows[0]["step"] == 100
    assert rows[0]["ckpt"] == "ckpt_step100.pt"
    assert rows[0]["n_eval"] == 5
    assert rows[0]["sr"] == pytest.approx(0.1)
    assert rows[1]["step"] == 200
    assert rows[1]["sr"] == pytest.approx(0.2)
    # Each entry includes wall_time + duration + rc.
    for r in rows:
        assert "wall_time" in r and "eval_duration_s" in r and r["rc"] == 0


def test_periodic_eval_watcher_skips_bad_eval(tmp_path: Path) -> None:
    """A failing eval subprocess should not crash the watcher — entry has rc!=0."""
    watch = tmp_path / "watch_bad"
    watch.mkdir()
    (watch / "ckpt_step50.pt").write_text("dummy")
    (watch / ".training_done").touch()

    # Stub that exits non-zero and writes no output.
    stub = tmp_path / "stub_fail.py"
    stub.write_text("import sys; sys.exit(7)\n")

    sys.path.insert(0, str(REPO_ROOT / "examples/embodiment"))
    try:
        import importlib

        if "periodic_eval_watcher" in sys.modules:
            importlib.reload(sys.modules["periodic_eval_watcher"])
        import periodic_eval_watcher as pew

        pew.EVAL_SCRIPT = stub
        fake_cfg = tmp_path / "fake.yaml"
        fake_cfg.write_text("placeholder: true\n")
        rc = pew.main(
            [
                "--watch_dir",
                str(watch),
                "--config",
                str(fake_cfg),
                "--num_eval",
                "3",
                "--gpu",
                "0",
                "--poll_interval",
                "1",
                "--max_wait",
                "30",
                "--max_iters",
                "5",
            ]
        )
    finally:
        sys.path.remove(str(REPO_ROOT / "examples/embodiment"))
    assert rc == 0

    curve = watch / "learning_curve.jsonl"
    assert curve.exists()
    rows = [json.loads(l) for l in curve.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    assert rows[0]["step"] == 50
    assert rows[0]["rc"] == 7
    assert rows[0].get("missing_output") is True
