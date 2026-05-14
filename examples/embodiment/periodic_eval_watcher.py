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

"""σ-QRT periodic evaluation watcher.

Polls a σ-QRT training run directory for intermediate
``ckpt_step{N}.pt`` snapshots produced by ``run_qrt_offline.py
--save_interval N`` and, for each new snapshot, fires a subprocess
running ``eval_libero_sr.py`` on a dedicated GPU. The per-snapshot SR
is appended to ``<watch_dir>/learning_curve.jsonl`` (one JSON object
per line) so the headline run produces a step-vs-SR learning curve.

Exit conditions
---------------
- ``<watch_dir>/.training_done`` sentinel exists AND no un-evaluated
  ``ckpt_step*.pt`` files remain.
- ``--max_wait`` seconds elapse with no new ckpts arriving.

Usage
-----
    python examples/embodiment/periodic_eval_watcher.py \\
        --watch_dir runs/w4_gate_parallel_<STAMP>/qrt_seed1 \\
        --config examples/embodiment/config/libero_long_qrt_openpi_pi05.yaml \\
        --num_eval 10 --gpu 6 --poll_interval 60

Notes
-----
- Subprocess CWD is the repo root so ``examples.embodiment.*`` imports
  resolve.
- Atomic write of ``learning_curve.jsonl`` uses a ``.tmp`` + ``os.replace``
  pattern so partial reads cannot occur.
- Eval failures (non-zero rc, missing JSON, parse errors) are logged and
  the ckpt is marked seen so the watcher does not loop on it.
- mtime is captured per ckpt so an overwrite (rare — the training script
  uses ``ckpt_step{N}`` with N as a unique step counter) is detected and
  the ckpt is re-evaluated.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    format="[eval_watcher] %(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = REPO_ROOT / "examples" / "embodiment" / "eval_libero_sr.py"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _step_from_path(p: Path) -> int:
    """Extract integer step from ``ckpt_step{N}.pt`` filename."""
    return int(p.stem.replace("ckpt_step", ""))


def _list_ckpts(watch: Path) -> list[Path]:
    """Return ckpt_step*.pt files in numeric step order."""
    out: list[Path] = []
    for p in watch.glob("ckpt_step*.pt"):
        try:
            _step_from_path(p)
        except ValueError:
            log.warning("skipping non-numeric ckpt name: %s", p.name)
            continue
        out.append(p)
    return sorted(out, key=_step_from_path)


def _append_jsonl_atomic(curve_path: Path, entry: dict) -> None:
    """Append a single JSON line via tmp + replace.

    Append-on-tmp is not perfectly atomic across multi-writer races, but
    here we have exactly one writer (this watcher), so the pattern guards
    only against partial-line reads by external consumers.
    """
    line = json.dumps(entry) + "\n"
    if curve_path.exists():
        body = curve_path.read_text()
    else:
        body = ""
    tmp = curve_path.with_suffix(curve_path.suffix + ".tmp")
    tmp.write_text(body + line)
    os.replace(tmp, curve_path)


def _run_eval(
    ckpt: Path,
    cfg: Path,
    num_eval: int,
    out_json: Path,
    env: dict[str, str],
    timeout: int,
) -> tuple[int, str, str]:
    """Run eval_libero_sr.py for one ckpt; return (rc, stdout, stderr)."""
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--config",
        str(cfg),
        "--ckpt",
        str(ckpt),
        "--num_eval",
        str(num_eval),
        "--output",
        str(out_json),
    ]
    log.info("subprocess: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            cwd=str(REPO_ROOT),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        return 124, exc.stdout or "", (exc.stderr or "") + f"\nTIMEOUT after {timeout}s"


def _ckpt_signature(p: Path) -> tuple[str, float]:
    """(name, mtime) — used to detect file overwrites."""
    try:
        return p.name, p.stat().st_mtime
    except FileNotFoundError:
        return p.name, 0.0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="σ-QRT periodic evaluation watcher (one GPU)."
    )
    p.add_argument(
        "--watch_dir",
        required=True,
        help="σ-QRT training output dir to poll for ckpt_step*.pt files.",
    )
    p.add_argument("--config", required=True, help="YAML config (same as training).")
    p.add_argument(
        "--num_eval",
        type=int,
        default=10,
        help="Episodes per ckpt for the learning curve (default 10).",
    )
    p.add_argument(
        "--gpu",
        type=int,
        required=True,
        help="GPU index for eval subprocess (CUDA_VISIBLE_DEVICES + MUJOCO_EGL_DEVICE_ID).",
    )
    p.add_argument(
        "--poll_interval",
        type=int,
        default=60,
        help="Seconds between directory polls (default 60).",
    )
    p.add_argument(
        "--max_wait",
        type=int,
        default=7200,
        help="Exit after this many seconds with no new ckpts (default 7200 = 2h).",
    )
    p.add_argument(
        "--eval_timeout",
        type=int,
        default=3600,
        help="Per-ckpt eval subprocess timeout in seconds (default 3600).",
    )
    p.add_argument(
        "--max_iters",
        type=int,
        default=0,
        help=(
            "If > 0, cap the number of poll iterations. Used only by tests "
            "to bound the watcher; production runs leave this at 0 = unbounded."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    watch = Path(args.watch_dir).resolve()
    watch.mkdir(parents=True, exist_ok=True)
    curve_path = watch / "learning_curve.jsonl"
    cfg_path = Path(args.config).resolve()

    if not EVAL_SCRIPT.exists():
        log.error("eval script not found at %s", EVAL_SCRIPT)
        return 2

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu)
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("LIBERO_TYPE", "standard")
    env.setdefault("ROBOT_PLATFORM", "LIBERO")
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    seen: dict[str, tuple[str, float]] = {}  # ckpt.name → signature(name, mtime)
    last_new_t = time.time()
    iters = 0

    log.info(
        "watch_dir=%s cfg=%s num_eval=%d gpu=%d poll_interval=%ds max_wait=%ds",
        watch,
        cfg_path,
        args.num_eval,
        args.gpu,
        args.poll_interval,
        args.max_wait,
    )

    while True:
        iters += 1
        training_done = (watch / ".training_done").exists()
        ckpts = _list_ckpts(watch)

        # Detect newly-arrived or overwritten ckpts.
        new_ckpts: list[Path] = []
        for c in ckpts:
            sig = _ckpt_signature(c)
            prev = seen.get(c.name)
            if prev is None or prev != sig:
                new_ckpts.append(c)
                seen[c.name] = sig

        if new_ckpts:
            last_new_t = time.time()

        for ckpt in new_ckpts:
            step = _step_from_path(ckpt)
            out_json = watch / f"eval_step{step}.json"
            log.info("evaluating ckpt step=%d → %s", step, out_json.name)
            t0 = time.time()
            rc, stdout, stderr = _run_eval(
                ckpt, cfg_path, args.num_eval, out_json, env, args.eval_timeout
            )
            wall = time.time() - t0
            entry: dict = {
                "step": step,
                "ckpt": ckpt.name,
                "wall_time": time.strftime("%H:%M:%S"),
                "eval_duration_s": round(wall, 1),
                "rc": rc,
            }
            if rc != 0:
                log.error(
                    "eval rc=%d for step=%d (duration=%.1fs); stderr tail:\n%s",
                    rc,
                    step,
                    wall,
                    stderr[-1000:],
                )
            if out_json.exists():
                try:
                    d = json.loads(out_json.read_text())
                    entry.update(
                        {
                            "sr": d.get("sr"),
                            "n_success": d.get("n_success"),
                            "n_eval": d.get("n_eval"),
                            "per_task_sr": d.get("per_task_sr"),
                        }
                    )
                except Exception as exc:  # parsing failure shouldn't kill watcher
                    log.error("failed to parse %s: %s", out_json, exc)
                    entry["parse_error"] = str(exc)
            else:
                log.error(
                    "eval output %s missing after subprocess; rc=%d",
                    out_json,
                    rc,
                )
                entry["missing_output"] = True
            _append_jsonl_atomic(curve_path, entry)
            log.info(
                "step=%d sr=%s n=%s/%s duration=%.1fs",
                step,
                entry.get("sr"),
                entry.get("n_success"),
                entry.get("n_eval"),
                wall,
            )

        # Exit when training is done AND we've drained every ckpt on disk.
        if training_done:
            current = _list_ckpts(watch)
            unevaluated = [c for c in current if c.name not in seen]
            if not unevaluated:
                log.info(
                    "training_done + all ckpts evaluated (count=%d). exit.",
                    len(seen),
                )
                return 0

        # Stale watchdog: no new ckpts within max_wait.
        idle_s = time.time() - last_new_t
        if idle_s > args.max_wait:
            log.warning(
                "max_wait %ds exceeded with no new ckpts (idle=%.0fs). exit.",
                args.max_wait,
                idle_s,
            )
            return 0

        if args.max_iters and iters >= args.max_iters:
            log.info("max_iters=%d reached; exit.", args.max_iters)
            return 0

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    sys.exit(main())
