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

"""σ-QRT β sweep Phase 1 periodic-eval coordinator.

Multi-lane equivalent of ``periodic_eval_watcher.py``. Polls the six
β-sweep training lane directories under a single ``--run_dir`` for
new ``ckpt_step{N}.pt`` files and dispatches a 25-ep async eval per
new ckpt across a pool of GPU worker threads (default GPU 3 + GPU 7).

Lanes
-----
- ``qrt_beta{0p1,0p3,1p0}_seed{1,2}`` under ``--run_dir`` —
  matches the layout written by ``experiments/beta_sweep_phase1_launcher.sh``.

Per-lane outputs
----------------
- ``<lane_out>/eval_step{N}.json`` — eval result JSON
  (same schema as ``eval_libero_sr.py``).
- ``<lane_out>/learning_curve.jsonl`` — append-only per-ckpt
  ``{step, ckpt, sr, n_success, n_eval, per_task_sr, wall_time,
  eval_duration_s, gpu, rc}``.

Exit conditions
---------------
- All six lanes have a ``.lane_done`` sentinel AND the job queue is drained
  AND a 120 s buffer pass finds no further new ckpts.
- ``--max_wait`` seconds elapse with no new ckpts arriving.

Design notes
------------
- Worker threads (not processes) — work is a blocking ``subprocess.run``,
  so the GIL is released and threads are simpler than ``mp.Pool``.
- Final ``ckpt.pt`` is intentionally **skipped** — the launcher already
  runs a 25-ep final eval on it (``<lane_out>/eval_sr.json``).
- An ``nvidia-smi`` memory precheck per dispatch guards against early
  GPU 7 contention (B0 eval still finishing). The worker re-queues the
  job and sleeps before retrying.
- ``learning_curve.jsonl`` writes use a per-lane lock + atomic
  ``.tmp`` + ``os.replace`` so two workers writing to different lanes
  are safe and the same-lane case is serialised.

Usage
-----
    python experiments/beta_sweep_periodic_eval_coordinator.py \\
        --run_dir runs/beta_sweep_phase1_20260514_043746 \\
        --gpus 3,7 \\
        --poll_interval 60 \\
        --max_wait 14400
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

logging.basicConfig(
    format="[beta_coord] %(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO_ROOT / "examples" / "embodiment" / "eval_libero_sr.py"
DEFAULT_CONFIG = (
    REPO_ROOT
    / "examples"
    / "embodiment"
    / "config"
    / "libero_long_qrt_openpi_pi05.yaml"
)

LANE_DIRS = [
    "qrt_beta0p1_seed1",
    "qrt_beta0p1_seed2",
    "qrt_beta0p3_seed1",
    "qrt_beta0p3_seed2",
    "qrt_beta1p0_seed1",
    "qrt_beta1p0_seed2",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _step_from_path(p: Path) -> int:
    """Extract integer step from ``ckpt_step{N}.pt`` filename."""
    return int(p.stem.replace("ckpt_step", ""))


def _list_intermediate_ckpts(lane_dir: Path) -> list[Path]:
    """All ``ckpt_step*.pt`` files in lane_dir (excludes final ``ckpt.pt``)."""
    if not lane_dir.exists():
        return []
    return sorted(
        lane_dir.glob("ckpt_step*.pt"),
        key=lambda p: _step_from_path(p),
    )


def _ckpt_signature(p: Path) -> tuple[str, float]:
    """(name, mtime) — used to detect rare file overwrites."""
    try:
        return p.name, p.stat().st_mtime
    except FileNotFoundError:
        return p.name, 0.0


def _append_jsonl_atomic(curve_path: Path, entry: dict, lock: threading.Lock) -> None:
    """Atomic append to ``learning_curve.jsonl`` (per-lane lock)."""
    line = json.dumps(entry) + "\n"
    with lock:
        body = curve_path.read_text() if curve_path.exists() else ""
        tmp = curve_path.with_suffix(curve_path.suffix + ".tmp")
        tmp.write_text(body + line)
        os.replace(tmp, curve_path)


def _gpu_free_mib(gpu_id: int) -> int | None:
    """Return free MiB on ``gpu_id`` via nvidia-smi. None on failure."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        proc = subprocess.run(
            [
                smi,
                f"--id={gpu_id}",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0:
            return None
        return int(proc.stdout.strip().splitlines()[0])
    except (ValueError, subprocess.TimeoutExpired):
        return None


# --------------------------------------------------------------------------- #
# Eval subprocess
# --------------------------------------------------------------------------- #
def _build_env(gpu_id: int) -> dict[str, str]:
    """Subprocess env mirroring the single-lane watcher."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("LIBERO_TYPE", "standard")
    env.setdefault("ROBOT_PLATFORM", "LIBERO")
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run_eval(
    ckpt: Path,
    cfg: Path,
    num_eval: int,
    num_envs: int,
    seed: int,
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
        "--num_envs",
        str(num_envs),
        "--seed",
        str(seed),
        "--output",
        str(out_json),
    ]
    log.info("subprocess (gpu=%s): %s", env["CUDA_VISIBLE_DEVICES"], " ".join(cmd))
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


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
class Job:
    __slots__ = ("ckpt", "lane_out", "step", "lane_name", "attempts")

    def __init__(self, ckpt: Path, lane_out: Path, step: int, lane_name: str) -> None:
        self.ckpt = ckpt
        self.lane_out = lane_out
        self.step = step
        self.lane_name = lane_name
        self.attempts = 0


def _worker(
    gpu_id: int,
    job_q: "queue.Queue[Job | None]",
    stop_evt: threading.Event,
    lane_locks: dict[str, threading.Lock],
    cfg_path: Path,
    num_eval: int,
    num_envs: int,
    seed: int,
    eval_timeout: int,
    gpu_min_free_mib: int,
) -> None:
    """Pull jobs off the queue and run sequential eval subprocesses."""
    env = _build_env(gpu_id)
    log.info("worker g%d started", gpu_id)
    while not stop_evt.is_set():
        try:
            job = job_q.get(timeout=10)
        except queue.Empty:
            continue
        if job is None:
            job_q.task_done()
            break

        # GPU precheck (cheap — guards against early-start contention).
        free_mib = _gpu_free_mib(gpu_id)
        if free_mib is not None and free_mib < gpu_min_free_mib:
            log.warning(
                "worker g%d: free=%dMiB < min=%dMiB — requeue ckpt=%s lane=%s "
                "(attempt %d)",
                gpu_id,
                free_mib,
                gpu_min_free_mib,
                job.ckpt.name,
                job.lane_name,
                job.attempts,
            )
            job.attempts += 1
            time.sleep(60)
            # Re-queue at the back; another worker may pick it up.
            job_q.put(job)
            job_q.task_done()
            continue

        out_json = job.lane_out / f"eval_step{job.step}.json"
        if out_json.exists():
            log.info(
                "worker g%d: %s/%s already evaluated — skip",
                gpu_id,
                job.lane_name,
                out_json.name,
            )
            job_q.task_done()
            continue

        log.info(
            "worker g%d: start lane=%s ckpt=%s step=%d",
            gpu_id,
            job.lane_name,
            job.ckpt.name,
            job.step,
        )
        t0 = time.time()
        rc, _stdout, stderr = _run_eval(
            job.ckpt,
            cfg_path,
            num_eval,
            num_envs,
            seed,
            out_json,
            env,
            eval_timeout,
        )
        wall = time.time() - t0

        entry: dict = {
            "step": job.step,
            "ckpt": job.ckpt.name,
            "lane": job.lane_name,
            "wall_time": time.strftime("%H:%M:%S"),
            "eval_duration_s": round(wall, 1),
            "gpu": gpu_id,
            "rc": rc,
        }
        if rc != 0:
            log.error(
                "worker g%d: lane=%s step=%d rc=%d duration=%.1fs; stderr tail:\n%s",
                gpu_id,
                job.lane_name,
                job.step,
                rc,
                wall,
                stderr[-1500:],
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
            except Exception as exc:  # parsing failure shouldn't kill worker
                log.error("worker g%d: failed to parse %s: %s", gpu_id, out_json, exc)
                entry["parse_error"] = str(exc)
        else:
            log.error(
                "worker g%d: eval output %s missing after subprocess; rc=%d",
                gpu_id,
                out_json,
                rc,
            )
            entry["missing_output"] = True

        _append_jsonl_atomic(
            job.lane_out / "learning_curve.jsonl",
            entry,
            lane_locks[job.lane_name],
        )
        log.info(
            "worker g%d: done lane=%s step=%d sr=%s n=%s/%s duration=%.1fs",
            gpu_id,
            job.lane_name,
            job.step,
            entry.get("sr"),
            entry.get("n_success"),
            entry.get("n_eval"),
            wall,
        )
        job_q.task_done()

    log.info("worker g%d exit", gpu_id)


# --------------------------------------------------------------------------- #
# Main coordinator loop
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="σ-QRT β sweep Phase 1 periodic-eval coordinator."
    )
    p.add_argument(
        "--run_dir",
        required=True,
        help="beta_sweep_phase1_<stamp>/ dir containing the six lane subdirs.",
    )
    p.add_argument(
        "--gpus",
        default="3,7",
        help="Comma-separated GPU IDs for the worker pool (default '3,7').",
    )
    p.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="YAML config (default libero_long_qrt_openpi_pi05.yaml).",
    )
    p.add_argument(
        "--num_eval",
        type=int,
        default=25,
        help="Episodes per ckpt (default 25).",
    )
    p.add_argument(
        "--num_envs",
        type=int,
        default=5,
        help="Parallel env workers per eval (default 5).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Eval seed (default 1).",
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
        default=14400,
        help="Exit after this many seconds idle (no new ckpts), default 14400 = 4h.",
    )
    p.add_argument(
        "--eval_timeout",
        type=int,
        default=3600,
        help="Per-ckpt eval subprocess timeout in seconds (default 3600).",
    )
    p.add_argument(
        "--drain_buffer_s",
        type=int,
        default=120,
        help=(
            "After all lane_done sentinels appear AND queue is empty, wait this "
            "many seconds and re-scan once before exit (default 120)."
        ),
    )
    p.add_argument(
        "--gpu_min_free_mib",
        type=int,
        default=20000,
        help=(
            "Minimum free MiB required on a worker GPU before dispatch; "
            "below this the job is re-queued (default 20000 = 20 GiB). "
            "Guards against early-start contention with the lingering B0 eval."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        log.error("run_dir does not exist: %s", run_dir)
        return 2
    if not EVAL_SCRIPT.exists():
        log.error("eval script not found at %s", EVAL_SCRIPT)
        return 2

    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        log.error("--gpus must list at least one GPU id")
        return 2

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        log.error("config does not exist: %s", cfg_path)
        return 2

    log.info(
        "run_dir=%s gpus=%s cfg=%s num_eval=%d num_envs=%d seed=%d "
        "poll=%ds max_wait=%ds gpu_min_free=%dMiB",
        run_dir,
        gpus,
        cfg_path,
        args.num_eval,
        args.num_envs,
        args.seed,
        args.poll_interval,
        args.max_wait,
        args.gpu_min_free_mib,
    )

    job_q: "queue.Queue[Job | None]" = queue.Queue()
    stop_evt = threading.Event()
    lane_locks = {name: threading.Lock() for name in LANE_DIRS}
    seen: dict[str, tuple[str, float]] = {}  # f"{lane}/{ckpt.name}" → signature

    threads: list[threading.Thread] = []
    for gpu in gpus:
        t = threading.Thread(
            target=_worker,
            args=(
                gpu,
                job_q,
                stop_evt,
                lane_locks,
                cfg_path,
                args.num_eval,
                args.num_envs,
                args.seed,
                args.eval_timeout,
                args.gpu_min_free_mib,
            ),
            name=f"eval-g{gpu}",
            daemon=True,
        )
        t.start()
        threads.append(t)

    last_new_t = time.time()
    drain_armed_at: float | None = None
    rc = 0
    try:
        while True:
            # Scan all six lanes.
            new_count = 0
            for lane_name in LANE_DIRS:
                lane_out = run_dir / lane_name
                for ckpt in _list_intermediate_ckpts(lane_out):
                    key = f"{lane_name}/{ckpt.name}"
                    sig = _ckpt_signature(ckpt)
                    if seen.get(key) == sig:
                        continue
                    seen[key] = sig
                    step = _step_from_path(ckpt)
                    job_q.put(Job(ckpt, lane_out, step, lane_name))
                    log.info(
                        "queued lane=%s ckpt=%s step=%d",
                        lane_name,
                        ckpt.name,
                        step,
                    )
                    new_count += 1

            if new_count:
                last_new_t = time.time()
                drain_armed_at = None  # any new work resets the drain timer

            # Termination: all six .lane_done present + queue drained.
            all_lanes_done = all(
                (run_dir / name / ".lane_done").exists() for name in LANE_DIRS
            )
            queue_drained = job_q.unfinished_tasks == 0
            if all_lanes_done and queue_drained:
                if drain_armed_at is None:
                    drain_armed_at = time.time()
                    log.info(
                        "all six .lane_done sentinels + queue drained; "
                        "drain buffer %ds armed",
                        args.drain_buffer_s,
                    )
                elif time.time() - drain_armed_at >= args.drain_buffer_s:
                    # Final rescan after the buffer.
                    final_new = 0
                    for lane_name in LANE_DIRS:
                        lane_out = run_dir / lane_name
                        for ckpt in _list_intermediate_ckpts(lane_out):
                            key = f"{lane_name}/{ckpt.name}"
                            sig = _ckpt_signature(ckpt)
                            if seen.get(key) == sig:
                                continue
                            seen[key] = sig
                            step = _step_from_path(ckpt)
                            job_q.put(Job(ckpt, lane_out, step, lane_name))
                            final_new += 1
                            log.info(
                                "buffer-scan queued lane=%s ckpt=%s step=%d",
                                lane_name,
                                ckpt.name,
                                step,
                            )
                    if final_new == 0 and job_q.unfinished_tasks == 0:
                        log.info(
                            "drain buffer elapsed with no further ckpts; exit "
                            "(total evaluated keys=%d).",
                            len(seen),
                        )
                        break
                    drain_armed_at = None  # found new work → keep going
            else:
                drain_armed_at = None

            # Stale watchdog.
            idle_s = time.time() - last_new_t
            if idle_s > args.max_wait and job_q.unfinished_tasks == 0:
                log.warning(
                    "max_wait %ds exceeded with no new ckpts (idle=%.0fs). exit.",
                    args.max_wait,
                    idle_s,
                )
                rc = 0
                break

            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        log.warning("KeyboardInterrupt — shutting down")
        rc = 130

    # Tell workers to stop after they drain anything still in flight.
    for _ in threads:
        job_q.put(None)
    stop_evt.set()
    for t in threads:
        t.join(timeout=30)

    log.info("coordinator exit rc=%d", rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
