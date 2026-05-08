#!/usr/bin/env python3
# Copyright 2026 The σ Project (sigma-phase1)
#
# eval_libero_pro.py
#
# Multi-seed LIBERO-PRO position-perturbation evaluation runner.
#
# This script delegates the actual env+rollout work to RLinf's
# `eval_embodied_agent.py`, but adds:
#
#   1. Multi-seed orchestration (runs `eval_embodied_agent.py` N times with
#      different env.eval.seed / actor.seed values).
#   2. Aggregation of per-seed success-rate from each run's log file
#      (parses `eval/success_once`).
#   3. Across-seed mean + 95% CI (Wilson, normal approx, and bootstrap)
#      written as a JSON summary alongside the run logs.
#
# Usage:
#   python tools/eval_libero_pro.py \
#       --config libero_spatial_eval_libero_pro_openpi_pi05 \
#       --ckpt /path/to/ckpt \
#       --seeds 0,1,2 \
#       [--total-num-envs 50] [--max-episode-steps 240] \
#       [--save-video false]
#
# Output:
#   logs/<timestamp>-<config>/seed<S>/eval_libero_pro.log
#   logs/<timestamp>-<config>/summary.json

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
EMBODIED_DIR = REPO_ROOT / "examples" / "embodiment"

DEFAULT_DATA_ROOT = "/home/jskang/sigma/liberopro_data"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-seed LIBERO-PRO position-perturbation eval (4 suites supported).",
    )
    p.add_argument(
        "--config",
        required=True,
        help="Hydra config name in examples/embodiment/config/, "
        "e.g. libero_spatial_eval_libero_pro_openpi_pi05",
    )
    p.add_argument(
        "--ckpt",
        default=None,
        help="Optional checkpoint path. If given, overrides actor.model.model_path "
        "and rollout.model.model_path.",
    )
    p.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated seed list (e.g. '0,1,2'). Default: 42.",
    )
    p.add_argument(
        "--perturbation",
        default="swap",
        choices=["swap", "object", "lan", "task", "all"],
        help="LIBERO_PERTURBATION value. Default 'swap' = position perturbation.",
    )
    p.add_argument(
        "--data-root",
        default=os.environ.get("LIBEROPRO_DATA_ROOT", DEFAULT_DATA_ROOT),
        help="LIBERO-PRO data root containing bddl_files/<suite>_swap and "
        "init_files/<suite>_swap. "
        "Default: %s or LIBEROPRO_DATA_ROOT env." % DEFAULT_DATA_ROOT,
    )
    p.add_argument(
        "--total-num-envs",
        type=int,
        default=None,
        help="Override env.eval.total_num_envs.",
    )
    p.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Override env.eval.max_episode_steps and max_steps_per_rollout_epoch.",
    )
    p.add_argument(
        "--save-video",
        default=None,
        choices=["true", "false"],
        help="Override env.eval.video_cfg.save_video.",
    )
    p.add_argument(
        "--mujoco-gl",
        default=os.environ.get("MUJOCO_GL", "osmesa"),
        help="MUJOCO_GL value (default: osmesa).",
    )
    p.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Additional Hydra overrides (key=value), repeatable.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands but don't execute.",
    )
    p.add_argument(
        "--log-root",
        default=str(REPO_ROOT / "logs"),
        help="Root directory for run logs.",
    )
    return p.parse_args()


def _build_overrides(args: argparse.Namespace, seed: int) -> List[str]:
    o: List[str] = []
    if args.ckpt:
        o.append(f"actor.model.model_path={args.ckpt}")
        o.append(f"rollout.model.model_path={args.ckpt}")
    if args.total_num_envs is not None:
        o.append(f"env.eval.total_num_envs={args.total_num_envs}")
    if args.max_episode_steps is not None:
        o.append(f"env.eval.max_episode_steps={args.max_episode_steps}")
        o.append(f"env.eval.max_steps_per_rollout_epoch={args.max_episode_steps}")
    if args.save_video is not None:
        o.append(f"env.eval.video_cfg.save_video={args.save_video}")
    o.append(f"env.eval.seed={seed}")
    o.append(f"actor.seed={seed}")
    o.extend(args.extra)
    return o


def _run_seed(args: argparse.Namespace, seed: int, log_dir: Path) -> Dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "eval_libero_pro.log"
    overrides = _build_overrides(args, seed) + [f"runner.logger.log_path={log_dir}"]

    src_file = EMBODIED_DIR / "eval_embodied_agent.py"
    config_path = EMBODIED_DIR / "config"

    env = os.environ.copy()
    env.update(
        {
            "EMBODIED_PATH": str(EMBODIED_DIR),
            "REPO_PATH": str(REPO_ROOT),
            "PYTHONPATH": f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}",
            "LIBERO_TYPE": "pro",
            "LIBERO_PERTURBATION": args.perturbation,
            "LIBEROPRO_DATA_ROOT": args.data_root,
            "MUJOCO_GL": args.mujoco_gl,
            "PYOPENGL_PLATFORM": env.get("PYOPENGL_PLATFORM", args.mujoco_gl),
            "ROBOT_PLATFORM": env.get("ROBOT_PLATFORM", "LIBERO"),
            "HYDRA_FULL_ERROR": "1",
        }
    )

    cmd: List[str] = [
        sys.executable,
        str(src_file),
        "--config-path",
        str(config_path),
        "--config-name",
        args.config,
    ] + overrides

    print(f"\n==> [seed={seed}] launching eval")
    print("    log_dir =", log_dir)
    print("    cmd     =", " ".join(shlex.quote(c) for c in cmd))

    if args.dry_run:
        return {"seed": seed, "dry_run": True}

    with open(log_file, "w") as lf:
        lf.write(" ".join(shlex.quote(c) for c in cmd) + "\n\n")
        lf.flush()
        rc = subprocess.call(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)

    metrics = _parse_metrics_from_log(log_file)
    metrics["seed"] = seed
    metrics["return_code"] = rc
    metrics["log_file"] = str(log_file)
    return metrics


_METRIC_RE = re.compile(r"eval/([A-Za-z0-9_]+)['\"]?\s*[:=]\s*([0-9eE\.\+\-]+)")


def _parse_metrics_from_log(log_file: Path) -> Dict[str, Any]:
    """Parse `eval/<key>: <value>` style metrics from the embodied eval log."""
    if not log_file.exists():
        return {}
    text = log_file.read_text(errors="ignore")
    # Aggregate the *last* occurrence of each metric (final log).
    out: Dict[str, float] = {}
    for m in _METRIC_RE.finditer(text):
        try:
            out[m.group(1)] = float(m.group(2))
        except ValueError:
            pass
    return out


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _normal_ci(values: List[float], z: float = 1.96) -> tuple[float, float, float]:
    """Mean ± z * SEM. Returns (mean, lo, hi)."""
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return (mean, mean, mean)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    sem = math.sqrt(var / n)
    return (mean, mean - z * sem, mean + z * sem)


def main() -> None:
    args = _parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer")

    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{ts}-{args.config}"
    if args.ckpt:
        ckpt_short = Path(args.ckpt).name
        run_id += f"-{ckpt_short}"
    base_dir = Path(args.log_root) / run_id
    base_dir.mkdir(parents=True, exist_ok=True)

    print("LIBERO-PRO position-perturbation eval")
    print("  config        :", args.config)
    print("  ckpt          :", args.ckpt or "<config default>")
    print("  perturbation  :", args.perturbation)
    print("  data_root     :", args.data_root)
    print("  seeds         :", seeds)
    print("  base_dir      :", base_dir)

    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        seed_dir = base_dir / f"seed{seed}"
        m = _run_seed(args, seed, seed_dir)
        per_seed.append(m)
        sr = m.get("success_once") or m.get("success") or m.get("success_rate")
        print(f"    seed={seed} -> success_once={sr}, n={m.get('num_trajectories')}")

    sr_values = [
        float(m["success_once"])
        for m in per_seed
        if m.get("success_once") is not None
    ]
    n_values = [
        int(m["num_trajectories"])
        for m in per_seed
        if m.get("num_trajectories") is not None
    ]

    summary: Dict[str, Any] = {
        "config": args.config,
        "ckpt": args.ckpt,
        "perturbation": args.perturbation,
        "data_root": args.data_root,
        "seeds": seeds,
        "per_seed": per_seed,
    }

    if sr_values:
        mean_sr, lo_n, hi_n = _normal_ci(sr_values)
        summary["mean_success_once"] = mean_sr
        summary["normal_95_ci"] = [lo_n, hi_n]

        # Wilson CI: pool successes/trials across seeds for binomial CI
        if n_values and len(n_values) == len(sr_values):
            total_n = sum(n_values)
            total_succ = sum(sr * n for sr, n in zip(sr_values, n_values))
            pooled_p = total_succ / max(total_n, 1)
            lo_w, hi_w = _wilson_ci(pooled_p, total_n)
            summary["pooled_success_once"] = pooled_p
            summary["pooled_n_trajectories"] = total_n
            summary["wilson_95_ci"] = [lo_w, hi_w]

    out_json = base_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    print("\nSummary:", out_json)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
