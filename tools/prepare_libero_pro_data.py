#!/usr/bin/env python3
# Copyright 2026 The σ Project (sigma-phase1)
#
# prepare_libero_pro_data.py
#
# Generate LIBERO-PRO position-perturbation (swap) bddl files + matching
# init_states for all 4 task suites (Spatial / Object / Goal / 10), using the
# upstream LIBERO-PRO repo's `perturbation.py` as the swap generator.
#
# Outputs:
#   <data_root>/bddl_files/libero_<suite>_swap/<task>_swap{N}.bddl
#   <data_root>/init_files/libero_<suite>_swap/<task>_swap{N}.pruned_init
#
# Usage:
#   python tools/prepare_libero_pro_data.py \
#       --libero-pro-root /home/jskang/sigma/LIBERO-PRO \
#       --data-root /home/jskang/sigma/liberopro_data \
#       [--num-variants 5] [--num-inits 50] [--nproc 4] [--suites libero_spatial,libero_object]
#
# This script is idempotent: if a bddl/init file already exists, it skips.
#
# Notes:
#   - Generating init_states requires `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`
#     (set automatically), and ~50 mujoco env startups per task variant. For
#     50 inits × 5 variants × 10 tasks per suite × 4 suites this is ~10,000
#     env startups (a few hours total on one machine; parallelisable).

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
import zipfile
from multiprocessing import Pool, set_start_method
from pathlib import Path

DEFAULT_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--libero-pro-root",
        default="/home/jskang/sigma/LIBERO-PRO",
        help="Path to a clone of https://github.com/Zxy-MLlab/LIBERO-PRO",
    )
    p.add_argument(
        "--data-root",
        default="/home/jskang/sigma/liberopro_data",
        help="Where to write bddl_files/<suite>_swap and init_files/<suite>_swap",
    )
    p.add_argument(
        "--suites",
        default=",".join(DEFAULT_SUITES),
        help="Comma-separated suite names. Default: all 4",
    )
    p.add_argument(
        "--num-variants",
        type=int,
        default=5,
        help="How many swap variants per task. Default 5.",
    )
    p.add_argument(
        "--num-inits",
        type=int,
        default=50,
        help="Init states per (task, variant). Default 50 (matches standard LIBERO).",
    )
    p.add_argument(
        "--nproc",
        type=int,
        default=4,
        help="Worker processes for init state generation.",
    )
    p.add_argument(
        "--seed-base",
        type=int,
        default=42,
        help="Base seed for swap variant RNG.",
    )
    p.add_argument(
        "--skip-bddl",
        action="store_true",
        help="Skip bddl regeneration (useful if you only want to (re)build init files).",
    )
    p.add_argument(
        "--skip-init",
        action="store_true",
        help="Skip init_state generation (useful if you only want bddls).",
    )
    return p.parse_args()


def _generate_bddls(args: argparse.Namespace, suites: list[str]) -> None:
    sys.path.insert(0, args.libero_pro_root)
    from perturbation import BDDLParser, SwapPerturbator  # noqa: WPS433

    src_bddl_root = Path(args.libero_pro_root) / "libero" / "libero" / "bddl_files"
    swap_cfg = Path(args.libero_pro_root) / "libero_ood" / "ood_spatial_relation.yaml"
    if not swap_cfg.is_file():
        raise SystemExit(f"missing swap config: {swap_cfg}")

    written = 0
    for suite in suites:
        src_dir = src_bddl_root / suite
        dst_dir = Path(args.data_root) / "bddl_files" / f"{suite}_swap"
        dst_dir.mkdir(parents=True, exist_ok=True)
        bddl_files = sorted(src_dir.glob("*.bddl"))
        print(f"[bddl][{suite}] {len(bddl_files)} tasks -> {dst_dir}")
        for bp in bddl_files:
            task_name = bp.stem
            content = bp.read_text()
            for v in range(args.num_variants):
                out = dst_dir / f"{task_name}_swap{v}.bddl"
                if out.is_file():
                    continue
                random.seed(
                    args.seed_base + v * 1000 + hash(task_name) % 100000
                )
                parser = BDDLParser(content)
                pert = SwapPerturbator(parser, str(swap_cfg))
                new_content = pert.perturb(task_suite_name=suite, task_name=task_name)
                out.write_text(new_content)
                written += 1
    print(f"[bddl] wrote {written} files")


def _gen_init_one(t):
    bddl_path, init_path, num_inits = t
    if Path(init_path).exists():
        return f"SKIP {Path(bddl_path).name}"
    from libero.libero.envs import OffScreenRenderEnv  # noqa: WPS433
    import numpy as np  # noqa: WPS433

    states = []
    for _ in range(num_inits):
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_path, camera_heights=128, camera_widths=128
        )
        states.append(env.get_sim_state())
        env.close()
    if not states:
        return f"EMPTY {Path(bddl_path).name}"
    states = np.array(states)
    Path(init_path).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(init_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("archive/data.pkl", pickle.dumps(states))
        zf.writestr("archive/version", b"1")
    return f"OK {Path(bddl_path).name} ({len(states)})"


def _generate_init_states(args: argparse.Namespace, suites: list[str]) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    tasks = []
    for suite in suites:
        bddl_dir = Path(args.data_root) / "bddl_files" / f"{suite}_swap"
        init_dir = Path(args.data_root) / "init_files" / f"{suite}_swap"
        init_dir.mkdir(parents=True, exist_ok=True)
        for bf in sorted(bddl_dir.glob("*.bddl")):
            tasks.append((str(bf), str(init_dir / f"{bf.stem}.pruned_init"), args.num_inits))

    print(f"[init] {len(tasks)} (task, variant) pairs to process")

    set_start_method("spawn", force=True)
    with Pool(args.nproc) as p:
        for i, r in enumerate(p.imap_unordered(_gen_init_one, tasks, chunksize=1)):
            print(f"  [{i + 1}/{len(tasks)}] {r}", flush=True)


def main() -> None:
    args = _parse_args()
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    print(f"libero-pro-root : {args.libero_pro_root}")
    print(f"data-root       : {args.data_root}")
    print(f"suites          : {suites}")

    if not args.skip_bddl:
        _generate_bddls(args, suites)
    if not args.skip_init:
        _generate_init_states(args, suites)
    print("DONE")


if __name__ == "__main__":
    main()
