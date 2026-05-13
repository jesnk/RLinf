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

"""σ-QRT Task 11 — small helpers shared between training/eval entry scripts.

Only the pieces actually used by `run_qrt_offline.py` and `eval_libero_sr.py`
live here. The collect script (Task 8) keeps its own copies to remain
standalone.
"""

from __future__ import annotations

import os
from typing import Any


def bootstrap_gl_env() -> None:
    """Auto-bootstrap EGL/NVIDIA before robosuite is imported.

    Mirrors `examples/embodiment/collect_base_vla_rollouts.py::_bootstrap_gl_env`.
    Safe to call from a script-level import; respects pre-existing MUJOCO_GL.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    os.environ.setdefault("LIBERO_TYPE", "standard")
    os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")

    if os.environ["MUJOCO_GL"].lower() != "egl":
        return

    nvidia_egl_root_candidates = [
        os.environ.get("NVIDIA_EGL_ROOT"),
        "/NHNHOME/WORKSPACE/0426030005_A/jisuan/dummy/.nvidia-egl-root",
    ]
    for root in nvidia_egl_root_candidates:
        if not root:
            continue
        icd_dir = f"{root}/usr/share/glvnd/egl_vendor.d"
        lib_dir = f"{root}/usr/lib/x86_64-linux-gnu"
        if os.path.isfile(f"{icd_dir}/10_nvidia.json") and os.path.isfile(
            f"{lib_dir}/libEGL_nvidia.so.0"
        ):
            os.environ.setdefault("__EGL_VENDOR_LIBRARY_DIRS", icd_dir)
            existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
            if lib_dir not in existing_ld.split(":"):
                os.environ["LD_LIBRARY_PATH"] = (
                    f"{lib_dir}:{existing_ld}" if existing_ld else lib_dir
                )
            break


def parse_override(value: str) -> Any:
    """Parse a CLI override RHS string → bool / int / float / str.

    OmegaConf.update doesn't auto-cast strings, so we coerce explicitly
    matching python-literal precedence: bool first (so "true" doesn't
    parse as a 1-letter string), then int, then float, else string.
    """
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def apply_overrides(cfg, overrides: list[str]) -> None:
    """Apply `key.path=value` overrides to an OmegaConf DictConfig in place."""
    from omegaconf import OmegaConf

    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"bad override (no '='): {ov!r}")
        k, v = ov.split("=", 1)
        parsed = parse_override(v)
        OmegaConf.update(cfg, k, parsed, merge=False)
