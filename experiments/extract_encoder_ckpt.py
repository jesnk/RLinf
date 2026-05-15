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

"""σ-QRT G2 follow-up — extract pre-warmed encoder + decoder from a Stage-2 ckpt.

Reads a full σ-QRT ckpt produced by ``examples/embodiment/run_qrt_offline.py``
(payload structure: ``encoder``/``decoder``/``actor``/``critic``/``cfg``/...)
and writes a stripped ckpt that ONLY contains the encoder + decoder state
dicts plus the model-cfg slice required to validate dim compatibility at
load time. Downstream runs can then ``--encoder_ckpt`` this artifact and
skip Stage-1 warmup entirely (≈100 min/lane saved on B200).

CLI
---
    python experiments/extract_encoder_ckpt.py \\
        --src_ckpt PATH_TO_STAGE2_CKPT.pt \\
        --output PATH_FOR_STRIPPED_CKPT.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _count_params(state_dict: dict) -> int:
    total = 0
    for t in state_dict.values():
        if isinstance(t, torch.Tensor):
            total += t.numel()
    return total


def _extract_model_cfg(cfg_obj) -> dict:
    """Extract the model sub-config slice required for compat checking.

    cfg_obj may be a plain dict (already serialized via OmegaConf.to_container)
    or an OmegaConf node. We accept both and return a plain dict containing
    only the encoder/decoder dim-determining keys. Keys are kept verbatim
    so the load-side mismatch check just compares dicts.
    """
    if cfg_obj is None:
        return {}
    # OmegaConf node → container
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(cfg_obj, DictConfig):
            cfg_obj = OmegaConf.to_container(cfg_obj, resolve=True)
    except ImportError:
        pass
    if not isinstance(cfg_obj, dict):
        return {}
    model = cfg_obj.get("model", {}) or {}
    # Keys that affect encoder/decoder shape. Any mismatch invalidates the
    # cached state_dict.
    keep = (
        "token_dim",
        "encoder_layers",
        "encoder_heads",
        "encoder_ffn",
        "decoder_layers",
        "decoder_heads",
        "decoder_ffn",
        "decoder_max_len",
    )
    return {k: model[k] for k in keep if k in model}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Extract encoder + decoder state_dicts from a σ-QRT ckpt."
    )
    p.add_argument("--src_ckpt", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    args = p.parse_args(argv)

    src_path = Path(args.src_ckpt)
    if not src_path.is_file():
        print(f"[extract] error: src_ckpt not found: {src_path}", file=sys.stderr)
        return 2

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[extract] loading {src_path} ...")
    payload = torch.load(src_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        print(
            f"[extract] error: ckpt is not a dict ({type(payload).__name__})",
            file=sys.stderr,
        )
        return 3

    # Allow nested {"payload": {...}} or {"state_dict": {...}} envelopes,
    # falling back to the top-level dict.
    if "encoder" not in payload:
        for nest_key in ("payload", "state_dict"):
            inner = payload.get(nest_key)
            if isinstance(inner, dict) and "encoder" in inner:
                payload = inner
                print(f"[extract] using nested key '{nest_key}'")
                break

    if "encoder" not in payload or "decoder" not in payload:
        print(
            "[extract] error: ckpt missing 'encoder' or 'decoder' state_dict; "
            f"top-level keys={sorted(payload.keys())}",
            file=sys.stderr,
        )
        return 4

    encoder_sd = payload["encoder"]
    decoder_sd = payload["decoder"]
    cfg_partial = {"model": _extract_model_cfg(payload.get("cfg"))}

    out_payload = {
        "encoder": encoder_sd,
        "decoder": decoder_sd,
        "cfg_partial": cfg_partial,
        # Provenance — non-load-bearing but useful when you need to trace
        # which Stage-2 run produced the warmup.
        "src_ckpt": str(src_path.resolve()),
        "src_variant": payload.get("variant"),
    }

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(out_payload, tmp_path)
    import os

    os.replace(tmp_path, out_path)

    enc_n = _count_params(encoder_sd)
    dec_n = _count_params(decoder_sd)
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(
        f"extracted: encoder_params={enc_n}, decoder_params={dec_n}, "
        f"file_size_MB={file_size_mb:.2f}"
    )
    print(f"[extract] cfg_partial.model={cfg_partial['model']}")
    print(f"[extract] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
