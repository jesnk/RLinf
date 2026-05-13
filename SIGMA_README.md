# σ-QRT branch (`feat/sigma-qrt-offline`)

This fork of [RLinf/RLinf](https://github.com/RLinf/RLinf) hosts the σ paper's
QRT track: **frozen VLA + RL-only Q-aware action token + offline RL** (pivot
from σ α1's full-VLA online RL).

## Layout

- Working checkout on brain1: `~/data/jskang/sigma-qrt/RLinf/`
- σ α1 work lives separately at `~/data/jskang/sigma/RLinf/` and is untouched.
- π0.5 LIBERO SFT base ckpt: `ckpts/pi05_libero_sft/` (HF: `RLinf/RLinf-Pi05-LIBERO-SFT`).

## Design / plan docs (offline, Windows delta repo)

- Plan: `work/hbz/sigma/sigma_qrt_w1_w3_implementation_plan.md`
- Design notes + paper references: `work/hbz/sigma/`

## Smoke test

```bash
source /home/jskang/data/jskang/sigma/.venv/bin/activate
cd /home/jskang/data/jskang/sigma-qrt/RLinf
PYTHONPATH=. python -m pytest tests/sigma_qrt/test_setup.py -v
```

The shared sigma venv has the heavy deps (torch+cu128, openpi, jax, rlinf editable);
`PYTHONPATH=.` shadows the editable mapping so this fork's `rlinf/` is imported.
A dedicated `~/data/jskang/sigma-qrt/.venv` will be introduced once we start
modifying `pyproject.toml` for QRT-specific deps.
