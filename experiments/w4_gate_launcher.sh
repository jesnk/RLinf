#!/bin/bash
# σ-QRT W4 gate launcher — runs 6 trainings + 7 evals sequentially on GPU 0.
# Output: runs/w4_gate_2026XXXX/{qrt,a1}_seed{1,2,3}/{metrics.json,ckpt.pt,eval_sr.json}
#                                b0/eval_sr.json
#         w4_gate_summary.json (compiled at end)

set -uo pipefail   # no -e because we want to continue if one variant crashes

cd ~/data/jskang/sigma-qrt/RLinf
source ~/data/jskang/sigma/.venv/bin/activate
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=runs/w4_gate_${STAMP}
mkdir -p ${OUT}
echo "[w4_gate] OUT=${OUT}" | tee ${OUT}/launcher.log

CFG=examples/embodiment/config/libero_long_qrt_openpi_pi05.yaml
BUF=data/sigma_qrt/libero_long/transitions_B2500.pkl

# ---------- σ-QRT × 3 seeds ----------
for SEED in 1 2 3; do
    VAR_OUT=${OUT}/qrt_seed${SEED}
    echo "[w4_gate $(date +%H:%M:%S)] σ-QRT seed=${SEED} start" | tee -a ${OUT}/launcher.log
    python examples/embodiment/run_qrt_offline.py \
        --config ${CFG} \
        --variant qrt \
        --output_dir ${VAR_OUT} \
        --override "data.offline_buffer_path=${BUF}" \
        --override "data.capacity=8000" \
        --override "training.batch_size=128" \
        --override "training.warmup_steps=2000" \
        --override "training.max_train_steps=10000" \
        --override "seed=${SEED}" \
        --no_wandb \
        > ${VAR_OUT}_train.log 2>&1
    TR_RC=$?
    echo "[w4_gate $(date +%H:%M:%S)] σ-QRT seed=${SEED} train_rc=${TR_RC}" | tee -a ${OUT}/launcher.log

    if [ -f ${VAR_OUT}/ckpt.pt ]; then
        echo "[w4_gate $(date +%H:%M:%S)] σ-QRT seed=${SEED} eval start" | tee -a ${OUT}/launcher.log
        python examples/embodiment/eval_libero_sr.py \
            --config ${CFG} \
            --ckpt ${VAR_OUT}/ckpt.pt \
            --num_eval 25 \
            --seed 1 \
            --output ${VAR_OUT}/eval_sr.json \
            > ${VAR_OUT}_eval.log 2>&1
        EV_RC=$?
        echo "[w4_gate $(date +%H:%M:%S)] σ-QRT seed=${SEED} eval_rc=${EV_RC}" | tee -a ${OUT}/launcher.log
    fi
done

# ---------- A1 × 3 seeds ----------
for SEED in 1 2 3; do
    VAR_OUT=${OUT}/a1_seed${SEED}
    echo "[w4_gate $(date +%H:%M:%S)] A1 seed=${SEED} start" | tee -a ${OUT}/launcher.log
    python examples/embodiment/run_qrt_offline.py \
        --config ${CFG} \
        --variant a1_frozen_encoder \
        --output_dir ${VAR_OUT} \
        --override "data.offline_buffer_path=${BUF}" \
        --override "data.capacity=8000" \
        --override "training.batch_size=128" \
        --override "training.warmup_steps=2000" \
        --override "training.max_train_steps=10000" \
        --override "seed=${SEED}" \
        --no_wandb \
        > ${VAR_OUT}_train.log 2>&1
    TR_RC=$?
    echo "[w4_gate $(date +%H:%M:%S)] A1 seed=${SEED} train_rc=${TR_RC}" | tee -a ${OUT}/launcher.log

    if [ -f ${VAR_OUT}/ckpt.pt ]; then
        echo "[w4_gate $(date +%H:%M:%S)] A1 seed=${SEED} eval start" | tee -a ${OUT}/launcher.log
        python examples/embodiment/eval_libero_sr.py \
            --config ${CFG} \
            --ckpt ${VAR_OUT}/ckpt.pt \
            --num_eval 25 \
            --seed 1 \
            --output ${VAR_OUT}/eval_sr.json \
            > ${VAR_OUT}_eval.log 2>&1
        EV_RC=$?
        echo "[w4_gate $(date +%H:%M:%S)] A1 seed=${SEED} eval_rc=${EV_RC}" | tee -a ${OUT}/launcher.log
    fi
done

# ---------- B0 zero-shot eval ----------
echo "[w4_gate $(date +%H:%M:%S)] B0 zero-shot eval start" | tee -a ${OUT}/launcher.log
mkdir -p ${OUT}/b0
python examples/embodiment/eval_libero_sr.py \
    --config ${CFG} \
    --num_eval 25 \
    --seed 1 \
    --output ${OUT}/b0/eval_sr.json \
    > ${OUT}/b0_eval.log 2>&1
echo "[w4_gate $(date +%H:%M:%S)] B0 eval_rc=$?" | tee -a ${OUT}/launcher.log

# ---------- Compile summary ----------
python -c "
import json
from pathlib import Path
out = Path('${OUT}')
summary = {'output_dir': str(out)}
for var in ['qrt', 'a1']:
    summary[var] = {}
    for s in (1, 2, 3):
        p = out / f'{var}_seed{s}' / 'eval_sr.json'
        if p.exists():
            summary[var][f'seed{s}'] = json.loads(p.read_text())
        else:
            summary[var][f'seed{s}'] = {'error': 'no eval file'}
b0p = out / 'b0' / 'eval_sr.json'
summary['b0'] = json.loads(b0p.read_text()) if b0p.exists() else {'error': 'no eval file'}
(out / 'w4_gate_summary.json').write_text(json.dumps(summary, indent=2))
print('=== W4 GATE SUMMARY ===')
print(json.dumps(summary, indent=2))
" 2>&1 | tee -a ${OUT}/launcher.log

echo "[w4_gate $(date +%H:%M:%S)] FULL CHAIN DONE" | tee -a ${OUT}/launcher.log
