#!/bin/bash
# σ-QRT W4 gate PARALLEL launcher — 6 trainings + B0 eval across 7 GPUs (0-6).
# Each variant-seed: train (~140 min) then eval (~12 min) on same GPU lane.
# Sequential within a GPU, parallel across GPU lanes. GPU 7 reserved as buffer.
# Output: runs/w4_gate_parallel_<STAMP>/{qrt,a1}_seed{1,2,3}/ + b0/
#         w4_gate_summary.json compiled after all 7 .lane_done sentinels appear.

set -uo pipefail   # no -e — we want to continue if one lane crashes

cd ~/data/jskang/sigma-qrt/RLinf
source ~/data/jskang/sigma/.venv/bin/activate
export PYTHONPATH=$(pwd)

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=runs/w4_gate_parallel_${STAMP}
LOG_DIR=logs/w4_gate_parallel_${STAMP}
mkdir -p ${OUT} ${LOG_DIR}
echo "[w4_parallel] OUT=${OUT}" | tee ${OUT}/launcher.log
echo "[w4_parallel] LOG_DIR=${LOG_DIR}" | tee -a ${OUT}/launcher.log

CFG=examples/embodiment/config/libero_long_qrt_openpi_pi05.yaml
BUF=data/sigma_qrt/libero_long/transitions_B2500.pkl

# ----- helper: one GPU lane = train then eval (sequential within lane) -----
# Args: GPU VARIANT SEED LANE_LABEL
#   VARIANT     = qrt | a1_frozen_encoder
#   LANE_LABEL  = qrt_seed1 | a1_seed1 | ...   (used for dir+log names; matches sequential script's
#                 naming so the existing summary code (keyed on qrt/a1) keeps working)
run_lane() {
    local GPU=$1; local VARIANT=$2; local SEED=$3; local LANE_LABEL=$4
    local LANE_OUT=${OUT}/${LANE_LABEL}
    local LANE_TAG=${LOG_DIR}/${LANE_LABEL}
    mkdir -p ${LANE_OUT}
    (
        export CUDA_VISIBLE_DEVICES=${GPU}
        export MUJOCO_EGL_DEVICE_ID=${GPU}
        echo "[lane g${GPU} ${LANE_LABEL}] $(date +%Y-%m-%d_%H:%M:%S) train start" >> ${LANE_TAG}.log
        python examples/embodiment/run_qrt_offline.py \
            --config ${CFG} \
            --variant ${VARIANT} \
            --output_dir ${LANE_OUT} \
            --override "data.offline_buffer_path=${BUF}" \
            --override "data.capacity=8000" \
            --override "training.batch_size=128" \
            --override "training.warmup_steps=2000" \
            --override "training.max_train_steps=10000" \
            --override "seed=${SEED}" \
            --bf16 \
            --no_wandb \
            > ${LANE_TAG}_train.log 2>&1
        local TR_RC=$?
        echo "[lane g${GPU} ${LANE_LABEL}] $(date +%Y-%m-%d_%H:%M:%S) train rc=${TR_RC}" >> ${LANE_TAG}.log
        if [ -f ${LANE_OUT}/ckpt.pt ]; then
            echo "[lane g${GPU} ${LANE_LABEL}] $(date +%Y-%m-%d_%H:%M:%S) eval start" >> ${LANE_TAG}.log
            python examples/embodiment/eval_libero_sr.py \
                --config ${CFG} \
                --ckpt ${LANE_OUT}/ckpt.pt \
                --num_eval 25 \
                --seed 1 \
                --output ${LANE_OUT}/eval_sr.json \
                > ${LANE_TAG}_eval.log 2>&1
            local EV_RC=$?
            echo "[lane g${GPU} ${LANE_LABEL}] $(date +%Y-%m-%d_%H:%M:%S) eval rc=${EV_RC}" >> ${LANE_TAG}.log
        else
            echo "[lane g${GPU} ${LANE_LABEL}] $(date +%Y-%m-%d_%H:%M:%S) NO ckpt.pt — skipping eval" >> ${LANE_TAG}.log
        fi
        touch ${LANE_OUT}/.lane_done
        echo "[lane g${GPU} ${LANE_LABEL}] $(date +%Y-%m-%d_%H:%M:%S) DONE" >> ${LANE_TAG}.log
    ) &
    echo $! > ${LANE_OUT}/.lane.pid
    echo "[w4_parallel] launched lane g${GPU} ${LANE_LABEL} pid=$(cat ${LANE_OUT}/.lane.pid)" | tee -a ${OUT}/launcher.log
}

# ----- helper: B0 zero-shot eval (no training) on dedicated GPU -----
run_b0() {
    local GPU=$1
    local LANE_OUT=${OUT}/b0
    local LANE_TAG=${LOG_DIR}/b0
    mkdir -p ${LANE_OUT}
    (
        export CUDA_VISIBLE_DEVICES=${GPU}
        export MUJOCO_EGL_DEVICE_ID=${GPU}
        echo "[b0 g${GPU}] $(date +%Y-%m-%d_%H:%M:%S) eval start" >> ${LANE_TAG}.log
        python examples/embodiment/eval_libero_sr.py \
            --config ${CFG} \
            --num_eval 25 \
            --seed 1 \
            --output ${LANE_OUT}/eval_sr.json \
            > ${LANE_TAG}_eval.log 2>&1
        local EV_RC=$?
        echo "[b0 g${GPU}] $(date +%Y-%m-%d_%H:%M:%S) eval rc=${EV_RC}" >> ${LANE_TAG}.log
        touch ${LANE_OUT}/.lane_done
        echo "[b0 g${GPU}] $(date +%Y-%m-%d_%H:%M:%S) DONE" >> ${LANE_TAG}.log
    ) &
    echo $! > ${LANE_OUT}/.lane.pid
    echo "[w4_parallel] launched b0 g${GPU} pid=$(cat ${LANE_OUT}/.lane.pid)" | tee -a ${OUT}/launcher.log
}

# ----- launch all 7 lanes (GPUs 0-6; GPU 7 reserved as buffer) -----
run_lane 0 qrt               1 qrt_seed1
run_lane 1 qrt               2 qrt_seed2
run_lane 2 qrt               3 qrt_seed3
run_lane 3 a1_frozen_encoder 1 a1_seed1
run_lane 4 a1_frozen_encoder 2 a1_seed2
run_lane 5 a1_frozen_encoder 3 a1_seed3
run_b0   6

# ----- wait for sentinels -----
EXPECTED=7
echo "[w4_parallel] $(date +%Y-%m-%d_%H:%M:%S) all ${EXPECTED} lanes launched, watching for sentinels..." | tee -a ${OUT}/launcher.log
while true; do
    DONE=$(find ${OUT}/ -maxdepth 2 -name .lane_done 2>/dev/null | wc -l)
    if [ "${DONE}" -ge "${EXPECTED}" ]; then
        echo "[w4_parallel] $(date +%Y-%m-%d_%H:%M:%S) all ${EXPECTED} lanes done" | tee -a ${OUT}/launcher.log
        break
    fi
    sleep 60
done

# ----- compile summary -----
python - <<PYEOF 2>&1 | tee -a ${OUT}/launcher.log
import json
from pathlib import Path
out = Path("${OUT}")
summary = {"output_dir": str(out)}
for var in ["qrt", "a1"]:
    summary[var] = {}
    for s in (1, 2, 3):
        p = out / f"{var}_seed{s}" / "eval_sr.json"
        if p.exists():
            summary[var][f"seed{s}"] = json.loads(p.read_text())
        else:
            summary[var][f"seed{s}"] = {"error": "no eval file"}
b0p = out / "b0" / "eval_sr.json"
summary["b0"] = json.loads(b0p.read_text()) if b0p.exists() else {"error": "no eval file"}
(out / "w4_gate_summary.json").write_text(json.dumps(summary, indent=2))
print("=== W4 GATE PARALLEL SUMMARY ===")
print(json.dumps(summary, indent=2))
PYEOF

echo "[w4_parallel] $(date +%Y-%m-%d_%H:%M:%S) CHAIN DONE" | tee -a ${OUT}/launcher.log
