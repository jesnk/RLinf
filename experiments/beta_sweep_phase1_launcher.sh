#!/bin/bash
# σ-QRT β sweep Phase 1 launcher
#
# Layout:
#   - GPU 0/1: σ-QRT β=0.1, seed 1/2
#   - GPU 2/4: σ-QRT β=0.3, seed 1/2  (GPU 3 reserved for other user — DO NOT TOUCH)
#   - GPU 5/6: σ-QRT β=1.0, seed 1/2
#   - GPU 7:   B0 zero-shot eval (50 ep, no training)
#
# Each σ-QRT lane: train 10000 steps then async vectorized eval (25 ep, 5 envs).
# B0 lane: just eval (50 ep, 5 envs, no ckpt).
# Sentinel: ${LANE_OUT}/.lane_done touched after each lane finishes.
# After all 7 sentinels, beta_sweep_summary.json compiled.

set -uo pipefail   # no -e — continue if one lane crashes

cd ~/data/jskang/sigma-qrt/RLinf
source ~/data/jskang/sigma/.venv/bin/activate
export PYTHONPATH=$(pwd)

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=runs/beta_sweep_phase1_${STAMP}
LOG_DIR=logs/beta_sweep_phase1_${STAMP}
mkdir -p ${OUT} ${LOG_DIR}
echo "[β-sweep p1] OUT=${OUT}" | tee ${OUT}/launcher.log
echo "[β-sweep p1] LOG_DIR=${LOG_DIR}" | tee -a ${OUT}/launcher.log

CFG=examples/embodiment/config/libero_long_qrt_openpi_pi05.yaml
BUF=data/sigma_qrt/libero_long/transitions_B2500.pkl

# ----- helper: σ-QRT lane = train then eval (sequential within lane) -----
run_qrt_lane() {
    local GPU=$1; local BETA=$2; local SEED=$3
    local BETA_TAG=$(echo ${BETA} | sed 's/\./p/g')
    local LANE_NAME=qrt_beta${BETA_TAG}_seed${SEED}
    local LANE_OUT=${OUT}/${LANE_NAME}
    local LANE_TAG=${LOG_DIR}/${LANE_NAME}
    mkdir -p ${LANE_OUT}
    (
        export CUDA_VISIBLE_DEVICES=${GPU}
        export MUJOCO_EGL_DEVICE_ID=${GPU}
        echo "[lane g${GPU} ${LANE_NAME}] $(date +%Y-%m-%d_%H:%M:%S) train start beta=${BETA} seed=${SEED}" >> ${LANE_TAG}.log
        python examples/embodiment/run_qrt_offline.py \
            --config ${CFG} \
            --variant qrt \
            --output_dir ${LANE_OUT} \
            --override "data.offline_buffer_path=${BUF}" \
            --override "data.capacity=8000" \
            --override "training.batch_size=128" \
            --override "training.warmup_steps=2000" \
            --override "training.max_train_steps=10000" \
            --override "training.beta_bc=${BETA}" \
            --override "seed=${SEED}" \
            --bf16 \
            --no_wandb \
            > ${LANE_TAG}_train.log 2>&1
        local TR_RC=$?
        echo "[lane g${GPU} ${LANE_NAME}] $(date +%Y-%m-%d_%H:%M:%S) train rc=${TR_RC}" >> ${LANE_TAG}.log
        if [ -f ${LANE_OUT}/ckpt.pt ]; then
            echo "[lane g${GPU} ${LANE_NAME}] $(date +%Y-%m-%d_%H:%M:%S) eval start" >> ${LANE_TAG}.log
            python examples/embodiment/eval_libero_sr.py \
                --config ${CFG} \
                --ckpt ${LANE_OUT}/ckpt.pt \
                --num_eval 25 \
                --num_envs 5 \
                --seed 1 \
                --output ${LANE_OUT}/eval_sr.json \
                > ${LANE_TAG}_eval.log 2>&1
            local EV_RC=$?
            echo "[lane g${GPU} ${LANE_NAME}] $(date +%Y-%m-%d_%H:%M:%S) eval rc=${EV_RC}" >> ${LANE_TAG}.log
        else
            echo "[lane g${GPU} ${LANE_NAME}] $(date +%Y-%m-%d_%H:%M:%S) NO ckpt.pt — skipping eval" >> ${LANE_TAG}.log
        fi
        touch ${LANE_OUT}/.lane_done
        echo "[lane g${GPU} ${LANE_NAME}] $(date +%Y-%m-%d_%H:%M:%S) DONE" >> ${LANE_TAG}.log
    ) &
    echo $! > ${LANE_OUT}/.lane.pid
    echo "[β-sweep p1] launched lane g${GPU} ${LANE_NAME} pid=$(cat ${LANE_OUT}/.lane.pid)" | tee -a ${OUT}/launcher.log
}

# ----- helper: B0 zero-shot eval (no training) on dedicated GPU -----
run_b0_lane() {
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
            --num_eval 50 \
            --num_envs 5 \
            --seed 1 \
            --output ${LANE_OUT}/eval_sr.json \
            > ${LANE_TAG}_eval.log 2>&1
        local EV_RC=$?
        echo "[b0 g${GPU}] $(date +%Y-%m-%d_%H:%M:%S) eval rc=${EV_RC}" >> ${LANE_TAG}.log
        touch ${LANE_OUT}/.lane_done
        echo "[b0 g${GPU}] $(date +%Y-%m-%d_%H:%M:%S) DONE" >> ${LANE_TAG}.log
    ) &
    echo $! > ${LANE_OUT}/.lane.pid
    echo "[β-sweep p1] launched b0 g${GPU} pid=$(cat ${LANE_OUT}/.lane.pid)" | tee -a ${OUT}/launcher.log
}

# ----- launch 6 σ-QRT lanes + 1 B0 (GPU 3 reserved for other user) -----
run_qrt_lane 0 0.1 1
run_qrt_lane 1 0.1 2
run_qrt_lane 2 0.3 1
run_qrt_lane 4 0.3 2
run_qrt_lane 5 1.0 1
run_qrt_lane 6 1.0 2
run_b0_lane  7

# ----- wait for all 7 sentinels -----
EXPECTED=7
echo "[β-sweep p1] $(date +%Y-%m-%d_%H:%M:%S) all ${EXPECTED} lanes launched; watching for sentinels..." | tee -a ${OUT}/launcher.log
while true; do
    DONE=$(find ${OUT}/ -maxdepth 2 -name .lane_done 2>/dev/null | wc -l)
    if [ "${DONE}" -ge "${EXPECTED}" ]; then
        echo "[β-sweep p1] $(date +%Y-%m-%d_%H:%M:%S) all ${EXPECTED} lanes done" | tee -a ${OUT}/launcher.log
        break
    fi
    sleep 60
done

# ----- compile summary -----
python - <<PYEOF 2>&1 | tee -a ${OUT}/launcher.log
import json
from pathlib import Path
out = Path("${OUT}")
summary = {"output_dir": str(out), "lanes": {}}
for d in sorted(out.iterdir()):
    if d.is_dir():
        ep = d / "eval_sr.json"
        if ep.exists():
            try:
                summary["lanes"][d.name] = json.loads(ep.read_text())
            except Exception as e:
                summary["lanes"][d.name] = {"error": f"parse failed: {e}"}
        else:
            summary["lanes"][d.name] = {"error": "no eval_sr.json"}
(out / "beta_sweep_summary.json").write_text(json.dumps(summary, indent=2))
print("=== β SWEEP PHASE 1 SUMMARY ===")
print(json.dumps(summary, indent=2))
PYEOF

echo "[β-sweep p1] $(date +%Y-%m-%d_%H:%M:%S) CHAIN DONE" | tee -a ${OUT}/launcher.log
