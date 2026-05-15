#!/bin/bash
# σ-QRT G2 α-sweep launcher (3 alphas × 2 seeds = 6 lanes, encoder cache).
#
# Goal
# ----
# G2 first round (residual_alpha=0.1) put peak SR=0.36 (g2res_seed2 step7k)
# below B0≈0.48. Hypothesis: α was too small — Δ collapsed (~0.01).
# Push Q-max signal harder by sweeping α ∈ {0.3, 1.0, 3.0}.
#
# Variant: a1_frozen_encoder + IQL + use_residual_actor=true.
# Encoder warm-start: cached/encoder_warmup_v1.pt (skip Stage-1 ≈100min/lane).
#
# Layout (6 training lanes on GPU 0-5; GPU 6+7 reserved for the
# periodic-eval coordinator pool):
#   - GPU 0: a03_seed1   (α=0.3, seed=1)
#   - GPU 1: a03_seed2   (α=0.3, seed=2)
#   - GPU 2: a10_seed1   (α=1.0, seed=1)
#   - GPU 3: a10_seed2   (α=1.0, seed=2)
#   - GPU 4: a30_seed1   (α=3.0, seed=1)
#   - GPU 5: a30_seed2   (α=3.0, seed=2)
#
# Hyperparams from v6_v2 best: tau=0.8, iql_beta=3, batch_size=128.
#
# Each lane:
#   1. train 10000 steps with --encoder_ckpt, --use_iql, residual actor,
#      α from lane, seed from lane, --save_interval 1000.
#   2. final async vectorized eval (25 ep, 5 envs, seed 1) on final ckpt.pt.
#   3. touch .lane_done — coordinator drain signal.

set -uo pipefail   # no -e — continue if one lane crashes

cd ~/data/jskang/sigma-qrt/RLinf
source ~/data/jskang/sigma/.venv/bin/activate
export PYTHONPATH=$(pwd)

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=runs/g2_alpha_sweep_${STAMP}
LOG_DIR=logs/g2_alpha_sweep_${STAMP}
mkdir -p ${OUT} ${LOG_DIR}
echo "[g2-asweep] OUT=${OUT}" | tee ${OUT}/launcher.log
echo "[g2-asweep] LOG_DIR=${LOG_DIR}" | tee -a ${OUT}/launcher.log

CFG=examples/embodiment/config/libero_long_qrt_iql_openpi_pi05.yaml
BUF=data/sigma_qrt/libero_long/transitions_B2500.pkl
ENC_CKPT=cached/encoder_warmup_v1.pt
TAU=0.8
BETA=3
SAVE_INTERVAL=1000
RES_REG=0.0

# ----- helper: lane = train then final eval (sequential within lane) -----
run_lane() {
    local GPU=$1; local ALPHA=$2; local SEED=$3; local TAG=$4
    local LANE_NAME=${TAG}_seed${SEED}
    local LANE_OUT=${OUT}/${LANE_NAME}
    local LANE_TAG=${LOG_DIR}/${LANE_NAME}
    mkdir -p ${LANE_OUT}
    (
        export CUDA_VISIBLE_DEVICES=${GPU}
        export MUJOCO_EGL_DEVICE_ID=${GPU}
        echo "[lane g${GPU} ${LANE_NAME}] $(date +%Y-%m-%d_%H:%M:%S) train start variant=a1_frozen_encoder tau=${TAU} beta=${BETA} seed=${SEED} α=${ALPHA} encoder_cache=${ENC_CKPT}" >> ${LANE_TAG}.log
        python examples/embodiment/run_qrt_offline.py \
            --config ${CFG} \
            --variant a1_frozen_encoder \
            --use_iql \
            --encoder_ckpt ${ENC_CKPT} \
            --output_dir ${LANE_OUT} \
            --override "data.offline_buffer_path=${BUF}" \
            --override "data.capacity=8000" \
            --override "training.batch_size=128" \
            --override "training.warmup_steps=10000" \
            --override "training.max_train_steps=10000" \
            --override "training.iql_tau=${TAU}" \
            --override "training.iql_beta=${BETA}" \
            --override "training.use_residual_actor=true" \
            --override "training.residual_alpha=${ALPHA}" \
            --override "training.residual_reg=${RES_REG}" \
            --override "seed=${SEED}" \
            --save_interval ${SAVE_INTERVAL} \
            --bf16 \
            --no_wandb \
            > ${LANE_TAG}_train.log 2>&1
        local TR_RC=$?
        echo "[lane g${GPU} ${LANE_NAME}] $(date +%Y-%m-%d_%H:%M:%S) train rc=${TR_RC}" >> ${LANE_TAG}.log
        if [ -f ${LANE_OUT}/ckpt.pt ]; then
            echo "[lane g${GPU} ${LANE_NAME}] $(date +%Y-%m-%d_%H:%M:%S) final eval start" >> ${LANE_TAG}.log
            python examples/embodiment/eval_libero_sr.py \
                --config ${CFG} \
                --ckpt ${LANE_OUT}/ckpt.pt \
                --num_eval 25 \
                --num_envs 5 \
                --seed 1 \
                --output ${LANE_OUT}/eval_sr.json \
                > ${LANE_TAG}_eval.log 2>&1
            local EV_RC=$?
            echo "[lane g${GPU} ${LANE_NAME}] $(date +%Y-%m-%d_%H:%M:%S) final eval rc=${EV_RC}" >> ${LANE_TAG}.log
        else
            echo "[lane g${GPU} ${LANE_NAME}] $(date +%Y-%m-%d_%H:%M:%S) NO ckpt.pt — skipping eval" >> ${LANE_TAG}.log
        fi
        touch ${LANE_OUT}/.lane_done
        echo "[lane g${GPU} ${LANE_NAME}] $(date +%Y-%m-%d_%H:%M:%S) DONE" >> ${LANE_TAG}.log
    ) &
    echo $! > ${LANE_OUT}/.lane.pid
    echo "[g2-asweep] launched lane g${GPU} ${LANE_NAME} pid=$(cat ${LANE_OUT}/.lane.pid)" | tee -a ${OUT}/launcher.log
}

# ----- launch 6 lanes: 3 alphas × 2 seeds -----
run_lane 0 0.3 1 a03
run_lane 1 0.3 2 a03
run_lane 2 1.0 1 a10
run_lane 3 1.0 2 a10
run_lane 4 3.0 1 a30
run_lane 5 3.0 2 a30

# ----- wait for all 6 sentinels -----
EXPECTED=6
echo "[g2-asweep] $(date +%Y-%m-%d_%H:%M:%S) all ${EXPECTED} lanes launched; watching for sentinels..." | tee -a ${OUT}/launcher.log
while true; do
    DONE=$(find ${OUT}/ -maxdepth 2 -name .lane_done 2>/dev/null | wc -l)
    if [ "${DONE}" -ge "${EXPECTED}" ]; then
        echo "[g2-asweep] $(date +%Y-%m-%d_%H:%M:%S) all ${EXPECTED} lanes done" | tee -a ${OUT}/launcher.log
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
(out / "g2_alpha_sweep_summary.json").write_text(json.dumps(summary, indent=2))
print("=== G2 α-SWEEP SUMMARY ===")
print(json.dumps(summary, indent=2))
PYEOF

echo "[g2-asweep] $(date +%Y-%m-%d_%H:%M:%S) CHAIN DONE" | tee -a ${OUT}/launcher.log
