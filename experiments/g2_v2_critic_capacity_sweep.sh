#!/bin/bash
# sigma-QRT G2 v2 critic capacity sweep (3 capacities x 2 seeds = 6 lanes).
#
# Goal
# ----
# Current critic MLP (yaml default hidden=256, layers=2) may be too SMALL
# to learn an informative Q surface in our IQL+residual setup. G3 qrt
# critic q_std grew 7x over 9k steps (step2k=0.0018 -> step9k=0.013) but
# never crossed 0.05 ("informative"). Frozen baseline q_std stays at 0.004.
# Hypothesis: bigger critic might learn informative Q surface faster,
# breaking the q_std flat ceiling.
#
# Variant: a1_frozen_encoder + IQL + use_residual_actor=true + encoder cache.
# v_hidden / v_layers mirror critic (default yaml behavior).
#
# Sweep (note: 'small' label uses the task-spec sweep values 512/3 — NOT
# the yaml default 256/2. So 'small' is already a modest cap bump vs the
# G2/G3 baseline. medium=1024/4, large=2048/5 are the real test of the
# capacity hypothesis.)
#
# Layout (6 training lanes on GPU 0-5; coord uses GPU 6,6,7,7 — 4 workers):
#   - GPU 0: cap_small_seed1   (hidden=512,  layers=3, seed=1)
#   - GPU 1: cap_small_seed2   (hidden=512,  layers=3, seed=2)
#   - GPU 2: cap_med_seed1     (hidden=1024, layers=4, seed=1)
#   - GPU 3: cap_med_seed2     (hidden=1024, layers=4, seed=2)
#   - GPU 4: cap_large_seed1   (hidden=2048, layers=5, seed=1)
#   - GPU 5: cap_large_seed2   (hidden=2048, layers=5, seed=2)
#
# Hparams from v6_v2 best: tau=0.8, iql_beta=3, batch_size=128,
# residual_alpha=0.1, max_train=10000, save_interval=1000, num_eval=25.
#
# Each lane:
#   1. train 10000 steps with --encoder_ckpt, --use_iql, residual actor,
#      seed per-lane, --save_interval 1000.
#   2. final async vectorized eval (25 ep, 5 envs, seed 1) on final ckpt.pt.
#   3. touch .lane_done — coordinator drain signal.
#
# After all 6 sentinels, a g2_v2_capacity_summary.json is compiled.

set -uo pipefail   # no -e — continue if one lane crashes

cd ~/data/jskang/sigma-qrt/RLinf
source ~/data/jskang/sigma/.venv/bin/activate
export PYTHONPATH=$(pwd)

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=runs/g2_v2_critic_capacity_${STAMP}
LOG_DIR=logs/g2_v2_critic_capacity_${STAMP}
mkdir -p ${OUT} ${LOG_DIR}
echo "[g2-v2-cap] OUT=${OUT}" | tee ${OUT}/launcher.log
echo "[g2-v2-cap] LOG_DIR=${LOG_DIR}" | tee -a ${OUT}/launcher.log

CFG=examples/embodiment/config/libero_long_qrt_iql_openpi_pi05.yaml
BUF=data/sigma_qrt/libero_long/transitions_B2500.pkl
ENC_CKPT=cached/encoder_warmup_v1.pt
TAU=0.8
BETA=3
SAVE_INTERVAL=1000
RES_ALPHA=0.1
RES_REG=0.0

# ----- helper: lane = train then final eval (sequential within lane) -----
run_lane() {
    local GPU=$1; local HID=$2; local LAY=$3; local SEED=$4; local TAG=$5
    local LANE_NAME=${TAG}_seed${SEED}
    local LANE_OUT=${OUT}/${LANE_NAME}
    local LANE_TAG=${LOG_DIR}/${LANE_NAME}
    mkdir -p ${LANE_OUT}
    (
        export CUDA_VISIBLE_DEVICES=${GPU}
        export MUJOCO_EGL_DEVICE_ID=${GPU}
        echo "[lane g${GPU} ${LANE_NAME}] $(date +%Y-%m-%d_%H:%M:%S) train start variant=a1_frozen_encoder tau=${TAU} beta=${BETA} seed=${SEED} hidden=${HID} layers=${LAY} alpha=${RES_ALPHA} encoder_cache=${ENC_CKPT}" >> ${LANE_TAG}.log
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
            --override "training.residual_alpha=${RES_ALPHA}" \
            --override "training.residual_reg=${RES_REG}" \
            --override "model.critic_hidden=${HID}" \
            --override "model.critic_layers=${LAY}" \
            --override "model.v_hidden=${HID}" \
            --override "model.v_layers=${LAY}" \
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
    echo "[g2-v2-cap] launched lane g${GPU} ${LANE_NAME} pid=$(cat ${LANE_OUT}/.lane.pid)" | tee -a ${OUT}/launcher.log
}

# ----- launch 6 lanes: 3 capacities x 2 seeds -----
run_lane 0 512  3 1 cap_small
run_lane 1 512  3 2 cap_small
run_lane 2 1024 4 1 cap_med
run_lane 3 1024 4 2 cap_med
run_lane 4 2048 5 1 cap_large
run_lane 5 2048 5 2 cap_large

# ----- wait for all 6 sentinels -----
EXPECTED=6
echo "[g2-v2-cap] $(date +%Y-%m-%d_%H:%M:%S) all ${EXPECTED} lanes launched; watching for sentinels..." | tee -a ${OUT}/launcher.log
while true; do
    DONE=$(find ${OUT}/ -maxdepth 2 -name .lane_done 2>/dev/null | wc -l)
    if [ "${DONE}" -ge "${EXPECTED}" ]; then
        echo "[g2-v2-cap] $(date +%Y-%m-%d_%H:%M:%S) all ${EXPECTED} lanes done" | tee -a ${OUT}/launcher.log
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
(out / "g2_v2_capacity_summary.json").write_text(json.dumps(summary, indent=2))
print("=== G2 v2 CRITIC CAPACITY SWEEP SUMMARY ===")
print(json.dumps(summary, indent=2))
PYEOF

echo "[g2-v2-cap] $(date +%Y-%m-%d_%H:%M:%S) CHAIN DONE" | tee -a ${OUT}/launcher.log
