#!/bin/bash
# sigma-QRT G2 v3 CQL conservative penalty sweep (3 α_cql x 2 seeds = 6 lanes).
#
# Goal
# ----
# G2 v2 critic capacity sweep best (med 1024/4) topped out at SR=0.18,
# well below B0=0.48. Diagnosis: critic q_std is flat (~0.005) because
# IQL only queries Q at dataset actions — there is no pressure for Q
# to be LOWER on OOD actions than buffer-supported ones. Without that
# pressure, the residual actor's Q-max gradient cannot prefer
# in-distribution Δ over OOD Δ, and Δ drifts out of the data manifold.
#
# Hypothesis
# ----------
# CQL adds an explicit "logsumexp_a Q(s,a) − Q(s,a_data)" penalty term
# to the critic loss. This:
#   - pushes Q(s, OOD_a) DOWN
#   - leaves Q(s, in_dist_a) anchored by the IQL TD target
#   - gives the residual actor a real "prefer-in-distribution" gradient
# Expected: less Δ OOD drift → SR > G2 v2 best (0.18), hopefully toward
# B0=0.48 floor; α_cql controls strength of the OOD penalty.
#
# Algorithm choice
# ----------------
# Full CQL (not CQL-lite or TD3+BC) implemented as additive term on
# top of IQL Q-loss. IQL still provides safe V-bootstrap (no OOD Q in
# TD target); CQL adds OOD pressure on top. Canonical "IQL+CQL" combo
# for narrow-buffer offline RL.
#
# Variant: a1_frozen_encoder + IQL + use_residual_actor=true + use_cql=true
# + encoder cache + medium critic (hidden=1024, layers=4, from G2 v2 best).
#
# Layout (6 training lanes on GPU 0-5; coord uses GPU 6,6,7,7 — 4 workers):
#   - GPU 0: cql_a05_seed1   (alpha_cql=0.5, seed=1)
#   - GPU 1: cql_a05_seed2   (alpha_cql=0.5, seed=2)
#   - GPU 2: cql_a10_seed1   (alpha_cql=1.0, seed=1)
#   - GPU 3: cql_a10_seed2   (alpha_cql=1.0, seed=2)
#   - GPU 4: cql_a50_seed1   (alpha_cql=5.0, seed=1)
#   - GPU 5: cql_a50_seed2   (alpha_cql=5.0, seed=2)
#
# Hparams from v6_v2 best + G2 v2 med: tau=0.8, iql_beta=3, batch=128,
# residual_alpha=0.1, critic_hidden=1024, critic_layers=4,
# max_train=10000, save_interval=1000, num_eval=25.
#
# Each lane:
#   1. train 10000 steps with --encoder_ckpt, --use_iql, --use_cql,
#      residual actor, seed per-lane, --save_interval 1000.
#   2. final async vectorized eval (25 ep, 5 envs, seed 1) on final ckpt.pt.
#   3. touch .lane_done — coordinator drain signal.
#
# After all 6 sentinels, a g2_v3_cql_summary.json is compiled.

set -uo pipefail   # no -e — continue if one lane crashes

cd ~/data/jskang/sigma-qrt/RLinf
source ~/data/jskang/sigma/.venv/bin/activate
export PYTHONPATH=$(pwd)

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=runs/g2_v3_cql_${STAMP}
LOG_DIR=logs/g2_v3_cql_${STAMP}
mkdir -p ${OUT} ${LOG_DIR}
echo "[g2-v3-cql] OUT=${OUT}" | tee ${OUT}/launcher.log
echo "[g2-v3-cql] LOG_DIR=${LOG_DIR}" | tee -a ${OUT}/launcher.log

CFG=examples/embodiment/config/libero_long_qrt_iql_openpi_pi05.yaml
BUF=data/sigma_qrt/libero_long/transitions_B2500.pkl
ENC_CKPT=cached/encoder_warmup_v1.pt
TAU=0.8
BETA=3
SAVE_INTERVAL=1000
RES_ALPHA=0.1
RES_REG=0.0
CRITIC_HID=1024
CRITIC_LAY=4

# ----- helper: lane = train then final eval (sequential within lane) -----
run_lane() {
    local GPU=$1; local ALPHA_CQL=$2; local SEED=$3; local TAG=$4
    local LANE_NAME=g2v3_${TAG}_seed${SEED}
    local LANE_OUT=${OUT}/${LANE_NAME}
    local LANE_TAG=${LOG_DIR}/${LANE_NAME}
    mkdir -p ${LANE_OUT}
    (
        export CUDA_VISIBLE_DEVICES=${GPU}
        export MUJOCO_EGL_DEVICE_ID=${GPU}
        echo "[lane g${GPU} ${LANE_NAME}] $(date +%Y-%m-%d_%H:%M:%S) train start variant=a1_frozen_encoder tau=${TAU} beta=${BETA} seed=${SEED} alpha_cql=${ALPHA_CQL} critic=${CRITIC_HID}/${CRITIC_LAY} alpha_res=${RES_ALPHA} encoder_cache=${ENC_CKPT}" >> ${LANE_TAG}.log
        python examples/embodiment/run_qrt_offline.py \
            --config ${CFG} \
            --variant a1_frozen_encoder \
            --use_iql \
            --use_cql \
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
            --override "training.cql_alpha=${ALPHA_CQL}" \
            --override "training.cql_num_random=10" \
            --override "training.cql_noise_std=0.3" \
            --override "model.critic_hidden=${CRITIC_HID}" \
            --override "model.critic_layers=${CRITIC_LAY}" \
            --override "model.v_hidden=${CRITIC_HID}" \
            --override "model.v_layers=${CRITIC_LAY}" \
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
    echo "[g2-v3-cql] launched lane g${GPU} ${LANE_NAME} pid=$(cat ${LANE_OUT}/.lane.pid)" | tee -a ${OUT}/launcher.log
}

# ----- launch 6 lanes: 3 α_cql x 2 seeds -----
run_lane 0 0.5 1 a05
run_lane 1 0.5 2 a05
run_lane 2 1.0 1 a10
run_lane 3 1.0 2 a10
run_lane 4 5.0 1 a50
run_lane 5 5.0 2 a50

# ----- wait for all 6 sentinels -----
EXPECTED=6
echo "[g2-v3-cql] $(date +%Y-%m-%d_%H:%M:%S) all ${EXPECTED} lanes launched; watching for sentinels..." | tee -a ${OUT}/launcher.log
while true; do
    DONE=$(find ${OUT}/ -maxdepth 2 -name .lane_done 2>/dev/null | wc -l)
    if [ "${DONE}" -ge "${EXPECTED}" ]; then
        echo "[g2-v3-cql] $(date +%Y-%m-%d_%H:%M:%S) all ${EXPECTED} lanes done" | tee -a ${OUT}/launcher.log
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
(out / "g2_v3_cql_summary.json").write_text(json.dumps(summary, indent=2))
print("=== G2 v3 CQL SWEEP SUMMARY ===")
print(json.dumps(summary, indent=2))
PYEOF

echo "[g2-v3-cql] $(date +%Y-%m-%d_%H:%M:%S) CHAIN DONE" | tee -a ${OUT}/launcher.log
