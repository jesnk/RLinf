#!/bin/bash
# σ-QRT G2 buffer diversification — diversified rollout collection (4 lanes).
#
# Goal
# ----
# G2 critic showed q_std_mean=0.0040 across action perturbations on the
# original π0.5-only buffer (transitions_B2500.pkl, 4600 chunks, 100 eps).
# Critic-driven Q-max signal is dead because the buffer's action support
# is too narrow. Diversify by adding lanes with different action-noise
# levels + a fully-random rollout lane, then merge into a buffer the same
# size as the original for direct comparison.
#
# Diversification knobs (collector CLI flags added in this branch):
#   --action_noise_std FLOAT   additive Gaussian σ on executed chunk
#                              (ref_action stored in buffer is un-noised)
#   --random_action_frac FLOAT per-chunk probability of replacing executed
#                              action with uniform [-1, 1]^A
#
# Note: π0.5's flow-matching SDE temperature (config noise_level=0.5) is
# hard to reach without patching openpi.sample_actions. Post-hoc Gaussian
# action noise is a first-order proxy and is sufficient for buffer
# diversification (the goal is wider action distribution support — not a
# faithful reproduction of a higher-temperature π0.5 policy). The "tXX"
# lane labels below are the action_noise_std value (not a flow-SDE temp).
#
# Lanes:
#   GPU 1: t005   action_noise_std=0.05  random_frac=0.0  25 eps
#   GPU 2: t010   action_noise_std=0.10  random_frac=0.0  25 eps
#   GPU 3: t020   action_noise_std=0.20  random_frac=0.0  25 eps
#   GPU 4: rand   action_noise_std=0.0   random_frac=1.0  20 eps
#
# Why not GPU 0: occupied by a stray zero_delta probe (eval_libero_sr.py)
# on the prior G2 residual chain. GPU 5,6,7 reserved for whatever the
# follow-up coord pool will use.
#
# Each lane:
#   1. Activate venv, set CUDA_VISIBLE_DEVICES + MUJOCO_EGL_DEVICE_ID
#      to its GPU.
#   2. Run collect_base_vla_rollouts.py with the lane's diversification
#      flags. seed=lane_index for reproducibility.
#   3. Touch ${LANE_OUT}/.lane_done sentinel.
#
# After all 4 sentinels, merge_diversified_buffers.py is launched
# separately by the orchestrator (NOT by this script — keeps the launcher
# focused on collection only).

set -uo pipefail

cd ~/data/jskang/sigma-qrt/RLinf
source ~/data/jskang/sigma/.venv/bin/activate
export PYTHONPATH=$(pwd)

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR=data/sigma_qrt/libero_long/diversified_${STAMP}
LOG_DIR=logs/g2_diversified_collect_${STAMP}
mkdir -p ${OUT_DIR} ${LOG_DIR}
echo "[g2-div-collect] OUT_DIR=${OUT_DIR}" | tee ${LOG_DIR}/launcher.log
echo "[g2-div-collect] LOG_DIR=${LOG_DIR}" | tee -a ${LOG_DIR}/launcher.log

CFG=examples/embodiment/config/libero_long_collect_rollouts_pi05.yaml

run_lane() {
    local GPU=$1; local TAG=$2; local NOISE=$3; local RAND_FRAC=$4; local NUM_EPS=$5; local SEED=$6
    local LANE_OUT=${OUT_DIR}/${TAG}.pkl
    local LANE_TAG=${LOG_DIR}/${TAG}
    (
        export CUDA_VISIBLE_DEVICES=${GPU}
        export MUJOCO_EGL_DEVICE_ID=${GPU}
        echo "[lane g${GPU} ${TAG}] $(date +%Y-%m-%d_%H:%M:%S) collect start noise_std=${NOISE} rand_frac=${RAND_FRAC} num_eps=${NUM_EPS} seed=${SEED}" >> ${LANE_TAG}.log
        python examples/embodiment/collect_base_vla_rollouts.py \
            --config ${CFG} \
            --num_episodes ${NUM_EPS} \
            --output ${LANE_OUT} \
            --seed ${SEED} \
            --action_noise_std ${NOISE} \
            --random_action_frac ${RAND_FRAC} \
            > ${LANE_TAG}_collect.log 2>&1
        local RC=$?
        echo "[lane g${GPU} ${TAG}] $(date +%Y-%m-%d_%H:%M:%S) collect rc=${RC}" >> ${LANE_TAG}.log
        touch ${OUT_DIR}/.${TAG}.lane_done
        echo "[lane g${GPU} ${TAG}] $(date +%Y-%m-%d_%H:%M:%S) DONE" >> ${LANE_TAG}.log
    ) &
    echo $! > ${OUT_DIR}/.${TAG}.lane.pid
    echo "[g2-div-collect] launched lane g${GPU} ${TAG} pid=$(cat ${OUT_DIR}/.${TAG}.lane.pid)" | tee -a ${LOG_DIR}/launcher.log
}

# ----- launch 4 lanes (avoid GPU 0 — stray zero_delta probe) -----
run_lane 1 t005 0.05 0.0 25 1
run_lane 2 t010 0.10 0.0 25 2
run_lane 3 t020 0.20 0.0 25 3
run_lane 4 rand 0.00 1.0 20 4

EXPECTED=4
echo "[g2-div-collect] $(date +%Y-%m-%d_%H:%M:%S) all ${EXPECTED} lanes launched; watching for sentinels..." | tee -a ${LOG_DIR}/launcher.log
while true; do
    DONE=$(find ${OUT_DIR}/ -maxdepth 1 -name '.*.lane_done' 2>/dev/null | wc -l)
    if [ "${DONE}" -ge "${EXPECTED}" ]; then
        echo "[g2-div-collect] $(date +%Y-%m-%d_%H:%M:%S) all ${EXPECTED} lanes done" | tee -a ${LOG_DIR}/launcher.log
        break
    fi
    sleep 60
done

echo "[g2-div-collect] $(date +%Y-%m-%d_%H:%M:%S) ALL DONE" | tee -a ${LOG_DIR}/launcher.log
echo "[g2-div-collect] OUT_DIR=${OUT_DIR}" | tee -a ${LOG_DIR}/launcher.log
ls -la ${OUT_DIR}/
