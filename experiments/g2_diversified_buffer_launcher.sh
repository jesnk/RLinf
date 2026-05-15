#!/bin/bash
# σ-QRT G2 diversified-buffer retrain (3 residual seeds + 3 baseline seeds).
#
# Goal
# ----
# G2 α-sweep (runs/g2_alpha_sweep_20260515_031832) plus the prior G2 residual
# chain showed Q-max signal is dead: q_std_mean=0.0040 across action perturbations
# on the original π0.5-only buffer (transitions_B2500.pkl). Higher α makes things
# worse (collapse), confirming the actor's residual update has no useful gradient
# signal from the critic.
#
# Hypothesis: narrow buffer = single π0.5 policy → critic sees only one action
# distribution → flat Q surface. Diversifying the buffer's action support should
# give the critic enough signal to learn a non-trivial Q(s,a).
#
# This chain re-runs G2 (α=0.1, tau=0.8, β=3) with the diversified buffer as the
# only delta vs prior chains. If the diversified buffer drives q_std > 0.05 and
# residual SR ≥ B0=0.48, the buffer-fix hypothesis is confirmed.
#
# Variant: a1_frozen_encoder + IQL + use_residual_actor=true.
# Encoder warm-start: cached/encoder_warmup_v1.pt (skip Stage-1 ≈100min/lane).
#
# Layout (6 training lanes on GPU 0-5; GPU 6+7 reserved for the periodic-eval
# coordinator pool launched separately):
#   - GPU 0: g2div_res_seed1   (use_residual_actor=true,  seed=1)
#   - GPU 1: g2div_res_seed2   (use_residual_actor=true,  seed=2)
#   - GPU 2: g2div_res_seed3   (use_residual_actor=true,  seed=3)
#   - GPU 3: g2div_base_seed1  (use_residual_actor=false, seed=1)
#   - GPU 4: g2div_base_seed2  (use_residual_actor=false, seed=2)
#   - GPU 5: g2div_base_seed3  (use_residual_actor=false, seed=3)
#
# Hyperparams from g2_alpha_sweep best (a1_frozen_encoder + IQL):
#   tau=0.8, iql_beta=3, batch_size=128, residual_alpha=0.1 (proven safe),
#   max_train_steps=10000, save_interval=1000.
#
# Each lane:
#   1. train 10000 steps with --encoder_ckpt, --use_iql, residual actor
#      (or not, baseline lanes), seed per-lane, --save_interval 1000.
#   2. final async vectorized eval (25 ep, 5 envs, seed 1) on final ckpt.pt.
#   3. touch .lane_done — coordinator drain signal.
#
# Buffer path is parameterized via env var BUF (defaults to the diversified
# buffer path; override at launch time if testing a different pkl).

set -uo pipefail   # no -e — continue if one lane crashes

cd ~/data/jskang/sigma-qrt/RLinf
source ~/data/jskang/sigma/.venv/bin/activate
export PYTHONPATH=$(pwd)

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=runs/g2_diversified_buffer_${STAMP}
LOG_DIR=logs/g2_diversified_buffer_${STAMP}
mkdir -p ${OUT} ${LOG_DIR}
echo "[g2-div] OUT=${OUT}" | tee ${OUT}/launcher.log
echo "[g2-div] LOG_DIR=${LOG_DIR}" | tee -a ${OUT}/launcher.log

CFG=examples/embodiment/config/libero_long_qrt_iql_openpi_pi05.yaml
BUF=${BUF:-data/sigma_qrt/libero_long/transitions_diversified_B2500plus.pkl}
ENC_CKPT=cached/encoder_warmup_v1.pt
TAU=0.8
BETA=3
ALPHA=0.1
SAVE_INTERVAL=1000
RES_REG=0.0
NUM_TRAIN_STEPS=10000

if [ ! -f "${BUF}" ]; then
    echo "[g2-div] FATAL: buffer not found: ${BUF}" | tee -a ${OUT}/launcher.log
    exit 1
fi
echo "[g2-div] BUF=${BUF}" | tee -a ${OUT}/launcher.log

# ----- helper: lane = train then final eval (sequential within lane) -----
run_lane() {
    local GPU=$1; local USE_RES=$2; local SEED=$3; local TAG=$4
    local LANE_NAME=${TAG}_seed${SEED}
    local LANE_OUT=${OUT}/${LANE_NAME}
    local LANE_TAG=${LOG_DIR}/${LANE_NAME}
    mkdir -p ${LANE_OUT}
    (
        export CUDA_VISIBLE_DEVICES=${GPU}
        export MUJOCO_EGL_DEVICE_ID=${GPU}
        echo "[lane g${GPU} ${LANE_NAME}] $(date +%Y-%m-%d_%H:%M:%S) train start variant=a1_frozen_encoder tau=${TAU} beta=${BETA} seed=${SEED} α=${ALPHA} use_residual=${USE_RES} encoder_cache=${ENC_CKPT} buffer=${BUF}" >> ${LANE_TAG}.log
        python examples/embodiment/run_qrt_offline.py \
            --config ${CFG} \
            --variant a1_frozen_encoder \
            --use_iql \
            --encoder_ckpt ${ENC_CKPT} \
            --output_dir ${LANE_OUT} \
            --override "data.offline_buffer_path=${BUF}" \
            --override "data.capacity=8000" \
            --override "training.batch_size=128" \
            --override "training.warmup_steps=${NUM_TRAIN_STEPS}" \
            --override "training.max_train_steps=${NUM_TRAIN_STEPS}" \
            --override "training.iql_tau=${TAU}" \
            --override "training.iql_beta=${BETA}" \
            --override "training.use_residual_actor=${USE_RES}" \
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
    echo "[g2-div] launched lane g${GPU} ${LANE_NAME} pid=$(cat ${LANE_OUT}/.lane.pid)" | tee -a ${OUT}/launcher.log
}

# ----- launch 6 lanes: 3 res seeds + 3 baseline seeds -----
run_lane 0 true  1 g2div_res
run_lane 1 true  2 g2div_res
run_lane 2 true  3 g2div_res
run_lane 3 false 1 g2div_base
run_lane 4 false 2 g2div_base
run_lane 5 false 3 g2div_base

# ----- wait for all 6 sentinels -----
EXPECTED=6
echo "[g2-div] $(date +%Y-%m-%d_%H:%M:%S) all ${EXPECTED} lanes launched; watching for sentinels..." | tee -a ${OUT}/launcher.log
while true; do
    DONE=$(find ${OUT}/ -maxdepth 2 -name .lane_done 2>/dev/null | wc -l)
    if [ "${DONE}" -ge "${EXPECTED}" ]; then
        echo "[g2-div] $(date +%Y-%m-%d_%H:%M:%S) all ${EXPECTED} lanes done" | tee -a ${OUT}/launcher.log
        break
    fi
    sleep 60
done

# ----- compile summary -----
python - <<PYEOF 2>&1 | tee -a ${OUT}/launcher.log
import json
from pathlib import Path
out = Path("${OUT}")
summary = {"output_dir": str(out), "buffer_path": "${BUF}", "lanes": {}}
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
(out / "g2_diversified_buffer_summary.json").write_text(json.dumps(summary, indent=2))
print("=== G2 DIVERSIFIED BUFFER SUMMARY ===")
print(json.dumps(summary, indent=2))
PYEOF

echo "[g2-div] $(date +%Y-%m-%d_%H:%M:%S) CHAIN DONE" | tee -a ${OUT}/launcher.log
