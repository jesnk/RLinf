#!/bin/bash
# σ-QRT IQL chain v6 launcher: A1 frozen-encoder vs σ-QRT joint-encoder under IQL.
#
# Goal
# ----
# Direct ablation of σ-QRT's main novelty (Q-aware joint encoder training)
# under IQL. Tests whether joint encoder HURTS vs frozen encoder when the
# Stage-2 objective is IQL (no Q-extrapolation by construction).
#
# Layout (6 training lanes on GPU 0-5; GPU 6 + 7 reserved for the
# periodic-eval coordinator pool — DO NOT TOUCH GPU 6/7 from this launcher):
#   - GPU 0: qrt_seed1 (variant=qrt,             tau=0.8, beta=3, seed=1)
#   - GPU 1: qrt_seed2 (variant=qrt,             tau=0.8, beta=3, seed=2)
#   - GPU 2: qrt_seed3 (variant=qrt,             tau=0.8, beta=3, seed=3)
#   - GPU 3: a1_seed1  (variant=a1_frozen_encoder,tau=0.8, beta=3, seed=1)
#   - GPU 4: a1_seed2  (variant=a1_frozen_encoder,tau=0.8, beta=3, seed=2)
#   - GPU 5: a1_seed3  (variant=a1_frozen_encoder,tau=0.8, beta=3, seed=3)
#
# Fixed hyperparams from chain v5 best: tau=0.8, iql_beta=3.
#
# Each lane:
#   1. train 10000 steps with --use_iql on libero_long_qrt_iql_openpi_pi05.yaml,
#      tau=0.8, beta=3, seed per-lane, --save_interval 1000 (finer granularity
#      than v5's 2000 — user request to better catch the v5 step-6000 peak).
#   2. final async vectorized eval (25 ep, 5 envs, seed 1) on the final
#      ckpt.pt.
#   3. touch ${LANE_OUT}/.lane_done — coordinator drain signal.
#
# After all 6 sentinels, an iql_v6_summary.json is compiled.

set -uo pipefail   # no -e — continue if one lane crashes

cd ~/data/jskang/sigma-qrt/RLinf
source ~/data/jskang/sigma/.venv/bin/activate
export PYTHONPATH=$(pwd)

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=runs/iql_v6_a1_vs_joint_${STAMP}
LOG_DIR=logs/iql_v6_a1_vs_joint_${STAMP}
mkdir -p ${OUT} ${LOG_DIR}
echo "[iql-v6] OUT=${OUT}" | tee ${OUT}/launcher.log
echo "[iql-v6] LOG_DIR=${LOG_DIR}" | tee -a ${OUT}/launcher.log

CFG=examples/embodiment/config/libero_long_qrt_iql_openpi_pi05.yaml
BUF=data/sigma_qrt/libero_long/transitions_B2500.pkl
TAU=0.8
BETA=3
SAVE_INTERVAL=1000

# ----- helper: IQL lane = train then final eval (sequential within lane) -----
run_lane() {
    local GPU=$1; local VARIANT=$2; local SEED=$3; local TAG=$4
    local LANE_NAME=${TAG}_seed${SEED}
    local LANE_OUT=${OUT}/${LANE_NAME}
    local LANE_TAG=${LOG_DIR}/${LANE_NAME}
    mkdir -p ${LANE_OUT}
    (
        export CUDA_VISIBLE_DEVICES=${GPU}
        export MUJOCO_EGL_DEVICE_ID=${GPU}
        echo "[lane g${GPU} ${LANE_NAME}] $(date +%Y-%m-%d_%H:%M:%S) train start variant=${VARIANT} tau=${TAU} beta=${BETA} seed=${SEED}" >> ${LANE_TAG}.log
        python examples/embodiment/run_qrt_offline.py \
            --config ${CFG} \
            --variant ${VARIANT} \
            --use_iql \
            --output_dir ${LANE_OUT} \
            --override "data.offline_buffer_path=${BUF}" \
            --override "data.capacity=8000" \
            --override "training.batch_size=128" \
            --override "training.warmup_steps=2000" \
            --override "training.max_train_steps=10000" \
            --override "training.iql_tau=${TAU}" \
            --override "training.iql_beta=${BETA}" \
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
    echo "[iql-v6] launched lane g${GPU} ${LANE_NAME} pid=$(cat ${LANE_OUT}/.lane.pid)" | tee -a ${OUT}/launcher.log
}

# ----- launch 6 lanes: 3 qrt + 3 a1_frozen_encoder, all tau=0.8 beta=3 -----
run_lane 0 qrt               1 qrt
run_lane 1 qrt               2 qrt
run_lane 2 qrt               3 qrt
run_lane 3 a1_frozen_encoder 1 a1
run_lane 4 a1_frozen_encoder 2 a1
run_lane 5 a1_frozen_encoder 3 a1

# ----- wait for all 6 sentinels -----
EXPECTED=6
echo "[iql-v6] $(date +%Y-%m-%d_%H:%M:%S) all ${EXPECTED} lanes launched; watching for sentinels..." | tee -a ${OUT}/launcher.log
while true; do
    DONE=$(find ${OUT}/ -maxdepth 2 -name .lane_done 2>/dev/null | wc -l)
    if [ "${DONE}" -ge "${EXPECTED}" ]; then
        echo "[iql-v6] $(date +%Y-%m-%d_%H:%M:%S) all ${EXPECTED} lanes done" | tee -a ${OUT}/launcher.log
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
(out / "iql_v6_summary.json").write_text(json.dumps(summary, indent=2))
print("=== IQL v6 A1 vs JOINT SUMMARY ===")
print(json.dumps(summary, indent=2))
PYEOF

echo "[iql-v6] $(date +%Y-%m-%d_%H:%M:%S) CHAIN DONE" | tee -a ${OUT}/launcher.log
