#!/bin/bash
# σ-QRT v7 G1.3 bypass test launcher: a1_raw_features × 3 seeds under IQL.
#
# Goal
# ----
# Tests whether the RLT encoder is even useful vs raw mean-pooled VLA
# features. Trains 3 seeds of variant=a1_raw_features (encoder fully
# bypassed) with the same IQL hyperparams as v6_v2 (tau=0.8, beta=3,
# max_train_steps=10000, save_interval=1000, batch_size=128).
#
# Compare final SR vs v6_v2's a1_frozen_encoder (encoder USED) to test
# Goal 1 of v7: is the encoder doing useful representation work?
#
# Layout (3 training lanes on GPU 1-3; GPU 4-5 reserved for the
# periodic-eval coordinator pool — launched separately):
#   - GPU 1: raw_seed1 (variant=a1_raw_features, tau=0.8, beta=3, seed=1)
#   - GPU 2: raw_seed2 (variant=a1_raw_features, tau=0.8, beta=3, seed=2)
#   - GPU 3: raw_seed3 (variant=a1_raw_features, tau=0.8, beta=3, seed=3)
#
# warmup_steps=0 because a1_raw_features SKIPS Stage 1 per worker design
# (encoder bypassed → no representation warmup needed).

set -uo pipefail

cd ~/data/jskang/sigma-qrt/RLinf
source ~/data/jskang/sigma/.venv/bin/activate
export PYTHONPATH=$(pwd)

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=runs/g1_bypass_test_${STAMP}
LOG_DIR=logs/g1_bypass_test_${STAMP}
mkdir -p ${OUT} ${LOG_DIR}
echo "[g1-bypass] OUT=${OUT}" | tee ${OUT}/launcher.log
echo "[g1-bypass] LOG_DIR=${LOG_DIR}" | tee -a ${OUT}/launcher.log

CFG=examples/embodiment/config/libero_long_qrt_iql_openpi_pi05.yaml
BUF=data/sigma_qrt/libero_long/transitions_B2500.pkl
TAU=0.8
BETA=3
SAVE_INTERVAL=1000

# ----- helper: IQL lane = train then final eval (sequential within lane) -----
run_lane() {
    local GPU=$1; local SEED=$2
    local LANE_NAME=raw_seed${SEED}
    local LANE_OUT=${OUT}/${LANE_NAME}
    local LANE_TAG=${LOG_DIR}/${LANE_NAME}
    mkdir -p ${LANE_OUT}
    (
        export CUDA_VISIBLE_DEVICES=${GPU}
        export MUJOCO_EGL_DEVICE_ID=${GPU}
        echo "[lane g${GPU} ${LANE_NAME}] $(date +%Y-%m-%d_%H:%M:%S) train start variant=a1_raw_features tau=${TAU} beta=${BETA} seed=${SEED}" >> ${LANE_TAG}.log
        python examples/embodiment/run_qrt_offline.py             --config ${CFG}             --variant a1_raw_features             --use_iql             --output_dir ${LANE_OUT}             --override "data.offline_buffer_path=${BUF}"             --override "data.capacity=8000"             --override "training.batch_size=128"             --override "training.warmup_steps=0"             --override "training.max_train_steps=10000"             --override "training.iql_tau=${TAU}"             --override "training.iql_beta=${BETA}"             --override "seed=${SEED}"             --save_interval ${SAVE_INTERVAL}             --bf16             --no_wandb             > ${LANE_TAG}_train.log 2>&1
        local TR_RC=$?
        echo "[lane g${GPU} ${LANE_NAME}] $(date +%Y-%m-%d_%H:%M:%S) train rc=${TR_RC}" >> ${LANE_TAG}.log
        if [ -f ${LANE_OUT}/ckpt.pt ]; then
            echo "[lane g${GPU} ${LANE_NAME}] $(date +%Y-%m-%d_%H:%M:%S) final eval start" >> ${LANE_TAG}.log
            python examples/embodiment/eval_libero_sr.py                 --config ${CFG}                 --ckpt ${LANE_OUT}/ckpt.pt                 --num_eval 25                 --num_envs 5                 --seed 1                 --output ${LANE_OUT}/eval_sr.json                 > ${LANE_TAG}_eval.log 2>&1
            local EV_RC=$?
            echo "[lane g${GPU} ${LANE_NAME}] $(date +%Y-%m-%d_%H:%M:%S) final eval rc=${EV_RC}" >> ${LANE_TAG}.log
        else
            echo "[lane g${GPU} ${LANE_NAME}] $(date +%Y-%m-%d_%H:%M:%S) NO ckpt.pt — skipping eval" >> ${LANE_TAG}.log
        fi
        touch ${LANE_OUT}/.lane_done
        echo "[lane g${GPU} ${LANE_NAME}] $(date +%Y-%m-%d_%H:%M:%S) DONE" >> ${LANE_TAG}.log
    ) &
    echo $! > ${LANE_OUT}/.lane.pid
    echo "[g1-bypass] launched lane g${GPU} ${LANE_NAME} pid=$(cat ${LANE_OUT}/.lane.pid)" | tee -a ${OUT}/launcher.log
}

# ----- launch 3 lanes: a1_raw_features × {seed 1,2,3} on GPU 1,2,3 -----
run_lane 1 1
run_lane 2 2
run_lane 3 3

# ----- wait for all 3 sentinels -----
EXPECTED=3
echo "[g1-bypass] $(date +%Y-%m-%d_%H:%M:%S) all ${EXPECTED} lanes launched; watching for sentinels..." | tee -a ${OUT}/launcher.log
while true; do
    DONE=$(find ${OUT}/ -maxdepth 2 -name .lane_done 2>/dev/null | wc -l)
    if [ "${DONE}" -ge "${EXPECTED}" ]; then
        echo "[g1-bypass] $(date +%Y-%m-%d_%H:%M:%S) all ${EXPECTED} lanes done" | tee -a ${OUT}/launcher.log
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
(out / "g1_bypass_summary.json").write_text(json.dumps(summary, indent=2))
print("=== G1.3 BYPASS TEST SUMMARY ===")
print(json.dumps(summary, indent=2))
PYEOF

echo "[g1-bypass] $(date +%Y-%m-%d_%H:%M:%S) CHAIN DONE" | tee -a ${OUT}/launcher.log
