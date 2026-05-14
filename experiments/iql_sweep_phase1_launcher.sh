#!/bin/bash
# σ-QRT IQL sweep Phase 1 launcher (chain v5)
#
# Layout (7 training lanes on GPU 0-6; GPU 7 reserved for the periodic-eval
# coordinator — DO NOT TOUCH GPU 7 from this launcher):
#   - GPU 0: iql_tau0p7_beta3_seed1
#   - GPU 1: iql_tau0p7_beta3_seed2
#   - GPU 2: iql_tau0p7_beta10_seed1
#   - GPU 3: iql_tau0p8_beta3_seed1
#   - GPU 4: iql_tau0p8_beta10_seed1
#   - GPU 5: iql_tau0p9_beta3_seed1
#   - GPU 6: iql_tau0p9_beta10_seed1
#
# Each lane:
#   1. train 10000 steps with --use_iql (IQL yaml: libero_long_qrt_iql_openpi_pi05.yaml)
#      with training.iql_tau / training.iql_beta / seed per-lane overrides.
#      Intermediate ckpt_step{2000,4000,6000,8000}.pt land in the lane dir;
#      the periodic-eval coordinator on GPU 7 watches them.
#   2. final async vectorized eval (25 ep, 5 envs, seed 1) on the final
#      ckpt.pt — written by the launcher, not by the coordinator (the
#      coordinator intentionally skips ckpt.pt to avoid double-eval).
#   3. touch ${LANE_OUT}/.lane_done — coordinator drain signal.
#
# After all 7 sentinels, an iql_sweep_summary.json is compiled.
#
# B0 is NOT re-run (already SR=0.42 from prior measurement).

set -uo pipefail   # no -e — continue if one lane crashes

cd ~/data/jskang/sigma-qrt/RLinf
source ~/data/jskang/sigma/.venv/bin/activate
export PYTHONPATH=$(pwd)

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=runs/iql_sweep_phase1_${STAMP}
LOG_DIR=logs/iql_sweep_phase1_${STAMP}
mkdir -p ${OUT} ${LOG_DIR}
echo "[iql-sweep p1] OUT=${OUT}" | tee ${OUT}/launcher.log
echo "[iql-sweep p1] LOG_DIR=${LOG_DIR}" | tee -a ${OUT}/launcher.log

CFG=examples/embodiment/config/libero_long_qrt_iql_openpi_pi05.yaml
BUF=data/sigma_qrt/libero_long/transitions_B2500.pkl

# ----- helper: σ-QRT IQL lane = train then final eval (sequential within lane) -----
run_iql_lane() {
    local GPU=$1; local TAU=$2; local BETA=$3; local SEED=$4
    local TAU_TAG=$(echo ${TAU} | sed 's/\./p/g')
    local BETA_TAG=$(echo ${BETA} | sed 's/\./p/g')
    local LANE_NAME=iql_tau${TAU_TAG}_beta${BETA_TAG}_seed${SEED}
    local LANE_OUT=${OUT}/${LANE_NAME}
    local LANE_TAG=${LOG_DIR}/${LANE_NAME}
    mkdir -p ${LANE_OUT}
    (
        export CUDA_VISIBLE_DEVICES=${GPU}
        export MUJOCO_EGL_DEVICE_ID=${GPU}
        echo "[lane g${GPU} ${LANE_NAME}] $(date +%Y-%m-%d_%H:%M:%S) train start tau=${TAU} beta=${BETA} seed=${SEED}" >> ${LANE_TAG}.log
        python examples/embodiment/run_qrt_offline.py \
            --config ${CFG} \
            --variant qrt \
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
    echo "[iql-sweep p1] launched lane g${GPU} ${LANE_NAME} pid=$(cat ${LANE_OUT}/.lane.pid)" | tee -a ${OUT}/launcher.log
}

# ----- launch 7 σ-QRT IQL lanes (GPU 7 reserved for periodic-eval coordinator) -----
run_iql_lane 0 0.7 3  1
run_iql_lane 1 0.7 3  2
run_iql_lane 2 0.7 10 1
run_iql_lane 3 0.8 3  1
run_iql_lane 4 0.8 10 1
run_iql_lane 5 0.9 3  1
run_iql_lane 6 0.9 10 1

# ----- wait for all 7 sentinels -----
EXPECTED=7
echo "[iql-sweep p1] $(date +%Y-%m-%d_%H:%M:%S) all ${EXPECTED} lanes launched; watching for sentinels..." | tee -a ${OUT}/launcher.log
while true; do
    DONE=$(find ${OUT}/ -maxdepth 2 -name .lane_done 2>/dev/null | wc -l)
    if [ "${DONE}" -ge "${EXPECTED}" ]; then
        echo "[iql-sweep p1] $(date +%Y-%m-%d_%H:%M:%S) all ${EXPECTED} lanes done" | tee -a ${OUT}/launcher.log
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
(out / "iql_sweep_summary.json").write_text(json.dumps(summary, indent=2))
print("=== IQL SWEEP PHASE 1 SUMMARY ===")
print(json.dumps(summary, indent=2))
PYEOF

echo "[iql-sweep p1] $(date +%Y-%m-%d_%H:%M:%S) CHAIN DONE" | tee -a ${OUT}/launcher.log
