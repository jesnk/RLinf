#!/bin/bash
# σ-QRT W4 gate PARALLEL launcher v2 — adds periodic eval lane on GPU 6.
#
# Layout vs v1 (w4_gate_launcher_parallel.sh):
#   - σ-QRT seed 1/2/3 on GPU 0/1/2 (training, with --save_interval 2000)
#   - A1  seed 1/2/3 on GPU 3/4/5 (training, with --save_interval 2000)
#   - B0 zero-shot eval on GPU 6 (~12 min, identical to v1)
#   - After B0 .lane_done sentinel appears, periodic_eval_watcher.py launches
#     on GPU 6, watching qrt_seed1/ for the learning curve of the headline
#     run. GPU 7 stays reserved as buffer.
#
# Output: runs/w4_gate_parallel_v2_<STAMP>/{qrt,a1}_seed{1,2,3}/ + b0/ +
#         qrt_seed1/learning_curve.jsonl + periodic_eval/.watcher_done
#         w4_gate_summary.json compiled after all 7 .lane_done sentinels.
#
# v1 (w4_gate_launcher_parallel.sh) is preserved unchanged.

set -uo pipefail   # no -e — we want to continue if one lane crashes

cd ~/data/jskang/sigma-qrt/RLinf
source ~/data/jskang/sigma/.venv/bin/activate
export PYTHONPATH=$(pwd)

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=runs/w4_gate_parallel_v2_${STAMP}
LOG_DIR=logs/w4_gate_parallel_v2_${STAMP}
mkdir -p ${OUT} ${LOG_DIR} ${OUT}/periodic_eval
echo "[w4_v2] OUT=${OUT}" | tee ${OUT}/launcher.log
echo "[w4_v2] LOG_DIR=${LOG_DIR}" | tee -a ${OUT}/launcher.log

CFG=examples/embodiment/config/libero_long_qrt_openpi_pi05.yaml
BUF=data/sigma_qrt/libero_long/transitions_B2500.pkl

# Tunables (override via env on invocation if needed)
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
PERIODIC_NUM_EVAL=${PERIODIC_NUM_EVAL:-10}
PERIODIC_POLL=${PERIODIC_POLL:-60}
PERIODIC_MAX_WAIT=${PERIODIC_MAX_WAIT:-7200}

# ----- helper: one GPU lane = train then eval (sequential within lane) -----
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
            --save_interval ${SAVE_INTERVAL} \
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
                --num_eval 50 \
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
    echo "[w4_v2] launched lane g${GPU} ${LANE_LABEL} pid=$(cat ${LANE_OUT}/.lane.pid)" | tee -a ${OUT}/launcher.log
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
            --num_eval 50 \
            --seed 1 \
            --output ${LANE_OUT}/eval_sr.json \
            > ${LANE_TAG}_eval.log 2>&1
        local EV_RC=$?
        echo "[b0 g${GPU}] $(date +%Y-%m-%d_%H:%M:%S) eval rc=${EV_RC}" >> ${LANE_TAG}.log
        touch ${LANE_OUT}/.lane_done
        echo "[b0 g${GPU}] $(date +%Y-%m-%d_%H:%M:%S) DONE" >> ${LANE_TAG}.log
    ) &
    echo $! > ${LANE_OUT}/.lane.pid
    echo "[w4_v2] launched b0 g${GPU} pid=$(cat ${LANE_OUT}/.lane.pid)" | tee -a ${OUT}/launcher.log
}

# ----- helper: periodic eval watcher — kicks in after B0 frees GPU 6 -----
# Watches qrt_seed1/ (the headline lane) for ckpt_step*.pt and feeds them
# through eval_libero_sr.py one at a time. Stops when .training_done is
# present AND all ckpts have been evaluated, or when max_wait elapses.
run_periodic_eval() {
    local GPU=$1
    local WATCH_DIR=${OUT}/qrt_seed1
    local PE_DIR=${OUT}/periodic_eval
    local PE_TAG=${LOG_DIR}/periodic_eval
    mkdir -p ${PE_DIR}
    (
        # Wait for B0 to finish so GPU is free.
        until [ -f ${OUT}/b0/.lane_done ]; do sleep 30; done
        echo "[periodic_eval g${GPU}] $(date +%Y-%m-%d_%H:%M:%S) B0 done; starting watcher on ${WATCH_DIR}" >> ${PE_TAG}.log
        python examples/embodiment/periodic_eval_watcher.py \
            --watch_dir ${WATCH_DIR} \
            --config ${CFG} \
            --num_eval ${PERIODIC_NUM_EVAL} \
            --gpu ${GPU} \
            --poll_interval ${PERIODIC_POLL} \
            --max_wait ${PERIODIC_MAX_WAIT} \
            > ${PE_TAG}.log 2>&1
        local PE_RC=$?
        echo "[periodic_eval g${GPU}] $(date +%Y-%m-%d_%H:%M:%S) watcher rc=${PE_RC}" >> ${PE_TAG}.log
        touch ${PE_DIR}/.watcher_done
    ) &
    echo $! > ${PE_DIR}/.watcher.pid
    echo "[w4_v2] launched periodic_eval g${GPU} pid=$(cat ${PE_DIR}/.watcher.pid) (will wait for B0)" | tee -a ${OUT}/launcher.log
}

# ----- launch all 7 training+B0 lanes (GPUs 0-6; GPU 7 reserved as buffer) -----
run_lane 0 qrt               1 qrt_seed1
run_lane 1 qrt               2 qrt_seed2
run_lane 2 qrt               3 qrt_seed3
run_lane 3 a1_frozen_encoder 1 a1_seed1
run_lane 4 a1_frozen_encoder 2 a1_seed2
run_lane 5 a1_frozen_encoder 3 a1_seed3
run_b0   6
run_periodic_eval 6

# ----- wait for sentinels (7 train/eval lanes; watcher is separate) -----
EXPECTED=7
echo "[w4_v2] $(date +%Y-%m-%d_%H:%M:%S) all ${EXPECTED} lanes launched + periodic_eval queued; watching for sentinels..." | tee -a ${OUT}/launcher.log
while true; do
    DONE=$(find ${OUT}/ -maxdepth 2 -name .lane_done 2>/dev/null | wc -l)
    if [ "${DONE}" -ge "${EXPECTED}" ]; then
        echo "[w4_v2] $(date +%Y-%m-%d_%H:%M:%S) all ${EXPECTED} lanes done" | tee -a ${OUT}/launcher.log
        break
    fi
    sleep 60
done

# Best-effort wait for periodic_eval watcher to drain (cap at 30 min after
# trainings finish — if it's still running after that, summary proceeds
# without the learning curve being finalized).
echo "[w4_v2] $(date +%Y-%m-%d_%H:%M:%S) waiting up to 30 min for periodic_eval to drain..." | tee -a ${OUT}/launcher.log
WAIT_LIMIT=1800
WAITED=0
while [ ! -f ${OUT}/periodic_eval/.watcher_done ] && [ ${WAITED} -lt ${WAIT_LIMIT} ]; do
    sleep 60
    WAITED=$((WAITED + 60))
done
if [ -f ${OUT}/periodic_eval/.watcher_done ]; then
    echo "[w4_v2] $(date +%Y-%m-%d_%H:%M:%S) periodic_eval watcher done" | tee -a ${OUT}/launcher.log
else
    echo "[w4_v2] $(date +%Y-%m-%d_%H:%M:%S) periodic_eval watcher still running after ${WAIT_LIMIT}s; proceeding to summary anyway" | tee -a ${OUT}/launcher.log
fi

# ----- compile summary (includes learning_curve count if present) -----
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
# Learning curve from periodic_eval_watcher.py.
lc_path = out / "qrt_seed1" / "learning_curve.jsonl"
if lc_path.exists():
    rows = [json.loads(line) for line in lc_path.read_text().splitlines() if line.strip()]
    summary["learning_curve"] = {
        "ckpt_count": len(rows),
        "path": str(lc_path),
        "steps": [r.get("step") for r in rows],
        "srs": [r.get("sr") for r in rows],
    }
else:
    summary["learning_curve"] = {"error": f"no learning_curve.jsonl at {lc_path}"}
(out / "w4_gate_summary.json").write_text(json.dumps(summary, indent=2))
print("=== W4 GATE PARALLEL V2 SUMMARY ===")
print(json.dumps(summary, indent=2))
PYEOF

echo "[w4_v2] $(date +%Y-%m-%d_%H:%M:%S) CHAIN DONE" | tee -a ${OUT}/launcher.log
