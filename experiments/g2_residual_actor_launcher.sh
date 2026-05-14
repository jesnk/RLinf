#!/bin/bash
# σ-QRT G2 residual-actor launcher (3 residual seeds + 3 non-residual baseline seeds).
#
# Goal
# ----
# Strengthen the σ-QRT G2 baseline via residual actor parametrization
# (PI RLT / TD3+BC-style). Actor learns Δ_θ on top of π0.5's reference
# action; SR floor = B0 (≈0.42) is structurally guaranteed by Δ→0.
#
#   a_pred(s) = ref_action(s) + Δ_θ(z_rl, s_p, ref_action)
#   L_actor   = E[exp(β·A)·‖Δ‖²] + α·-Q1(s, a_pred) + λ·‖Δ‖²
#
# Why: G1 verdict confirmed σ-QRT joint encoder representation is valid
# (R²=0.94 vs frozen 0.63), but peak SR (0.19-0.24) trails B0 (0.42).
# Root cause hypothesis: small actor MLP cannot match π0.5's BC fidelity
# from scratch. Residual parametrization sidesteps that — Δ=0 ⇒ B0.
#
# Layout (6 training lanes on GPU 0-5; GPU 6 + 7 reserved for the
# periodic-eval coordinator pool — DO NOT TOUCH GPU 6/7 from this launcher):
#   - GPU 0: g2res_seed1   (variant=a1_frozen_encoder, use_residual_actor=true,  seed=1)
#   - GPU 1: g2res_seed2   (variant=a1_frozen_encoder, use_residual_actor=true,  seed=2)
#   - GPU 2: g2res_seed3   (variant=a1_frozen_encoder, use_residual_actor=true,  seed=3)
#   - GPU 3: g2base_seed1  (variant=a1_frozen_encoder, use_residual_actor=false, seed=1)
#   - GPU 4: g2base_seed2  (variant=a1_frozen_encoder, use_residual_actor=false, seed=2)
#   - GPU 5: g2base_seed3  (variant=a1_frozen_encoder, use_residual_actor=false, seed=3)
#
# All lanes use a1_frozen_encoder (encoder frozen after Stage-1 warmup —
# faster + simpler than joint qrt; v6_v2 already showed a1 ≈ qrt under IQL).
# Hyperparams from v6_v2 best: tau=0.8, iql_beta=3.
#
# Each lane:
#   1. train 10000 steps with --use_iql, tau=0.8, beta=3, seed per-lane,
#      warmup_steps=10000 (clean Stage-1), --save_interval 1000.
#   2. final async vectorized eval (25 ep, 5 envs, seed 1) on the final
#      ckpt.pt.
#   3. touch ${LANE_OUT}/.lane_done — coordinator drain signal.
#
# After all 6 sentinels, a g2_residual_summary.json is compiled.

set -uo pipefail   # no -e — continue if one lane crashes

cd ~/data/jskang/sigma-qrt/RLinf
source ~/data/jskang/sigma/.venv/bin/activate
export PYTHONPATH=$(pwd)

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=runs/g2_residual_actor_${STAMP}
LOG_DIR=logs/g2_residual_actor_${STAMP}
mkdir -p ${OUT} ${LOG_DIR}
echo "[g2-res] OUT=${OUT}" | tee ${OUT}/launcher.log
echo "[g2-res] LOG_DIR=${LOG_DIR}" | tee -a ${OUT}/launcher.log

CFG=examples/embodiment/config/libero_long_qrt_iql_openpi_pi05.yaml
BUF=data/sigma_qrt/libero_long/transitions_B2500.pkl
TAU=0.8
BETA=3
SAVE_INTERVAL=1000
RES_ALPHA=0.1
RES_REG=0.0

# ----- helper: lane = train then final eval (sequential within lane) -----
run_lane() {
    local GPU=$1; local SEED=$2; local TAG=$3; local USE_RES=$4
    local LANE_NAME=${TAG}_seed${SEED}
    local LANE_OUT=${OUT}/${LANE_NAME}
    local LANE_TAG=${LOG_DIR}/${LANE_NAME}
    mkdir -p ${LANE_OUT}
    (
        export CUDA_VISIBLE_DEVICES=${GPU}
        export MUJOCO_EGL_DEVICE_ID=${GPU}
        echo "[lane g${GPU} ${LANE_NAME}] $(date +%Y-%m-%d_%H:%M:%S) train start variant=a1_frozen_encoder tau=${TAU} beta=${BETA} seed=${SEED} use_residual=${USE_RES} alpha=${RES_ALPHA}" >> ${LANE_TAG}.log
        python examples/embodiment/run_qrt_offline.py \
            --config ${CFG} \
            --variant a1_frozen_encoder \
            --use_iql \
            --output_dir ${LANE_OUT} \
            --override "data.offline_buffer_path=${BUF}" \
            --override "data.capacity=8000" \
            --override "training.batch_size=128" \
            --override "training.warmup_steps=10000" \
            --override "training.max_train_steps=10000" \
            --override "training.iql_tau=${TAU}" \
            --override "training.iql_beta=${BETA}" \
            --override "training.use_residual_actor=${USE_RES}" \
            --override "training.residual_alpha=${RES_ALPHA}" \
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
    echo "[g2-res] launched lane g${GPU} ${LANE_NAME} pid=$(cat ${LANE_OUT}/.lane.pid)" | tee -a ${OUT}/launcher.log
}

# ----- launch 6 lanes: 3 residual + 3 baseline -----
run_lane 0 1 g2res  true
run_lane 1 2 g2res  true
run_lane 2 3 g2res  true
run_lane 3 1 g2base false
run_lane 4 2 g2base false
run_lane 5 3 g2base false

# ----- wait for all 6 sentinels -----
EXPECTED=6
echo "[g2-res] $(date +%Y-%m-%d_%H:%M:%S) all ${EXPECTED} lanes launched; watching for sentinels..." | tee -a ${OUT}/launcher.log
while true; do
    DONE=$(find ${OUT}/ -maxdepth 2 -name .lane_done 2>/dev/null | wc -l)
    if [ "${DONE}" -ge "${EXPECTED}" ]; then
        echo "[g2-res] $(date +%Y-%m-%d_%H:%M:%S) all ${EXPECTED} lanes done" | tee -a ${OUT}/launcher.log
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
(out / "g2_residual_summary.json").write_text(json.dumps(summary, indent=2))
print("=== G2 RESIDUAL ACTOR SUMMARY ===")
print(json.dumps(summary, indent=2))
PYEOF

echo "[g2-res] $(date +%Y-%m-%d_%H:%M:%S) CHAIN DONE" | tee -a ${OUT}/launcher.log
