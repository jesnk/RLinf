#!/bin/bash
# σ-QRT-BC sanity training: BC-only (β_bc=100) on 2 seeds in parallel
# to test if the eval pipeline + actor module works when actor is forced to
# be a near-clone of ref_action. If this also gets 0% SR → eval pipeline bug.
# If ≈40% (≈ B0) → eval OK, original A1 actor's Q-gradient broke things.
set -uo pipefail
cd ~/data/jskang/sigma-qrt/RLinf
source ~/data/jskang/sigma/.venv/bin/activate
export PYTHONPATH=$(pwd)

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=runs/sigma_qrt_bc_sanity_${STAMP}
mkdir -p ${OUT}

CFG=examples/embodiment/config/libero_long_qrt_openpi_pi05.yaml
BUF=data/sigma_qrt/libero_long/transitions_B2500.pkl

run_bc_lane() {
    local GPU=$1; local SEED=$2
    local LANE=${OUT}/qrt_bc_seed${SEED}
    mkdir -p ${LANE}
    (
        export CUDA_VISIBLE_DEVICES=${GPU}
        export MUJOCO_EGL_DEVICE_ID=${GPU}
        echo "[lane g${GPU} seed${SEED}] $(date +%Y-%m-%d_%H:%M:%S) train start"
        python examples/embodiment/run_qrt_offline.py \
            --config ${CFG} \
            --variant qrt \
            --output_dir ${LANE} \
            --override "data.offline_buffer_path=${BUF}" \
            --override "data.capacity=8000" \
            --override "training.batch_size=128" \
            --override "training.warmup_steps=500" \
            --override "training.max_train_steps=3000" \
            --override "training.beta_bc=100" \
            --override "seed=${SEED}" \
            --bf16 \
            --no_wandb \
            > ${LANE}/train.log 2>&1
        local TR_RC=$?
        echo "[lane g${GPU} seed${SEED}] $(date +%Y-%m-%d_%H:%M:%S) train rc=${TR_RC}"
        if [ -f ${LANE}/ckpt.pt ]; then
            echo "[lane g${GPU} seed${SEED}] $(date +%Y-%m-%d_%H:%M:%S) eval start"
            python examples/embodiment/eval_libero_sr.py \
                --config ${CFG} \
                --ckpt ${LANE}/ckpt.pt \
                --num_eval 25 \
                --seed 1 \
                --output ${LANE}/eval_sr.json \
                > ${LANE}/eval.log 2>&1
            echo "[lane g${GPU} seed${SEED}] $(date +%Y-%m-%d_%H:%M:%S) eval rc=$?"
        else
            echo "[lane g${GPU} seed${SEED}] no ckpt.pt produced — skipping eval"
        fi
        touch ${LANE}/.lane_done
    ) &
    echo $! > ${LANE}/.lane.pid
}

run_bc_lane 4 1
run_bc_lane 5 2

# Wait for both lanes
while [ $(find ${OUT} -name .lane_done | wc -l) -lt 2 ]; do
    sleep 60
done

echo "=== sigma-QRT-BC SANITY RESULTS ===" | tee ${OUT}/summary.log
for s in 1 2; do
    SR=$(cat ${OUT}/qrt_bc_seed${s}/eval_sr.json 2>/dev/null | python -c "import json,sys; print(json.load(sys.stdin).get('sr'))" 2>/dev/null)
    echo "qrt_bc_seed${s}: sr=${SR}" | tee -a ${OUT}/summary.log
done

echo "=== DONE === ${OUT}"
