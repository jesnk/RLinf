#! /bin/bash
# eval_libero_pro.sh
#
# LIBERO-PRO position-perturbation (swap) evaluation wrapper.
#
# Usage:
#   bash examples/embodiment/eval_libero_pro.sh <CONFIG_NAME> [CKPT_PATH] [SEED]
#
# Examples:
#   # π0.5 SFT sanity:
#   bash examples/embodiment/eval_libero_pro.sh libero_spatial_eval_libero_pro_openpi_pi05
#
#   # πRL spatial 50ep:
#   bash examples/embodiment/eval_libero_pro.sh libero_spatial_eval_libero_pro_openpi_pi05 \
#        /home/jskang/data/jskang/sigma/logs/pirl_spatial_phase2_*/pirl_spatial_phase2/checkpoints/global_step_50
#
#   # DSRL spatial 20ep:
#   bash examples/embodiment/eval_libero_pro.sh libero_spatial_eval_libero_pro_openpi_pi05 \
#        /home/jskang/data/jskang/sigma/logs/dsrl_libero_spatial_pi05_*/dsrl_libero_spatial_pi05/checkpoints/global_step_20

set -e

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"

# LIBERO rendering. egl works on systems without osmesa GL.
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

# LIBERO-PRO mode
export LIBERO_TYPE=pro
export LIBERO_PERTURBATION=${LIBERO_PERTURBATION:-swap}     # position perturbation
export LIBEROPRO_DATA_ROOT=${LIBEROPRO_DATA_ROOT:-/home/jskang/sigma/liberopro_data}

export ROBOT_PLATFORM=${ROBOT_PLATFORM:-LIBERO}
export HYDRA_FULL_ERROR=1

if [ -z "$1" ]; then
    echo "Usage: bash $0 <CONFIG_NAME> [CKPT_PATH] [SEED]" >&2
    echo "  e.g. libero_spatial_eval_libero_pro_openpi_pi05" >&2
    exit 1
fi
CONFIG_NAME=$1
CKPT_PATH=${2:-}
SEED=${3:-42}

OVERRIDES=()
if [ -n "$CKPT_PATH" ]; then
    OVERRIDES+=("actor.model.model_path=${CKPT_PATH}")
    OVERRIDES+=("rollout.model.model_path=${CKPT_PATH}")
fi
OVERRIDES+=("env.eval.seed=${SEED}")
OVERRIDES+=("actor.seed=${SEED}")

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H%M%S')-${CONFIG_NAME}-seed${SEED}"
LOG_FILE="${LOG_DIR}/eval_libero_pro.log"
mkdir -p "${LOG_DIR}"

echo "==> LIBERO-PRO Eval"
echo "    CONFIG     = ${CONFIG_NAME}"
echo "    CKPT       = ${CKPT_PATH:-<config default>}"
echo "    SEED       = ${SEED}"
echo "    PERTURB    = ${LIBERO_PERTURBATION}"
echo "    DATA_ROOT  = ${LIBEROPRO_DATA_ROOT}"
echo "    LOG_DIR    = ${LOG_DIR}"

CMD=(python "${SRC_FILE}" --config-path "${EMBODIED_PATH}/config/" --config-name "${CONFIG_NAME}" \
     "runner.logger.log_path=${LOG_DIR}" "${OVERRIDES[@]}")

echo "${CMD[@]}" > "${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
