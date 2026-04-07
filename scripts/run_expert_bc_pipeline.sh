#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PRED_MODE="${PRED_MODE:-tp_net}"
V_PREY_TEST="${V_PREY_TEST:-1.5}"
V_DRONE_TEST="${V_DRONE_TEST:-1.5}"
EPISODE_LENGTH="${EPISODE_LENGTH:-1000}"
EVAL_GPU="${EVAL_GPU:-0}"

GENERIC_BATCH_ENVS="${GENERIC_BATCH_ENVS:-2048}"
NUM_WAVES="${NUM_WAVES:-100}"
MIN_SUCCESS_STEPS="${MIN_SUCCESS_STEPS:-1}"
DATASET_DTYPE="${DATASET_DTYPE:-float16}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/expert_datasets}"

BC_EPOCHS="${BC_EPOCHS:-10}"
BC_BATCH_SIZE="${BC_BATCH_SIZE:-4096}"
BC_LR="${BC_LR:-0.0005}"
BC_DEVICE="${BC_DEVICE:-cuda:0}"
BC_SAVE_TAG="${BC_SAVE_TAG:-expert_bc_2048x100}"
BC_ENTROPY_BONUS_COEF="${BC_ENTROPY_BONUS_COEF:-0.0}"
BC_ACTION_MSE_COEF="${BC_ACTION_MSE_COEF:-1.0}"
BC_LOG_PROB_COEF="${BC_LOG_PROB_COEF:-0.0}"
BC_AUX_VEL_CMD_COEF="${BC_AUX_VEL_CMD_COEF:-0.0}"
BC_AUX_WAYPOINT_COEF="${BC_AUX_WAYPOINT_COEF:-0.0}"
BC_AUX_ASSIGNMENT_COEF="${BC_AUX_ASSIGNMENT_COEF:-0.0}"
BC_AUX_TRAP_COEF="${BC_AUX_TRAP_COEF:-0.0}"
BC_EQUALIZE_EPISODE_WEIGHT="${BC_EQUALIZE_EPISODE_WEIGHT:-false}"
BC_FRONT_WEIGHT_ALPHA="${BC_FRONT_WEIGHT_ALPHA:-0.0}"
BC_MIN_EPISODE_LEN="${BC_MIN_EPISODE_LEN:--1}"
BC_MAX_EPISODE_LEN="${BC_MAX_EPISODE_LEN:--1}"
BC_KEEP_PREFIX_STEPS="${BC_KEEP_PREFIX_STEPS:--1}"
BC_KEEP_SUFFIX_STEPS="${BC_KEEP_SUFFIX_STEPS:-0}"

mkdir -p "${DATASET_ROOT}"

echo "========================================================"
echo "  Expert BC Pipeline"
echo "  collect batch_envs = ${GENERIC_BATCH_ENVS}"
echo "  collect num_waves  = ${NUM_WAVES}"
echo "  v_drone / v_prey   = ${V_DRONE_TEST} / ${V_PREY_TEST}"
echo "  episode_length     = ${EPISODE_LENGTH}"
echo "  dataset_root       = ${DATASET_ROOT}"
echo "  bc epochs          = ${BC_EPOCHS}"
echo "  bc batch_size      = ${BC_BATCH_SIZE}"
echo "  bc device          = ${BC_DEVICE}"
echo "  bc mse / logp coef = ${BC_ACTION_MSE_COEF} / ${BC_LOG_PROB_COEF}"
echo "  bc aux vel/wp/as/tr= ${BC_AUX_VEL_CMD_COEF} / ${BC_AUX_WAYPOINT_COEF} / ${BC_AUX_ASSIGNMENT_COEF} / ${BC_AUX_TRAP_COEF}"
echo "  bc equal/front wt  = ${BC_EQUALIZE_EPISODE_WEIGHT} / ${BC_FRONT_WEIGHT_ALPHA}"
echo "  bc ep/prefix/suffix= ${BC_MIN_EPISODE_LEN}:${BC_MAX_EPISODE_LEN} / ${BC_KEEP_PREFIX_STEPS} / ${BC_KEEP_SUFFIX_STEPS}"
echo "========================================================"

before_list="$(mktemp)"
after_list="$(mktemp)"
trap 'rm -f "${before_list}" "${after_list}"' EXIT

find "${DATASET_ROOT}" -maxdepth 1 -mindepth 1 -type d | sort > "${before_list}"

COLLECT_SUCCESS_DATASET=true \
DATASET_DIR="${DATASET_ROOT}" \
MIN_SUCCESS_STEPS="${MIN_SUCCESS_STEPS}" \
DATASET_DTYPE="${DATASET_DTYPE}" \
GENERIC_BATCH_ENVS="${GENERIC_BATCH_ENVS}" \
NUM_WAVES="${NUM_WAVES}" \
EVAL_GPU="${EVAL_GPU}" \
V_DRONE_TEST="${V_DRONE_TEST}" \
EPISODE_LENGTH="${EPISODE_LENGTH}" \
bash "${PROJECT_ROOT}/scripts/expert_isaac_eval.sh" "${PRED_MODE}" "${V_PREY_TEST}" 0 0

find "${DATASET_ROOT}" -maxdepth 1 -mindepth 1 -type d | sort > "${after_list}"

dataset_dir="$(comm -13 "${before_list}" "${after_list}" | tail -n1)"
if [[ -z "${dataset_dir}" ]]; then
    dataset_dir="$(find "${DATASET_ROOT}" -maxdepth 1 -mindepth 1 -type d | sort | tail -n1)"
fi
if [[ -z "${dataset_dir}" || ! -d "${dataset_dir}" ]]; then
    echo "[ERROR] Could not determine collected dataset directory under ${DATASET_ROOT}" >&2
    exit 1
fi

echo ""
echo "Collected dataset: ${dataset_dir}"
echo ""

EPOCHS="${BC_EPOCHS}" \
BATCH_SIZE="${BC_BATCH_SIZE}" \
SAVE_TAG="${BC_SAVE_TAG}" \
BC_LR="${BC_LR}" \
ENTROPY_BONUS_COEF="${BC_ENTROPY_BONUS_COEF}" \
ACTION_MSE_COEF="${BC_ACTION_MSE_COEF}" \
LOG_PROB_COEF="${BC_LOG_PROB_COEF}" \
AUX_VEL_CMD_COEF="${BC_AUX_VEL_CMD_COEF}" \
AUX_WAYPOINT_COEF="${BC_AUX_WAYPOINT_COEF}" \
AUX_ASSIGNMENT_COEF="${BC_AUX_ASSIGNMENT_COEF}" \
AUX_TRAP_COEF="${BC_AUX_TRAP_COEF}" \
EQUALIZE_EPISODE_WEIGHT="${BC_EQUALIZE_EPISODE_WEIGHT}" \
FRONT_WEIGHT_ALPHA="${BC_FRONT_WEIGHT_ALPHA}" \
MIN_EPISODE_LEN="${BC_MIN_EPISODE_LEN}" \
MAX_EPISODE_LEN="${BC_MAX_EPISODE_LEN}" \
KEEP_PREFIX_STEPS="${BC_KEEP_PREFIX_STEPS}" \
KEEP_SUFFIX_STEPS="${BC_KEEP_SUFFIX_STEPS}" \
DEVICE="${BC_DEVICE}" \
bash "${PROJECT_ROOT}/scripts/train_expert_bc.sh" "${dataset_dir}"
