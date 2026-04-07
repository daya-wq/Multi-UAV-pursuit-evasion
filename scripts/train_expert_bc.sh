#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_EXE="${CONDA_EXE:-/home/uavlab/miniconda3/bin/conda}"

DATASET_DIR="${1:-}"
if [[ -z "${DATASET_DIR}" ]]; then
    echo "Usage: bash scripts/train_expert_bc.sh <dataset_dir>"
    exit 1
fi

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
SAVE_TAG="${SAVE_TAG:-expert_bc}"
BC_LR="${BC_LR:-0.0005}"
ENTROPY_BONUS_COEF="${ENTROPY_BONUS_COEF:-0.0}"
ACTION_MSE_COEF="${ACTION_MSE_COEF:-1.0}"
LOG_PROB_COEF="${LOG_PROB_COEF:-0.0}"
AUX_VEL_CMD_COEF="${AUX_VEL_CMD_COEF:-0.0}"
AUX_WAYPOINT_COEF="${AUX_WAYPOINT_COEF:-0.0}"
AUX_ASSIGNMENT_COEF="${AUX_ASSIGNMENT_COEF:-0.0}"
AUX_TRAP_COEF="${AUX_TRAP_COEF:-0.0}"
EQUALIZE_EPISODE_WEIGHT="${EQUALIZE_EPISODE_WEIGHT:-false}"
FRONT_WEIGHT_ALPHA="${FRONT_WEIGHT_ALPHA:-0.0}"
MIN_EPISODE_LEN="${MIN_EPISODE_LEN:--1}"
MAX_EPISODE_LEN="${MAX_EPISODE_LEN:--1}"
KEEP_PREFIX_STEPS="${KEEP_PREFIX_STEPS:--1}"
KEEP_SUFFIX_STEPS="${KEEP_SUFFIX_STEPS:-0}"
TP_MODEL_DIR="${TP_MODEL_DIR:-}"
MODEL_DIR="${MODEL_DIR:-}"
DEVICE="${DEVICE:-cuda:0}"

cd "${PROJECT_ROOT}"

CMD=(
    python3 scripts/train_expert_bc.py
    task=HideAndSeek
    headless=true
    wandb.mode=disabled
    --dataset_dir="${DATASET_DIR}"
    --epochs="${EPOCHS}"
    --batch_size="${BATCH_SIZE}"
    --save_tag="${SAVE_TAG}"
    --bc_lr="${BC_LR}"
    --entropy_bonus_coef="${ENTROPY_BONUS_COEF}"
    --action_mse_coef="${ACTION_MSE_COEF}"
    --log_prob_coef="${LOG_PROB_COEF}"
    --aux_vel_cmd_coef="${AUX_VEL_CMD_COEF}"
    --aux_waypoint_coef="${AUX_WAYPOINT_COEF}"
    --aux_assignment_coef="${AUX_ASSIGNMENT_COEF}"
    --aux_trap_coef="${AUX_TRAP_COEF}"
    --equalize_episode_weight="${EQUALIZE_EPISODE_WEIGHT}"
    --front_weight_alpha="${FRONT_WEIGHT_ALPHA}"
    --min_episode_len="${MIN_EPISODE_LEN}"
    --max_episode_len="${MAX_EPISODE_LEN}"
    --keep_prefix_steps="${KEEP_PREFIX_STEPS}"
    --keep_suffix_steps="${KEEP_SUFFIX_STEPS}"
    --device="${DEVICE}"
)

if [[ -n "${TP_MODEL_DIR}" ]]; then
    CMD+=("tp_model_dir=${TP_MODEL_DIR}")
fi
if [[ -n "${MODEL_DIR}" ]]; then
    CMD+=("model_dir=${MODEL_DIR}")
fi

"${CONDA_EXE}" run -n sim --no-capture-output "${CMD[@]}"
