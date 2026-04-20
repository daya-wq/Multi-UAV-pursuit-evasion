#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -n "${PYTHON_BIN}" ]]; then
  PYTHON_CMD=("${PYTHON_BIN}")
else
  PYTHON_CMD=(conda run -n sim python)
fi

NOW="$(date +%Y%m%d_%H%M%S)"
NUM_WAVES="${NUM_WAVES:-5}"
GENERIC_BATCH_ENVS="${GENERIC_BATCH_ENVS:-1024}"
N_GENERIC="${N_GENERIC:-$((NUM_WAVES * GENERIC_BATCH_ENVS))}"
V_DRONE_TEST="${V_DRONE_TEST:-1.5}"
V_PREY_TEST="${V_PREY_TEST:-1.5}"
EPISODE_LENGTH="${EPISODE_LENGTH:-1200}"

DATASET_NAME="${DATASET_NAME:-tp_expert2_oraclenext_frontbox_1024x5_${NOW}}"
TP_SAVE_TAG="${TP_SAVE_TAG:-tp_supervised_${NOW}}"
ANALYSIS_DIR="${ANALYSIS_DIR:-analysis/${DATASET_NAME}}"

mkdir -p "${PROJECT_ROOT}/tp_datasets" "${PROJECT_ROOT}/analysis"

COLLECT_LOG="${PROJECT_ROOT}/${ANALYSIS_DIR}/collect_tp.log"
TRAIN_LOG="${PROJECT_ROOT}/${ANALYSIS_DIR}/train_tp.log"
mkdir -p "$(dirname "${COLLECT_LOG}")"

echo "============================================================"
echo "TP retrain pipeline"
echo "  dataset_name      = ${DATASET_NAME}"
echo "  analysis_dir      = ${PROJECT_ROOT}/${ANALYSIS_DIR}"
echo "  num_waves         = ${NUM_WAVES}"
echo "  batch_envs        = ${GENERIC_BATCH_ENVS}"
echo "  n_generic         = ${N_GENERIC}"
echo "  v_drone           = ${V_DRONE_TEST}"
echo "  v_prey            = ${V_PREY_TEST}"
echo "  episode_length    = ${EPISODE_LENGTH}"
echo "  tp_save_tag       = ${TP_SAVE_TAG}"
echo "============================================================"

"${PYTHON_CMD[@]}" scripts/expert_isaac_eval.py \
  --pred_mode oracle_next \
  --strategy_variant expert2 \
  --enable_goal_mode false \
  --enable_close_mode true \
  --enable_rush_mode false \
  --expert2_front_layout symmetric \
  --n_video 0 \
  --n_generic "${N_GENERIC}" \
  --generic_batch_envs "${GENERIC_BATCH_ENVS}" \
  --random_init true \
  --episode_length "${EPISODE_LENGTH}" \
  --max_steps "${EPISODE_LENGTH}" \
  --collect_tp_dataset true \
  --tp_dataset_dir "${PROJECT_ROOT}/tp_datasets" \
  --tp_dataset_name "${DATASET_NAME}" \
  --tp_dataset_dtype float16 \
  --v_drone_test "${V_DRONE_TEST}" \
  --v_prey_test "${V_PREY_TEST}" \
  task=HideAndSeek \
  headless=true \
  task.sim.device=cuda:0 \
  algo.use_TP_net=1 \
  |& tee "${COLLECT_LOG}"

"${PYTHON_CMD[@]}" scripts/train_tp_supervised.py \
  --dataset_dir "${PROJECT_ROOT}/tp_datasets/${DATASET_NAME}" \
  --save_dir "${PROJECT_ROOT}/checkpoints" \
  --save_tag "${TP_SAVE_TAG}" \
  --analysis_dir "${PROJECT_ROOT}/${ANALYSIS_DIR}/tp_train" \
  --epochs "${TP_EPOCHS:-20}" \
  --batch_size "${TP_BATCH_SIZE:-8192}" \
  --lr "${TP_LR:-1e-4}" \
  --weight_decay "${TP_WEIGHT_DECAY:-0.0}" \
  --val_ratio "${TP_VAL_RATIO:-0.1}" \
  --num_workers "${TP_NUM_WORKERS:-4}" \
  --device "${TP_DEVICE:-cuda:0}" \
  |& tee "${TRAIN_LOG}"
