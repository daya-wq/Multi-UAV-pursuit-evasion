#!/usr/bin/env bash
# expert_isaac_eval.sh
# =====================
# 在 Isaac Sim 中运行专家策略可视化评估：
#   - 录制 2 个视频（随机初始化）
#   - 测试随机初始位置下的专家泛化能力（默认 100 轮）
#
# 使用 GPU 0（唯一有 Vulkan context 的 GPU）。
# headless=true + enable_render(True) 实现 offscreen 录制，无需显示器。
#
# 用法：
#   bash scripts/expert_isaac_eval.sh                           # tp_net 模式
#   bash scripts/expert_isaac_eval.sh noise                     # noise 模式
#   bash scripts/expert_isaac_eval.sh oracle_pos                # 真实当前位置 oracle
#   bash scripts/expert_isaac_eval.sh oracle_next               # 真实下一时刻 oracle
#   bash scripts/expert_isaac_eval.sh tp_net 1.5 2 100         # 完整参数
#
# 参数顺序（均可选）：
#   $1  pred_mode    noise | tp_net | oracle_pos | oracle_next   (default: tp_net)
#   $2  v_prey_test  float            (default: 1.5)
#   $3  n_video      int              (default: 2)
#   $4  n_generic    int              (default: 100)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_EXE="${CONDA_EXE:-/home/uavlab/miniconda3/bin/conda}"
ISAACSIM_PATH="${ISAACSIM_PATH:-/data/uavlab/isaac_sim_2022.2.0}"

# ── GPU 0 (the active Vulkan GPU; GPU 1 has no Vulkan context)
# headless=true + enable_render(True) → offscreen rendering, no display needed
EVAL_GPU="${EVAL_GPU:-0}"
ALGO_USE_TP_NET="${ALGO_USE_TP_NET:-1}"

PRED_MODE="${1:-tp_net}"
V_PREY_TEST="${2:-1.5}"
N_VIDEO="${3:-2}"
N_GENERIC="${4:-100}"
V_PREY_SCHEDULE="${V_PREY_SCHEDULE:-}"
V_DRONE_TEST="${V_DRONE_TEST:-1.5}"
EPISODE_LENGTH="${EPISODE_LENGTH:-1000}"
if [[ "${N_VIDEO}" == "0" ]]; then
    GENERIC_BATCH_ENVS="${GENERIC_BATCH_ENVS:-8}"
else
    GENERIC_BATCH_ENVS="${GENERIC_BATCH_ENVS:-1}"
fi
NUM_WAVES="${NUM_WAVES:-}"
if [[ -n "${NUM_WAVES}" && "${N_VIDEO}" == "0" ]]; then
    N_GENERIC="$(( GENERIC_BATCH_ENVS * NUM_WAVES ))"
fi
COLLECT_SUCCESS_DATASET="${COLLECT_SUCCESS_DATASET:-false}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/expert_datasets}"
DATASET_NAME="${DATASET_NAME:-}"
MIN_SUCCESS_STEPS="${MIN_SUCCESS_STEPS:-1}"
DATASET_DTYPE="${DATASET_DTYPE:-float16}"
GENERIC_SEED_BASE="${GENERIC_SEED_BASE:-999}"
STRATEGY_VARIANT="${STRATEGY_VARIANT:-baseline}"
FORWARD_DIR_MODE="${FORWARD_DIR_MODE:-default}"
EXPERT2_FRONT_LAYOUT="${EXPERT2_FRONT_LAYOUT:-symmetric}"
EXPERT_INTERCEPT_PRED_STEP="${EXPERT_INTERCEPT_PRED_STEP:-5}"
EXPERT_INTERCEPT_USE_DIRECT_PRED="${EXPERT_INTERCEPT_USE_DIRECT_PRED:-false}"
if [[ "${STRATEGY_VARIANT}" == "expert2" ]]; then
    ENABLE_GOAL_MODE="${ENABLE_GOAL_MODE:-false}"
    ENABLE_CLOSE_MODE="${ENABLE_CLOSE_MODE:-true}"
    ENABLE_RUSH_MODE="${ENABLE_RUSH_MODE:-false}"
else
    ENABLE_GOAL_MODE="${ENABLE_GOAL_MODE:-true}"
    ENABLE_CLOSE_MODE="${ENABLE_CLOSE_MODE:-true}"
    ENABLE_RUSH_MODE="${ENABLE_RUSH_MODE:-true}"
fi

# 自动找 TP 权重（仅 tp_net 模式需要）
TP_WEIGHT=""
if [[ "${PRED_MODE}" == "tp_net" ]]; then
    TP_WEIGHT=$(find "${PROJECT_ROOT}/checkpoints" -name "tp_only_*.pt" 2>/dev/null \
        | sort | tail -n1 || true)
    if [[ -z "${TP_WEIGHT}" ]]; then
        echo "[WARN] tp_only_*.pt not found, will use noise mode"
        PRED_MODE="noise"
        TP_WEIGHT=""
    fi
fi

VIDEO_DIR="${PROJECT_ROOT}/eval_videos/expert"
VIDEO_SEED_BASE="${VIDEO_SEED_BASE:-0}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${PROJECT_ROOT}/expert_eval_${TIMESTAMP}.log"

echo "========================================================"
echo "  Expert Isaac Sim Evaluation"
echo "  pred_mode   = ${PRED_MODE}"
echo "  v_prey_test = ${V_PREY_TEST}"
if [[ -n "${V_PREY_SCHEDULE}" ]]; then
echo "  v_prey_sched= ${V_PREY_SCHEDULE}"
fi
echo "  n_video     = ${N_VIDEO}"
echo "  n_generic   = ${N_GENERIC}"
echo "  v_drone_test= ${V_DRONE_TEST}"
echo "  ep_length   = ${EPISODE_LENGTH}"
echo "  batch_envs  = ${GENERIC_BATCH_ENVS}"
if [[ -n "${NUM_WAVES}" ]]; then
echo "  num_waves   = ${NUM_WAVES}"
fi
echo "  collect_ds  = ${COLLECT_SUCCESS_DATASET}"
echo "  strategy    = ${STRATEGY_VARIANT}"
echo "  forward_dir = ${FORWARD_DIR_MODE}"
echo "  goal_mode   = ${ENABLE_GOAL_MODE}"
echo "  close_mode  = ${ENABLE_CLOSE_MODE}"
echo "  rush_mode   = ${ENABLE_RUSH_MODE}"
echo "  front_layout= ${EXPERT2_FRONT_LAYOUT}"
echo "  intercept   = step${EXPERT_INTERCEPT_PRED_STEP} + $([[ \"${EXPERT_INTERCEPT_USE_DIRECT_PRED}\" == \"true\" ]] && echo \"direct_pred\" || echo \"lookahead\")"
if [[ "${COLLECT_SUCCESS_DATASET}" == "true" ]]; then
echo "  dataset_dir = ${DATASET_DIR}"
echo "  min_succ_st = ${MIN_SUCCESS_STEPS}"
echo "  dataset_dt  = ${DATASET_DTYPE}"
fi
echo "  GPU         = ${EVAL_GPU} (CUDA device 0 inside process)"
echo "  use_TP_net  = ${ALGO_USE_TP_NET}"
echo "  tp_weight   = ${TP_WEIGHT:-none}"
echo "  video_dir   = ${VIDEO_DIR}"
echo "  video_seed  = ${VIDEO_SEED_BASE}"
echo "  generic_seed= ${GENERIC_SEED_BASE}"
echo "  log         = ${LOG_FILE}"
echo "========================================================"

mkdir -p "${VIDEO_DIR}"

export ISAACSIM_PATH="${ISAACSIM_PATH}"
cd "${PROJECT_ROOT}"

"${CONDA_EXE}" run -n sim --no-capture-output \
    env \
        CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
        PYTHONUNBUFFERED=1 \
    python3 scripts/expert_isaac_eval.py \
        task=HideAndSeek \
        headless=true \
        wandb.mode=disabled \
        task.env.num_envs=1 \
        task.sim.device=cuda:0 \
        task.sim.active_gpu=0 \
        task.sim.physics_gpu=0 \
        algo.use_TP_net="${ALGO_USE_TP_NET}" \
        --pred_mode="${PRED_MODE}" \
        --strategy_variant="${STRATEGY_VARIANT}" \
        --forward_dir_mode="${FORWARD_DIR_MODE}" \
        --enable_goal_mode="${ENABLE_GOAL_MODE}" \
        --enable_close_mode="${ENABLE_CLOSE_MODE}" \
        --enable_rush_mode="${ENABLE_RUSH_MODE}" \
        --expert2_front_layout="${EXPERT2_FRONT_LAYOUT}" \
        --expert_intercept_pred_step="${EXPERT_INTERCEPT_PRED_STEP}" \
        --expert_intercept_use_direct_pred="${EXPERT_INTERCEPT_USE_DIRECT_PRED}" \
        --tp_weight="${TP_WEIGHT}" \
        --n_video="${N_VIDEO}" \
        --n_generic="${N_GENERIC}" \
        --v_prey_test="${V_PREY_TEST}" \
        --v_prey_schedule="${V_PREY_SCHEDULE}" \
        --v_drone_test="${V_DRONE_TEST}" \
        --video_dir="${VIDEO_DIR}" \
        --video_seed_base="${VIDEO_SEED_BASE}" \
        --generic_seed_base="${GENERIC_SEED_BASE}" \
        --generic_batch_envs="${GENERIC_BATCH_ENVS}" \
        --random_init=true \
        --episode_length="${EPISODE_LENGTH}" \
        --max_steps="${EPISODE_LENGTH}" \
        --collect_success_dataset="${COLLECT_SUCCESS_DATASET}" \
        --dataset_dir="${DATASET_DIR}" \
        --dataset_name="${DATASET_NAME}" \
        --min_success_steps="${MIN_SUCCESS_STEPS}" \
        --dataset_dtype="${DATASET_DTYPE}" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "========================================================"
echo "  Done! Videos saved to: ${VIDEO_DIR}"
echo "  Log: ${LOG_FILE}"
echo "========================================================"
