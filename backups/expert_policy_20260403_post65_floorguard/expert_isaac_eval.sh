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
#   bash scripts/expert_isaac_eval.sh tp_net 1.5 2 100         # 完整参数
#
# 参数顺序（均可选）：
#   $1  pred_mode    noise | tp_net   (default: tp_net)
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

PRED_MODE="${1:-tp_net}"
V_PREY_TEST="${2:-1.5}"
N_VIDEO="${3:-2}"
N_GENERIC="${4:-100}"
V_DRONE_TEST="${V_DRONE_TEST:-1.5}"
EPISODE_LENGTH="${EPISODE_LENGTH:-1000}"

# 自动找 TP 权重
TP_WEIGHT=$(find "${PROJECT_ROOT}/checkpoints" -name "tp_only_*.pt" 2>/dev/null \
    | sort | tail -n1 || true)
if [[ -z "${TP_WEIGHT}" ]]; then
    echo "[WARN] tp_only_*.pt not found, will use noise mode"
    PRED_MODE="noise"
    TP_WEIGHT=""
fi

VIDEO_DIR="${PROJECT_ROOT}/eval_videos/expert"
VIDEO_SEED_BASE="${VIDEO_SEED_BASE:-0}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${PROJECT_ROOT}/expert_eval_${TIMESTAMP}.log"

echo "========================================================"
echo "  Expert Isaac Sim Evaluation"
echo "  pred_mode   = ${PRED_MODE}"
echo "  v_prey_test = ${V_PREY_TEST}"
echo "  n_video     = ${N_VIDEO}"
echo "  n_generic   = ${N_GENERIC}"
echo "  v_drone_test= ${V_DRONE_TEST}"
echo "  ep_length   = ${EPISODE_LENGTH}"
echo "  GPU         = ${EVAL_GPU} (CUDA device 0 inside process)"
echo "  tp_weight   = ${TP_WEIGHT:-none}"
echo "  video_dir   = ${VIDEO_DIR}"
echo "  video_seed  = ${VIDEO_SEED_BASE}"
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
        --pred_mode="${PRED_MODE}" \
        --tp_weight="${TP_WEIGHT}" \
        --n_video="${N_VIDEO}" \
        --n_generic="${N_GENERIC}" \
        --v_prey_test="${V_PREY_TEST}" \
        --v_drone_test="${V_DRONE_TEST}" \
        --video_dir="${VIDEO_DIR}" \
        --video_seed_base="${VIDEO_SEED_BASE}" \
        --random_init=true \
        --episode_length="${EPISODE_LENGTH}" \
        --max_steps="${EPISODE_LENGTH}" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "========================================================"
echo "  Done! Videos saved to: ${VIDEO_DIR}"
echo "  Log: ${LOG_FILE}"
echo "========================================================"
