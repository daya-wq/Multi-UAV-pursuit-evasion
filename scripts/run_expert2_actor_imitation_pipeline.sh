#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_EXE="${CONDA_EXE:-/home/uavlab/miniconda3/bin/conda}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

PRED_MODE="${PRED_MODE:-noise}"
V_DRONE_TEST="${V_DRONE_TEST:-1.5}"
V_PREY_TEST="${V_PREY_TEST:-1.5}"
V_PREY_SCHEDULE="${V_PREY_SCHEDULE:-1.5}"
EPISODE_LENGTH="${EPISODE_LENGTH:-1200}"
EVAL_GPU="${EVAL_GPU:-0}"
IMITATION_USE_TP_NET="${IMITATION_USE_TP_NET:-false}"

GENERIC_BATCH_ENVS="${GENERIC_BATCH_ENVS:-1024}"
NUM_WAVES="${NUM_WAVES:-50}"
GENERIC_SEED_BASE="${GENERIC_SEED_BASE:-20260411}"
MIN_SUCCESS_STEPS="${MIN_SUCCESS_STEPS:-25}"
DATASET_DTYPE="${DATASET_DTYPE:-float16}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/expert_datasets}"
DATASET_NAME="${DATASET_NAME:-expert2_antcoll_${GENERIC_BATCH_ENVS}x${NUM_WAVES}_${TIMESTAMP}}"

RUN_COLLECT="${RUN_COLLECT:-true}"
RUN_BC="${RUN_BC:-true}"
RUN_DAGGER="${RUN_DAGGER:-true}"
START_FROM_SCRATCH="${START_FROM_SCRATCH:-true}"

BC_EPOCHS="${BC_EPOCHS:-8}"
BC_BATCH_SIZE="${BC_BATCH_SIZE:-4096}"
BC_LR="${BC_LR:-0.0005}"
BC_DEVICE="${BC_DEVICE:-cuda:0}"
BC_SAVE_TAG="${BC_SAVE_TAG:-expert2_antcoll_bc_${GENERIC_BATCH_ENVS}x${NUM_WAVES}}"
BC_MODEL_DIR="${BC_MODEL_DIR:-}"
BC_TP_MODEL_DIR="${BC_TP_MODEL_DIR:-}"
BC_ACTION_MSE_COEF="${BC_ACTION_MSE_COEF:-1.0}"
BC_LOG_PROB_COEF="${BC_LOG_PROB_COEF:-0.0}"
BC_AUX_WAYPOINT_COEF="${BC_AUX_WAYPOINT_COEF:-0.02}"
BC_AUX_ASSIGNMENT_COEF="${BC_AUX_ASSIGNMENT_COEF:-0.05}"
BC_FRONT_WEIGHT_ALPHA="${BC_FRONT_WEIGHT_ALPHA:-0.5}"
BC_KEEP_PREFIX_STEPS="${BC_KEEP_PREFIX_STEPS:--1}"
BC_KEEP_SUFFIX_STEPS="${BC_KEEP_SUFFIX_STEPS:-64}"
BC_EVAL_EVERY="${BC_EVAL_EVERY:-1}"
BC_N_EVAL="${BC_N_EVAL:-1024}"
BC_EVAL_BATCH_ENVS="${BC_EVAL_BATCH_ENVS:-1024}"
BC_EVAL_SUCCESS_THRESHOLD="${BC_EVAL_SUCCESS_THRESHOLD:-0.60}"

DAGGER_MODEL_DIR="${DAGGER_MODEL_DIR:-}"
DAGGER_BATCH_ENVS="${DAGGER_BATCH_ENVS:-${GENERIC_BATCH_ENVS}}"
DAGGER_WAVES="${DAGGER_WAVES:-50}"
DAGGER_SEED_BASE="${DAGGER_SEED_BASE:-30360411}"
DAGGER_LR="${DAGGER_LR:-0.0001}"
DAGGER_ACCUM_BATCH_SIZE="${DAGGER_ACCUM_BATCH_SIZE:-8192}"
DAGGER_MIX_INIT="${DAGGER_MIX_INIT:-0.50}"
DAGGER_MIX_FINAL="${DAGGER_MIX_FINAL:-0.10}"
DAGGER_ONLINE_SUCCESS_ONLY="${DAGGER_ONLINE_SUCCESS_ONLY:-true}"
DAGGER_DEFER_ONLINE_UPDATES="${DAGGER_DEFER_ONLINE_UPDATES:-true}"
DAGGER_ONLINE_MIN_WAVE_CAPTURE_RATE="${DAGGER_ONLINE_MIN_WAVE_CAPTURE_RATE:-0.60}"
DAGGER_ONLINE_RATIO_WARMUP_WAVES="${DAGGER_ONLINE_RATIO_WARMUP_WAVES:-10}"
DAGGER_ONLINE_RATIO_INIT="${DAGGER_ONLINE_RATIO_INIT:-0.10}"
DAGGER_ONLINE_RATIO_STEP="${DAGGER_ONLINE_RATIO_STEP:-0.05}"
DAGGER_ONLINE_RATIO_MAX="${DAGGER_ONLINE_RATIO_MAX:-0.40}"
DAGGER_MIXED_UPDATES_PER_WAVE="${DAGGER_MIXED_UPDATES_PER_WAVE:-0}"
DAGGER_REPLAY_UPDATES_PER_WAVE="${DAGGER_REPLAY_UPDATES_PER_WAVE:-4}"
DAGGER_REPLAY_BATCH_SIZE="${DAGGER_REPLAY_BATCH_SIZE:-4096}"
DAGGER_REPLAY_FRONT_WEIGHT_ALPHA="${DAGGER_REPLAY_FRONT_WEIGHT_ALPHA:-0.5}"
DAGGER_ONLINE_GOAL_WEIGHT_ALPHA="${DAGGER_ONLINE_GOAL_WEIGHT_ALPHA:-0.5}"
DAGGER_ONLINE_DISAGREE_WEIGHT_ALPHA="${DAGGER_ONLINE_DISAGREE_WEIGHT_ALPHA:-2.0}"
DAGGER_EVAL_EVERY="${DAGGER_EVAL_EVERY:-10}"
DAGGER_N_EVAL="${DAGGER_N_EVAL:-1024}"
DAGGER_EVAL_BATCH_ENVS="${DAGGER_EVAL_BATCH_ENVS:-1024}"
DAGGER_SAVE_TAG="${DAGGER_SAVE_TAG:-expert2_antcoll_dagger_${DAGGER_BATCH_ENVS}x${DAGGER_WAVES}}"
DAGGER_DATASET_ROOT="${DAGGER_DATASET_ROOT:-${PROJECT_ROOT}/expert_datasets}"
DAGGER_DATASET_NAME="${DAGGER_DATASET_NAME:-dagger_expert2_antcoll_${DAGGER_BATCH_ENVS}x${DAGGER_WAVES}_${TIMESTAMP}}"

STRATEGY_VARIANT="expert2"
FORWARD_DIR_MODE="${FORWARD_DIR_MODE:-default}"
EXPERT2_FRONT_LAYOUT="${EXPERT2_FRONT_LAYOUT:-symmetric}"
ENABLE_GOAL_MODE="${ENABLE_GOAL_MODE:-false}"
ENABLE_CLOSE_MODE="${ENABLE_CLOSE_MODE:-true}"
ENABLE_RUSH_MODE="${ENABLE_RUSH_MODE:-false}"
EXPERT_INTERCEPT_PRED_STEP="${EXPERT_INTERCEPT_PRED_STEP:-5}"
EXPERT_INTERCEPT_USE_DIRECT_PRED="${EXPERT_INTERCEPT_USE_DIRECT_PRED:-false}"
DAGGER_TP_WEIGHT="${DAGGER_TP_WEIGHT:-${BC_TP_MODEL_DIR}}"

DATASET_DIR="${DATASET_ROOT}/${DATASET_NAME}"

if [[ "${START_FROM_SCRATCH}" == "true" ]]; then
    BC_MODEL_DIR=""
    DAGGER_MODEL_DIR=""
fi

echo "========================================================"
echo "  Expert2 actor imitation pipeline"
echo "  strategy       = ${STRATEGY_VARIANT} goal=${ENABLE_GOAL_MODE} close=${ENABLE_CLOSE_MODE} rush=${ENABLE_RUSH_MODE}"
echo "  from scratch   = ${START_FROM_SCRATCH}"
echo "  collect        = ${RUN_COLLECT} | batch_envs=${GENERIC_BATCH_ENVS} waves=${NUM_WAVES}"
echo "  dagger         = ${RUN_DAGGER} | batch_envs=${DAGGER_BATCH_ENVS} waves=${DAGGER_WAVES}"
echo "  env TP net     = ${IMITATION_USE_TP_NET}"
echo "  intercept      = step${EXPERT_INTERCEPT_PRED_STEP} + $([[ "${EXPERT_INTERCEPT_USE_DIRECT_PRED}" == "true" ]] && echo "direct_pred" || echo "lookahead")"
echo "  dagger data    = expert replay + online gate>${DAGGER_ONLINE_MIN_WAVE_CAPTURE_RATE}, online ratio ${DAGGER_ONLINE_RATIO_INIT} warmup=${DAGGER_ONLINE_RATIO_WARMUP_WAVES} step=${DAGGER_ONLINE_RATIO_STEP} max=${DAGGER_ONLINE_RATIO_MAX}"
echo "  bc eval gate   = ${BC_EVAL_SUCCESS_THRESHOLD} | eval_every=${BC_EVAL_EVERY} n_eval=${BC_N_EVAL} batch_envs=${BC_EVAL_BATCH_ENVS}"
echo "  v_prey_schedule= ${V_PREY_SCHEDULE}"
echo "  dataset_dir    = ${DATASET_DIR}"
echo "========================================================"

mkdir -p "${DATASET_ROOT}"

if [[ "${RUN_COLLECT}" == "true" ]]; then
    COLLECT_SUCCESS_DATASET=true \
    DATASET_DIR="${DATASET_ROOT}" \
    DATASET_NAME="${DATASET_NAME}" \
    MIN_SUCCESS_STEPS="${MIN_SUCCESS_STEPS}" \
    DATASET_DTYPE="${DATASET_DTYPE}" \
    GENERIC_BATCH_ENVS="${GENERIC_BATCH_ENVS}" \
    NUM_WAVES="${NUM_WAVES}" \
    GENERIC_SEED_BASE="${GENERIC_SEED_BASE}" \
    V_PREY_SCHEDULE="${V_PREY_SCHEDULE}" \
    V_DRONE_TEST="${V_DRONE_TEST}" \
    EPISODE_LENGTH="${EPISODE_LENGTH}" \
    EVAL_GPU="${EVAL_GPU}" \
    ALGO_USE_TP_NET="$([[ "${IMITATION_USE_TP_NET}" == "true" ]] && echo 1 || echo 0)" \
    STRATEGY_VARIANT="${STRATEGY_VARIANT}" \
    FORWARD_DIR_MODE="${FORWARD_DIR_MODE}" \
    EXPERT2_FRONT_LAYOUT="${EXPERT2_FRONT_LAYOUT}" \
    ENABLE_GOAL_MODE="${ENABLE_GOAL_MODE}" \
    ENABLE_CLOSE_MODE="${ENABLE_CLOSE_MODE}" \
    ENABLE_RUSH_MODE="${ENABLE_RUSH_MODE}" \
    EXPERT_INTERCEPT_PRED_STEP="${EXPERT_INTERCEPT_PRED_STEP}" \
    EXPERT_INTERCEPT_USE_DIRECT_PRED="${EXPERT_INTERCEPT_USE_DIRECT_PRED}" \
    bash "${PROJECT_ROOT}/scripts/expert_isaac_eval.sh" "${PRED_MODE}" "${V_PREY_TEST}" 0 0
fi

if [[ ! -d "${DATASET_DIR}" ]]; then
    echo "[ERROR] Dataset directory not found: ${DATASET_DIR}" >&2
    exit 1
fi

if [[ "${RUN_BC}" == "true" ]]; then
    EPOCHS="${BC_EPOCHS}" \
    BATCH_SIZE="${BC_BATCH_SIZE}" \
    SAVE_TAG="${BC_SAVE_TAG}" \
    BC_LR="${BC_LR}" \
    ACTION_MSE_COEF="${BC_ACTION_MSE_COEF}" \
    LOG_PROB_COEF="${BC_LOG_PROB_COEF}" \
    AUX_WAYPOINT_COEF="${BC_AUX_WAYPOINT_COEF}" \
    AUX_ASSIGNMENT_COEF="${BC_AUX_ASSIGNMENT_COEF}" \
    FRONT_WEIGHT_ALPHA="${BC_FRONT_WEIGHT_ALPHA}" \
    KEEP_PREFIX_STEPS="${BC_KEEP_PREFIX_STEPS}" \
    KEEP_SUFFIX_STEPS="${BC_KEEP_SUFFIX_STEPS}" \
    EVAL_EVERY="${BC_EVAL_EVERY}" \
    N_EVAL="${BC_N_EVAL}" \
    EVAL_BATCH_ENVS="${BC_EVAL_BATCH_ENVS}" \
    EPISODE_LENGTH="${EPISODE_LENGTH}" \
    V_PREY_TEST="${V_PREY_TEST}" \
    V_DRONE_TEST="${V_DRONE_TEST}" \
    EVAL_SUCCESS_THRESHOLD="${BC_EVAL_SUCCESS_THRESHOLD}" \
    DEVICE="${BC_DEVICE}" \
    ALGO_USE_TP_NET="$([[ "${IMITATION_USE_TP_NET}" == "true" ]] && echo 1 || echo 0)" \
    MODEL_DIR="${BC_MODEL_DIR}" \
    TP_MODEL_DIR="${BC_TP_MODEL_DIR}" \
    bash "${PROJECT_ROOT}/scripts/train_expert_bc.sh" "${DATASET_DIR}"
fi

BC_SUMMARY_PATH="$(find "${PROJECT_ROOT}/checkpoints" -path "*${BC_SAVE_TAG}_*" -name "bc_summary.json" | sort | tail -n1 || true)"
BC_BEST_CAPTURE="0.0"
BC_THRESHOLD_REACHED="false"
if [[ -n "${BC_SUMMARY_PATH}" ]]; then
    BC_BEST_CAPTURE="$(python3 - <<'PY' "${BC_SUMMARY_PATH}"
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
print(data.get("best_capture_rate", 0.0))
PY
)"
    BC_THRESHOLD_REACHED="$(python3 - <<'PY' "${BC_SUMMARY_PATH}" "${BC_EVAL_SUCCESS_THRESHOLD}"
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
best = float(data.get("best_capture_rate", 0.0))
thr = float(sys.argv[2])
print("true" if best >= thr else "false")
PY
)"
    if [[ -z "${DAGGER_MODEL_DIR}" ]]; then
        DAGGER_MODEL_DIR="$(python3 - <<'PY' "${BC_SUMMARY_PATH}"
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
print(data.get("best_checkpoint") or "")
PY
)"
    fi
fi

if [[ "${RUN_DAGGER}" == "true" && "${BC_THRESHOLD_REACHED}" != "true" ]]; then
    echo "[INFO] Skip DAgger: BC best capture ${BC_BEST_CAPTURE} < gate ${BC_EVAL_SUCCESS_THRESHOLD}"
    RUN_DAGGER="false"
fi
if [[ "${RUN_DAGGER}" == "true" && -z "${DAGGER_MODEL_DIR}" ]]; then
    echo "[ERROR] Could not find BC best checkpoint for DAgger." >&2
    exit 1
fi

if [[ "${RUN_DAGGER}" == "true" ]]; then
    cd "${PROJECT_ROOT}"
    CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
    PYTHONUNBUFFERED=1 \
    "${CONDA_EXE}" run -n sim --no-capture-output \
        python3 scripts/train_actor_dagger.py \
        task=HideAndSeek \
        headless=true \
        wandb.mode=disabled \
        algo.use_TP_net=$([[ "${IMITATION_USE_TP_NET}" == "true" ]] && echo 1 || echo 0) \
        task.sim.device=cuda:0 \
        task.sim.active_gpu=0 \
        task.sim.physics_gpu=0 \
        --model_dir="${DAGGER_MODEL_DIR}" \
        --pred_mode="${PRED_MODE}" \
        --strategy_variant="${STRATEGY_VARIANT}" \
        --forward_dir_mode="${FORWARD_DIR_MODE}" \
        --enable_goal_mode="${ENABLE_GOAL_MODE}" \
        --enable_close_mode="${ENABLE_CLOSE_MODE}" \
        --enable_rush_mode="${ENABLE_RUSH_MODE}" \
        --expert2_front_layout="${EXPERT2_FRONT_LAYOUT}" \
        --expert_intercept_pred_step="${EXPERT_INTERCEPT_PRED_STEP}" \
        --expert_intercept_use_direct_pred="${EXPERT_INTERCEPT_USE_DIRECT_PRED}" \
        --tp_weight="${DAGGER_TP_WEIGHT}" \
        --waves="${DAGGER_WAVES}" \
        --batch_envs="${DAGGER_BATCH_ENVS}" \
        --episode_length="${EPISODE_LENGTH}" \
        --v_prey_test="${V_PREY_TEST}" \
        --v_prey_schedule="${V_PREY_SCHEDULE}" \
        --v_drone_test="${V_DRONE_TEST}" \
        --bc_lr="${DAGGER_LR}" \
        --accum_batch_size="${DAGGER_ACCUM_BATCH_SIZE}" \
        --expert_mix_prob_init="${DAGGER_MIX_INIT}" \
        --expert_mix_prob_final="${DAGGER_MIX_FINAL}" \
        --online_success_only="${DAGGER_ONLINE_SUCCESS_ONLY}" \
        --defer_online_updates="${DAGGER_DEFER_ONLINE_UPDATES}" \
        --online_min_wave_capture_rate="${DAGGER_ONLINE_MIN_WAVE_CAPTURE_RATE}" \
        --online_ratio_warmup_waves="${DAGGER_ONLINE_RATIO_WARMUP_WAVES}" \
        --online_ratio_init="${DAGGER_ONLINE_RATIO_INIT}" \
        --online_ratio_step="${DAGGER_ONLINE_RATIO_STEP}" \
        --online_ratio_max="${DAGGER_ONLINE_RATIO_MAX}" \
        --mixed_updates_per_wave="${DAGGER_MIXED_UPDATES_PER_WAVE}" \
        --online_dataset_dir="${DAGGER_DATASET_ROOT}" \
        --online_dataset_name="${DAGGER_DATASET_NAME}" \
        --online_dataset_dtype="${DATASET_DTYPE}" \
        --replay_dataset_dir="${DATASET_DIR}" \
        --replay_updates_per_wave="${DAGGER_REPLAY_UPDATES_PER_WAVE}" \
        --replay_batch_size="${DAGGER_REPLAY_BATCH_SIZE}" \
        --replay_front_weight_alpha="${DAGGER_REPLAY_FRONT_WEIGHT_ALPHA}" \
        --online_goal_weight_alpha="${DAGGER_ONLINE_GOAL_WEIGHT_ALPHA}" \
        --online_disagreement_weight_alpha="${DAGGER_ONLINE_DISAGREE_WEIGHT_ALPHA}" \
        --eval_every="${DAGGER_EVAL_EVERY}" \
        --n_eval="${DAGGER_N_EVAL}" \
        --eval_batch_envs="${DAGGER_EVAL_BATCH_ENVS}" \
        --seed_base="${DAGGER_SEED_BASE}" \
        --save_tag="${DAGGER_SAVE_TAG}" \
        --device="cuda:0"
fi

echo "========================================================"
echo "  Pipeline finished"
echo "  expert dataset : ${DATASET_DIR}"
echo "  bc summary     : ${BC_SUMMARY_PATH:-none}"
echo "  bc best cap    : ${BC_BEST_CAPTURE}"
echo "  dagger dataset : ${DAGGER_DATASET_ROOT}/${DAGGER_DATASET_NAME}"
echo "  dagger init    : ${DAGGER_MODEL_DIR:-none}"
echo "========================================================"
