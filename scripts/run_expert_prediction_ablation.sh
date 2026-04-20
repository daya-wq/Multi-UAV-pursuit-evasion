#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

V_PREY_TEST="${V_PREY_TEST:-1.5}"
V_DRONE_TEST="${V_DRONE_TEST:-1.5}"
EPISODE_LENGTH="${EPISODE_LENGTH:-1000}"
GENERIC_BATCH_ENVS="${GENERIC_BATCH_ENVS:-256}"
NUM_WAVES="${NUM_WAVES:-20}"
EVAL_GPU="${EVAL_GPU:-0}"
N_VIDEO="${N_VIDEO:-0}"
N_GENERIC="${N_GENERIC:-0}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/analysis/expert_prediction_ablation}"

if [[ "$#" -gt 0 ]]; then
    MODES=("$@")
else
    MODES=("noise" "tp_net" "oracle_pos" "oracle_next")
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUT_ROOT}/${TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "========================================================"
echo "  Expert Prediction Ablation"
echo "  modes             = ${MODES[*]}"
echo "  v_prey / v_drone  = ${V_PREY_TEST} / ${V_DRONE_TEST}"
echo "  episode_length    = ${EPISODE_LENGTH}"
echo "  generic_batch_envs= ${GENERIC_BATCH_ENVS}"
echo "  num_waves         = ${NUM_WAVES}"
echo "  eval_gpu          = ${EVAL_GPU}"
echo "  run_dir           = ${RUN_DIR}"
echo "========================================================"

for mode in "${MODES[@]}"; do
    mode_log="${LOG_DIR}/${mode}.log"
    echo ""
    echo "--------------------------------------------------------"
    echo "  Running mode=${mode}"
    echo "  log=${mode_log}"
    echo "--------------------------------------------------------"
    (
        cd "${PROJECT_ROOT}"
        EVAL_GPU="${EVAL_GPU}" \
        V_DRONE_TEST="${V_DRONE_TEST}" \
        EPISODE_LENGTH="${EPISODE_LENGTH}" \
        GENERIC_BATCH_ENVS="${GENERIC_BATCH_ENVS}" \
        NUM_WAVES="${NUM_WAVES}" \
        bash "${PROJECT_ROOT}/scripts/expert_isaac_eval.sh" "${mode}" "${V_PREY_TEST}" "${N_VIDEO}" "${N_GENERIC}"
    ) | tee "${mode_log}"
done

python3 "${PROJECT_ROOT}/scripts/summarize_expert_prediction_ablation.py" \
    --log_dir "${LOG_DIR}" \
    --out_dir "${RUN_DIR}" \
    --v_prey "${V_PREY_TEST}" \
    --v_drone "${V_DRONE_TEST}" \
    --episode_length "${EPISODE_LENGTH}" \
    --batch_envs "${GENERIC_BATCH_ENVS}" \
    --num_waves "${NUM_WAVES}" \
    --modes "${MODES[@]}"

echo ""
echo "========================================================"
echo "  Done"
echo "  run_dir  = ${RUN_DIR}"
echo "  summary  = ${RUN_DIR}/summary.md"
echo "========================================================"
