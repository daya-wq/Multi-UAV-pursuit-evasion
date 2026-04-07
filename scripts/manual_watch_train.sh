#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data/uavlab/multi-uav-pursuit2"
CONDA_EXE="/home/uavlab/miniconda3/bin/conda"
ISAACSIM_PATH="/data/uavlab/isaac_sim_2022.2.0"
GPU_PHYSICAL="${GPU_PHYSICAL:-1}"
TOTAL_FRAMES="${TOTAL_FRAMES:-2000000000}"
NUM_ENVS="${NUM_ENVS:-2048}"
EVAL_INTERVAL="${EVAL_INTERVAL:-4000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
CHECK_INTERVAL="${CHECK_INTERVAL:-30}"
RESTART_PAUSE="${RESTART_PAUSE:-10}"

STATE_DIR="${PROJECT_ROOT}/.manual_train_watch"
SUPERVISOR_LOG="${PROJECT_ROOT}/manual_train_supervisor.log"
RECOVERY_LOG="${PROJECT_ROOT}/manual_train_recovery.log"
SUPERVISOR_PID_FILE="${PROJECT_ROOT}/manual_train_supervisor.pid"
CURRENT_PID_FILE="${STATE_DIR}/current_launcher.pid"
CURRENT_LOG_FILE="${STATE_DIR}/current_log.txt"
CURRENT_CKPT_FILE="${STATE_DIR}/current_checkpoint.txt"

mkdir -p "${STATE_DIR}"
touch "${SUPERVISOR_LOG}" "${RECOVERY_LOG}"
echo $$ > "${SUPERVISOR_PID_FILE}"

cleanup_on_exit() {
    rm -f "${SUPERVISOR_PID_FILE}"
}
trap cleanup_on_exit EXIT

log() {
    local msg="$*"
    printf '[%s] %s\n' "$(date '+%F %T')" "${msg}" | tee -a "${SUPERVISOR_LOG}"
}

record_recovery() {
    local msg="$*"
    printf '[%s] %s\n' "$(date '+%F %T')" "${msg}" | tee -a "${RECOVERY_LOG}" | tee -a "${SUPERVISOR_LOG}"
}

current_pid() {
    [[ -f "${CURRENT_PID_FILE}" ]] && cat "${CURRENT_PID_FILE}" || true
}

current_log() {
    [[ -f "${CURRENT_LOG_FILE}" ]] && cat "${CURRENT_LOG_FILE}" || true
}

set_current_state() {
    local pid="$1"
    local log_file="$2"
    local checkpoint="${3:-}"
    printf '%s\n' "${pid}" > "${CURRENT_PID_FILE}"
    printf '%s\n' "${log_file}" > "${CURRENT_LOG_FILE}"
    printf '%s\n' "${checkpoint}" > "${CURRENT_CKPT_FILE}"
}

find_latest_checkpoint() {
    find "${PROJECT_ROOT}/checkpoints" -maxdepth 2 -type f -name 'checkpoint_*.pt' -printf '%T@ %p\n' 2>/dev/null \
        | sort -n \
        | tail -n 1 \
        | cut -d' ' -f2-
}

log_indicates_completion() {
    local log_file="$1"
    [[ -n "${log_file}" && -f "${log_file}" ]] || return 1
    rg -q 'Final Eval at|checkpoint_final\.pt' "${log_file}"
}

summarize_failure() {
    local log_file="$1"
    local cause=""

    if [[ -z "${log_file}" || ! -f "${log_file}" ]]; then
        echo "missing_log"
        return
    fi

    cause="$(
        rg -i -o \
            'NotImplementedError:.*|ModuleNotFoundError:.*|cuda out of memory.*|OutOfMemoryError:.*|cudaError.*|Segmentation fault.*|段错误.*|core dumped.*|KeyboardInterrupt.*|Killed.*|Traceback.*' \
            "${log_file}" \
            | tail -n 1 \
            || true
    )"

    if [[ -n "${cause}" ]]; then
        echo "${cause}"
        return
    fi

    tail -n 40 "${log_file}" | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g' | cut -c1-240
}

cleanup_residuals() {
    local stale_pid="${1:-}"

    log "Cleanup: removing stale train/Isaac processes and GPU leftovers"

    if [[ -n "${stale_pid}" ]]; then
        kill -9 "${stale_pid}" 2>/dev/null || true
    fi

    pkill -9 -f 'scripts/train.py' 2>/dev/null || true
    pkill -9 -u uavlab -f kit 2>/dev/null || true

    if command -v nvidia-smi >/dev/null 2>&1; then
        while IFS=, read -r raw_pid raw_name raw_mem; do
            local pid name owner
            pid="$(echo "${raw_pid}" | xargs || true)"
            name="$(echo "${raw_name}" | xargs || true)"
            [[ -n "${pid}" ]] || continue
            [[ "${name}" == *gnome-remote-desktop-daemon* ]] && continue
            owner="$(ps -o user= -p "${pid}" 2>/dev/null | xargs || true)"
            [[ "${owner}" == "uavlab" ]] || continue
            kill -9 "${pid}" 2>/dev/null || true
        done < <(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || true)
    fi

    sleep 2
}

launch_training() {
    local checkpoint="${1:-}"
    local ts log_file launcher_pid
    ts="$(date +%Y%m%d_%H%M%S)"
    log_file="${PROJECT_ROOT}/formal_training_gpu1_${ts}.log"

    local -a cmd=(
        "${CONDA_EXE}" run -n sim --no-capture-output env
        "CUDA_VISIBLE_DEVICES=${GPU_PHYSICAL}"
        "PYTHONUNBUFFERED=1"
        python3 scripts/train.py
        headless=true
        wandb.mode=disabled
        task=HideAndSeek
        task.use_eval=1
        task.env.num_envs="${NUM_ENVS}"
        total_frames="${TOTAL_FRAMES}"
        eval_interval="${EVAL_INTERVAL}"
        save_interval="${SAVE_INTERVAL}"
        task.sim.device=cuda:0
        task.sim.active_gpu=0
        task.sim.physics_gpu=0
    )

    if [[ -n "${checkpoint}" ]]; then
        cmd+=("model_dir=${checkpoint}")
    fi

    (
        export ISAACSIM_PATH="${ISAACSIM_PATH}"
        cd "${PROJECT_ROOT}"
        exec "${cmd[@]}" 2>&1 | tee -a "${log_file}"
    ) &
    launcher_pid=$!

    set_current_state "${launcher_pid}" "${log_file}" "${checkpoint}"
    log "Launched training: pid=${launcher_pid} checkpoint=${checkpoint:-scratch} log=$(basename "${log_file}")"
}

adopt_existing_training() {
    local pid="${ADOPT_PID:-}"
    local log_file="${ADOPT_LOG:-}"
    local checkpoint="${ADOPT_CKPT:-}"

    if [[ -z "${log_file}" && -f /tmp/current_training_log.txt ]]; then
        log_file="$(cat /tmp/current_training_log.txt || true)"
    fi
    if [[ -n "${log_file}" && "${log_file}" != /* ]]; then
        log_file="${PROJECT_ROOT}/${log_file}"
    fi

    if [[ -z "${pid}" ]]; then
        pid="$(pgrep -o -f "conda run -n sim --no-capture-output env CUDA_VISIBLE_DEVICES=${GPU_PHYSICAL} PYTHONUNBUFFERED=1 python3 scripts/train.py" || true)"
    fi

    if [[ -z "${checkpoint}" ]]; then
        checkpoint="$(find_latest_checkpoint)"
    fi

    if [[ -n "${pid}" && -n "${log_file}" && -f "${log_file}" ]]; then
        set_current_state "${pid}" "${log_file}" "${checkpoint}"
        log "Adopted existing training: pid=${pid} checkpoint=${checkpoint:-unknown} log=$(basename "${log_file}")"
        return 0
    fi

    return 1
}

heartbeat() {
    local pid="$1"
    local log_file="$2"
    local latest_ckpt log_size

    latest_ckpt="$(find_latest_checkpoint)"
    log_size="0"
    [[ -f "${log_file}" ]] && log_size="$(stat -c '%s' "${log_file}" 2>/dev/null || echo 0)"

    log "Heartbeat: pid=${pid} checkpoint=${latest_ckpt:-none} log=$(basename "${log_file}") bytes=${log_size}"
}

main() {
    local loop_count=0

    log "Manual training supervisor started (no auto_train.sh)"

    if ! adopt_existing_training; then
        launch_training "$(find_latest_checkpoint)"
    fi

    while true; do
        local pid log_file latest_ckpt cause
        pid="$(current_pid)"
        log_file="$(current_log)"

        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            loop_count=$((loop_count + 1))
            if (( loop_count % 20 == 0 )); then
                heartbeat "${pid}" "${log_file}"
            fi
            sleep "${CHECK_INTERVAL}"
            continue
        fi

        if log_indicates_completion "${log_file}"; then
            latest_ckpt="$(find_latest_checkpoint)"
            log "Training completed successfully: checkpoint=${latest_ckpt:-none} log=$(basename "${log_file}")"
            break
        fi

        cause="$(summarize_failure "${log_file}")"
        latest_ckpt="$(find_latest_checkpoint)"
        record_recovery "Recovery needed: cause=${cause:-unknown} checkpoint=${latest_ckpt:-none} prior_pid=${pid:-none} prior_log=$(basename "${log_file:-missing}")"
        cleanup_residuals "${pid}"
        sleep "${RESTART_PAUSE}"
        launch_training "${latest_ckpt}"
    done
}

main "$@"
