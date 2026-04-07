#!/bin/bash
###############################################################################
# auto_train.sh — 自动重启训练脚本（含挂起检测）
#
# 用法：  nohup bash scripts/auto_train.sh > auto_train_monitor.log 2>&1 &
# 停止：  kill $(cat /data/uavlab/multi-uav-pursuit2/auto_train.pid)
###############################################################################

PROJECT_ROOT="/data/uavlab/multi-uav-pursuit2"
CONDA_BASE="/home/uavlab/miniconda3"

echo $$ > "${PROJECT_ROOT}/auto_train.pid"

# Conda init
source "${CONDA_BASE}/etc/profile.d/conda.sh"
export ISAACSIM_PATH="/data/uavlab/isaac_sim_2022.2.0"
conda activate sim
cd "${PROJECT_ROOT}"

# ======================== 用户可配置区 ========================
TOTAL_FRAMES=2000000000       # 20 亿步
NUM_ENVS=2048                 # 并行环境数
EVAL_INTERVAL=4000            # 评估间隔
SCENARIO_FLAG="empty"         # empty / wall / narrow_gap / passage
MAX_RESTARTS=100              # 最多自动重启次数
COOLDOWN=30                   # 崩溃后等待秒数
HANG_TIMEOUT=600              # 挂起检测超时（秒），日志无输出超过此时间则判定为挂起
HANG_CHECK_INTERVAL=60        # 挂起检测轮询间隔（秒）
# ==============================================================

# 专用权重目录
CKPT_DIR="${PROJECT_ROOT}/checkpoints/auto_train_${SCENARIO_FLAG}"
mkdir -p "${CKPT_DIR}"

TRAIN_LOG="${PROJECT_ROOT}/formal_training.log"

find_latest_checkpoint() {
    local latest=""
    local max_step=0
    for ckpt in "${CKPT_DIR}"/checkpoint_*.pt; do
        [ -f "$ckpt" ] || continue
        local basename
        basename=$(basename "$ckpt")
        local step
        step=$(echo "$basename" | sed 's/checkpoint_//;s/\.pt//')
        if [ -n "$step" ] && [ "$step" -eq "$step" ] 2>/dev/null; then
            if [ "$step" -gt "$max_step" ]; then
                max_step=$step
                latest=$ckpt
            fi
        fi
    done
    echo "$latest"
}

echo "=============================================="
echo " Auto-Restart Training Loop"
echo " 时间: $(date)"
echo " 场景: ${SCENARIO_FLAG} | 环境: ${NUM_ENVS}"
echo " 目标: ${TOTAL_FRAMES} frames"
echo " 权重目录: ${CKPT_DIR}"
echo " 挂起超时: ${HANG_TIMEOUT}s"
echo "=============================================="

count=0
while [ $count -lt $MAX_RESTARTS ]; do
    count=$((count + 1))
    echo ""
    echo "[$(date)] ===== 第 ${count}/${MAX_RESTARTS} 次启动 ====="

    CKPT=$(find_latest_checkpoint)
    MODEL_ARG=""
    if [ -n "$CKPT" ]; then
        echo "[$(date)] 加载权重: ${CKPT}"
        MODEL_ARG="model_dir=${CKPT}"
    else
        echo "[$(date)] 从零开始训练"
    fi

    echo "[$(date)] 启动训练..."
    # 后台启动训练，通过 watchdog 监控挂起
    python3 scripts/train.py \
        headless=true \
        wandb.mode=disabled \
        task=HideAndSeek \
        task.use_eval=1 \
        task.use_random_cylinder=0 \
        task.scenario_flag=${SCENARIO_FLAG} \
        task.env.num_envs=${NUM_ENVS} \
        total_frames=${TOTAL_FRAMES} \
        eval_interval=${EVAL_INTERVAL} \
        ${MODEL_ARG} \
        > "${TRAIN_LOG}" 2>&1 &
    TRAIN_PID=$!
    echo "[$(date)] 训练 PID: ${TRAIN_PID}"

    # ---- Watchdog: 检测日志文件是否持续更新 ----
    HUNG=false
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        sleep "${HANG_CHECK_INTERVAL}"
        # 进程可能在 sleep 期间结束
        kill -0 "$TRAIN_PID" 2>/dev/null || break

        if [ -f "${TRAIN_LOG}" ]; then
            LAST_MOD=$(stat -c %Y "${TRAIN_LOG}" 2>/dev/null || echo 0)
            NOW=$(date +%s)
            SILENT=$((NOW - LAST_MOD))
            if [ "$SILENT" -ge "$HANG_TIMEOUT" ]; then
                echo "[$(date)] ⏰ 日志已 ${SILENT}s 无更新，判定训练挂起！"
                echo "[$(date)] 杀死训练进程 ${TRAIN_PID} ..."
                kill -9 "$TRAIN_PID" 2>/dev/null
                wait "$TRAIN_PID" 2>/dev/null
                HUNG=true
                break
            fi
        fi
    done

    # 等待子进程退出（正常退出或崩溃）
    wait "$TRAIN_PID" 2>/dev/null
    EXIT_CODE=$?
    if $HUNG; then
        echo "[$(date)] 训练进程被 watchdog 杀死 (挂起)"
    else
        echo "[$(date)] 训练进程退出，exit code: ${EXIT_CODE}"
    fi

    # 将本次产生的所有权重文件归档到统一权重目录
    LATEST_RUN_DIR=$(ls -dt "${PROJECT_ROOT}"/checkpoints/HideAndSeek_*/  2>/dev/null | head -n 1)
    if [ -n "$LATEST_RUN_DIR" ] && [ -d "$LATEST_RUN_DIR" ]; then
        for f in "${LATEST_RUN_DIR}"checkpoint_*.pt; do
            [ -f "$f" ] || continue
            cp -n "$f" "${CKPT_DIR}/" 2>/dev/null || true
        done
        echo "[$(date)] 已将权重归档到 ${CKPT_DIR}"
    fi

    if grep -q "Final Eval" "${TRAIN_LOG}" 2>/dev/null; then
        echo "[$(date)] ✅ 训练正常完成！"
        break
    fi

    if $HUNG; then
        echo "[$(date)] ⚠️  训练挂起，${COOLDOWN}秒后自动重启..."
    else
        echo "[$(date)] ⚠️  Isaac Sim 崩溃，${COOLDOWN}秒后自动重启..."
    fi
    sleep ${COOLDOWN}
done

echo "[$(date)] 脚本结束。"
rm -f "${PROJECT_ROOT}/auto_train.pid"
