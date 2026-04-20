#!/bin/bash
# RL warmstart v9: from V7 best checkpoint (85.4% eval success @ 206M frames)
#
# ROOT CAUSE OF V8 FAILURE:
#   clip_param was 0.02 → RL reward signal suppressed 8x
#   → policy_loss was only 20% of gradient, expert_kl dominated 80%
#   → policy couldn't learn from collision penalties, floor collisions increased monotonically
#
# FIX (3 priorities):
#   Priority 1: clip_param restored to 0.1 (critical! enables RL learning)
#   Priority 2: entropy_coef=0.001, expert_kl_coef=0.13 (match V7 best-point balance)
#   Priority 3: optimizer state now saved/loaded in code (no command-line change needed)
#
# Config matches V7 at its BEST POINT (eval #21, 206M frames, kl_coef≈0.128):
#   - actor_lr: 1.5e-5 (same as V7)
#   - clip_param: 0.1 (same as V7)
#   - entropy_coef: 0.001 (same as V7)
#   - expert_kl_coef: 0.13 FIXED (frozen at V7's best-point value, no decay)
#   - log_std_min: -2.0 (same as V7, prevents std collapse)
#   - train_every: 64 (same as V7)
#   - num_envs: 3072 (same as V7)
#   - v_prey: 1.5, v_drone: 1.5 (same as V7)
#
# Checkpoint: V7 best → checkpoint_196804608.pt
# Batch: 3072 envs × 64 steps = 196,608 frames/iter

set -euo pipefail
cd "$(dirname "$0")/.."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_v9_fixed_${TIMESTAMP}.log"

echo "============================================"
echo "  V9 Fixed Training"
echo "  Fixes: clip_param=0.1, entropy=0.001,"
echo "         expert_kl=0.13 frozen, optimizer state"
echo "  Checkpoint: V7 best (85.4% success)"
echo "============================================"

nohup bash -c '
source /home/uavlab/miniconda3/etc/profile.d/conda.sh
conda activate sim
source setup_conda_env.sh

python scripts/train.py \
  task=HideAndSeek \
  headless=true \
  seed=42 \
  model_dir=checkpoints/HideAndSeek_20260419_013651/checkpoint_196804608.pt \
  wandb.mode=disabled \
  wandb.run_name=v9_fixed_clip0.1_kl0.13 \
  task.env.num_envs=3072 \
  task.v_drone=1.5 \
  task.v_prey=1.5 \
  algo.use_TP_net=1 \
  algo.train_every=64 \
  algo.clip_param=0.1 \
  algo.entropy_coef=0.001 \
  algo.actor.lr=1.5e-5 \
  algo.actor.lr_scheduler_kwargs.eta_min=0.000001 \
  algo.actor.log_std_init=-2.0 \
  algo.actor.log_std_min=-2.0 \
  algo.actor.log_std_max=-1.0 \
  algo.actor.bc_aux.enabled=true \
  algo.actor.bc_aux.hidden_dim=256 \
  algo.actor.bc_aux.condition_action_on_aux=true \
  algo.actor.bc_aux.condition_hidden_dim=256 \
  algo.warmstart.actor_freeze.enabled=false \
  algo.warmstart.actor_lr_warmup.enabled=false \
  algo.warmstart.actor_log_std_schedule.enabled=false \
  algo.warmstart.expert_kl.enabled=true \
  algo.warmstart.expert_kl.coef_init=0.13 \
  algo.warmstart.expert_kl.coef_final=0.13 \
  algo.warmstart.expert_kl.decay_frames=0 \
  algo.warmstart.expert_kl.decay_after_eval_success.enabled=false \
  algo.warmstart.expert_kl.mse_boost_enabled=false \
  algo.warmstart.kl_reg.enabled=true \
  algo.warmstart.kl_reg.target_kl=0.03 \
  total_frames=500000000 \
  max_iters=20000 \
  eval_interval=50 \
  save_interval=200 \
  seed=42
' > "$LOG_FILE" 2>&1 &

PID=$!
echo "V9 launched! PID=$PID"
echo "Log: $LOG_FILE"
echo "Tail: tail -f $LOG_FILE"
echo "$PID" > "$LOG_DIR/v9_pid.txt"
