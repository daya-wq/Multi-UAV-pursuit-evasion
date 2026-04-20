#!/bin/bash
# RL warmstart v7 (corrected): from v5 best checkpoint (~47% eval success)
#
# LESSON LEARNED: v7 original failed because we changed 3 things at once:
#   - LR: 4e-6 → 5e-5 (12.5x)
#   - role_reward_coef: 0.3 → 0.5
#   - train_every: 64 → 128
# The reward coefficient change broke the value_normalizer calibration,
# causing the critic to produce wrong advantages. Combined with the higher LR,
# the policy was destroyed in the first 2 iterations.
#
# FIX: Only change LR (4e-6 → 1.5e-5, modest 3.75x increase).
# Keep ALL other params identical to v5 to preserve critic/value_normalizer alignment.
#
# Differences from v5:
#   1. actor_lr: 4e-6 → 1.5e-5 (3.75x increase, conservative)
#   2. actor_lr_warmup: DISABLED (already trained, no need to warm up)
#   3. log_std: held at -2.0, no adaptive decrease (avoid v4's collapse)
#
# Everything else is IDENTICAL to v5:
#   - train_every: 64
#   - role_reward_coef: 0.30
#   - close_reward_coef: 0.30
#   - clip_param: 0.1
#   - entropy_coef: 0.001
#   - expert_kl: 0.3 → 0.05 over 300M
#   - num_envs: 3072
#
# Batch: 3072 envs × 64 steps = 196,608 frames/iter

set -euo pipefail
cd "$(dirname "$0")/.."

LOG_FILE="analysis/rl_warmstart_v7.log"

nohup bash -c '
source /home/uavlab/miniconda3/etc/profile.d/conda.sh
conda activate sim
source setup_conda_env.sh

python scripts/train.py \
  task=HideAndSeek \
  headless=true \
  wandb.mode=disabled \
  model_dir=checkpoints/HideAndSeek_20260418_123612/checkpoint_275447808.pt \
  algo.use_TP_net=1 \
  algo.warmstart.actor_freeze.enabled=false \
  algo.warmstart.actor_lr_warmup.enabled=false \
  algo.actor.lr=1.5e-5 \
  algo.actor.log_std_init=-2.0 \
  algo.actor.log_std_min=-2.0 \
  algo.actor.log_std_max=-1.0 \
  algo.warmstart.actor_log_std_schedule.enabled=false \
  algo.warmstart.expert_kl.enabled=true \
  algo.warmstart.expert_kl.coef_init=0.3 \
  algo.warmstart.expert_kl.coef_final=0.05 \
  algo.warmstart.expert_kl.decay_frames=300000000 \
  algo.warmstart.expert_kl.mse_boost_enabled=false \
  algo.entropy_coef=0.001 \
  algo.clip_param=0.1 \
  algo.train_every=64 \
  task.env.num_envs=3072 \
  task.v_prey=1.5 \
  task.v_drone=1.5 \
  task.expert2_role_reward_coef=0.30 \
  task.expert2_close_reward_coef=0.30 \
  total_frames=500000000 \
  max_iters=20000 \
  eval_interval=50 \
  save_interval=200 \
  seed=42
' > "$LOG_FILE" 2>&1 &

echo "v7 (corrected) launched! PID=$!"
echo "Log: $LOG_FILE"
echo "Tail: tail -f $LOG_FILE"
