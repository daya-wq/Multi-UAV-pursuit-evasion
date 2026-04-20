#!/bin/bash
# RL warmstart v6: from v5 best checkpoint
#
# Key changes from v5:
#   1. Actor LR: 4e-6 → 2e-4 (50x increase, approx_kl was only 0.0008)
#   2. role_reward_coef: 0.30 → 1.0 (stronger interception shaping)
#   3. train_every: 64 → 256 (longer rollouts, better GAE estimates)
#   4. Continue from v5 checkpoint ~393M (eval success ~40%)
#
# Batch: 1024 envs × 256 steps = 262,144 frames/iter

set -euo pipefail
cd "$(dirname "$0")/.."

LOG_FILE="analysis/rl_warmstart_v6.log"

nohup bash -c '
source /home/uavlab/miniconda3/etc/profile.d/conda.sh
conda activate sim
source setup_conda_env.sh

python scripts/train.py \
  task=HideAndSeek \
  headless=true \
  wandb.mode=disabled \
  model_dir=checkpoints/HideAndSeek_20260418_123612/checkpoint_393412608.pt \
  algo.use_TP_net=1 \
  algo.warmstart.actor_freeze.enabled=false \
  algo.warmstart.actor_lr_warmup.enabled=false \
  algo.actor.lr=2e-4 \
  algo.actor.log_std_init=-2.0 \
  algo.actor.log_std_min=-3.5 \
  algo.actor.log_std_max=-1.0 \
  algo.warmstart.actor_log_std_schedule.enabled=true \
  algo.warmstart.actor_log_std_schedule.adaptive_release.enabled=true \
  algo.warmstart.actor_log_std_schedule.adaptive_release.success_threshold=0.70 \
  algo.warmstart.actor_log_std_schedule.adaptive_release.mse_threshold=1.0 \
  algo.warmstart.actor_log_std_schedule.adaptive_release.consecutive_evals=3 \
  algo.warmstart.actor_log_std_schedule.adaptive_release.min_frames=50000000 \
  algo.warmstart.actor_log_std_schedule.adaptive_release.init_std=0.1353 \
  algo.warmstart.actor_log_std_schedule.adaptive_release.std_increment=-0.003 \
  algo.warmstart.actor_log_std_schedule.adaptive_release.step_frames=20000000 \
  algo.warmstart.actor_log_std_schedule.adaptive_release.max_std=0.20 \
  algo.warmstart.actor_log_std_schedule.frames="[0]" \
  algo.warmstart.actor_log_std_schedule.values="[-2.0]" \
  algo.warmstart.expert_kl.enabled=true \
  algo.warmstart.expert_kl.coef_init=0.3 \
  algo.warmstart.expert_kl.coef_final=0.05 \
  algo.warmstart.expert_kl.decay_frames=300000000 \
  algo.warmstart.expert_kl.mse_boost_enabled=false \
  algo.entropy_coef=0.001 \
  algo.clip_param=0.1 \
  algo.train_every=256 \
  task.env.num_envs=1024 \
  task.v_prey=1.5 \
  task.v_drone=1.5 \
  task.expert2_role_reward_coef=1.0 \
  task.expert2_close_reward_coef=1.0 \
  total_frames=500000000 \
  max_iters=20000 \
  eval_interval=50 \
  save_interval=200 \
  seed=42
' > "$LOG_FILE" 2>&1 &

echo "v6 launched! PID=$!"
echo "Log: $LOG_FILE"
echo "Tail: tail -f $LOG_FILE"
