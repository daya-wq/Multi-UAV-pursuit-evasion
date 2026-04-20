#!/bin/bash
# RL warmstart v5: from v3 checkpoint, log_std=-2.0 held until eval success>0.7, expert_kl=0.3
#
# Key differences from v4:
#   1. Loads v3 checkpoint (not v4)
#   2. log_std held FIXED at -2.0 via adaptive_release gate (success_threshold=0.70)
#      - After eval success ≥ 0.70 for 3 consecutive evals, begin decreasing std
#      - Decrease by 0.003 every 20M frames (init_std=0.1353 → floor at log_std_min=-3.5)
#   3. Expert KL coef raised back to 0.3 (was 0.05 in v4)
#   4. LR warmup disabled (continuing from trained checkpoint)

set -euo pipefail
cd "$(dirname "$0")/.."

LOG_FILE="analysis/rl_warmstart_v5.log"

nohup bash -c '
source /home/uavlab/miniconda3/etc/profile.d/conda.sh
conda activate sim
source setup_conda_env.sh

python scripts/train.py \
  task=HideAndSeek \
  headless=true \
  wandb.mode=disabled \
  model_dir=checkpoints/HideAndSeek_20260417_181246/checkpoint_final.pt \
  algo.use_TP_net=1 \
  algo.warmstart.actor_freeze.enabled=false \
  algo.warmstart.actor_lr_warmup.enabled=false \
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
  task.env.num_envs=3072 \
  task.v_prey=1.5 \
  task.v_drone=1.5 \
  total_frames=500000000 \
  max_iters=20000 \
  eval_interval=50 \
  save_interval=200 \
  seed=42
' > "$LOG_FILE" 2>&1 &

echo "v5 launched! PID=$!"
echo "Log: $LOG_FILE"
echo "Tail: tail -f $LOG_FILE"
