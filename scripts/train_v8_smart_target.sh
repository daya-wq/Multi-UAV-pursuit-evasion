#!/bin/bash
# Continue training from V7 best checkpoint with enhanced target dynamics
# Target changes: rep_coef=1.3, vel_damping=0.15, tilt=35, accel=3.0
set -e
cd "$(dirname "$0")/.."

source /home/uavlab/miniconda3/etc/profile.d/conda.sh
conda activate sim
export PYTHONPATH="${PYTHONPATH:-}"
source setup_conda_env.sh 2>/dev/null || true

CKPT="checkpoints/HideAndSeek_20260419_013651/checkpoint_196804608.pt"
RUN_NAME="v8_smart_target_rep1.3_tilt35"

echo "============================================"
echo "  Continuing training from V7 best checkpoint"
echo "  Checkpoint: $CKPT"
echo "  Run name:   $RUN_NAME"
echo ""
echo "  Target dynamics changes:"
echo "    target_repulsion_coef: 1.0 -> 1.3"
echo "    target_velocity_damping: 0.25 -> 0.15"
echo "    target_uav_max_tilt_deg: 25 -> 35"
echo "    target_accel_limit: 2.0 -> 3.0"
echo "============================================"

python scripts/train.py \
  task=HideAndSeek \
  headless=true \
  seed=42 \
  model_dir="$CKPT" \
  wandb.mode=disabled \
  wandb.run_name="$RUN_NAME" \
  \
  task.env.num_envs=3072 \
  task.v_prey=1.5 \
  task.v_drone=1.5 \
  \
  task.target_repulsion_coef=1.3 \
  task.target_velocity_damping=0.15 \
  task.target_uav_max_tilt_deg=35.0 \
  task.target_accel_limit=3.0 \
  \
  algo.use_TP_net=1 \
  algo.train_every=64 \
  algo.clip_param=0.02 \
  algo.actor.lr=0.000004 \
  algo.actor.log_std_init=-2.0 \
  algo.actor.log_std_min=-2.0 \
  algo.actor.log_std_max=-1.0 \
  algo.actor.bc_aux.enabled=true \
  algo.actor.bc_aux.hidden_dim=256 \
  algo.actor.bc_aux.condition_action_on_aux=true \
  algo.actor.bc_aux.condition_hidden_dim=256 \
  \
  algo.warmstart.actor_freeze.enabled=false \
  algo.warmstart.actor_lr_warmup.enabled=false \
  algo.warmstart.actor_log_std_schedule.enabled=false \
  algo.warmstart.expert_kl.enabled=true \
  algo.warmstart.expert_kl.coef_init=0.3 \
  algo.warmstart.expert_kl.coef_final=0.3 \
  algo.warmstart.expert_kl.decay_frames=0 \
  algo.warmstart.kl_reg.enabled=true \
  algo.warmstart.kl_reg.target_kl=0.03 \
  \
  total_frames=2000000000 \
  save_interval=100 \
  2>&1 | tee "logs/train_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
