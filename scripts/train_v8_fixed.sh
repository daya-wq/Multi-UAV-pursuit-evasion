#!/bin/bash
# V8 Fixed: Continue from V7 best checkpoint (196804608)
# FIXES vs previous attempt:
#   1. log_std_min restored to -4.605170 (was wrongly set to -2.0)
#   2. mse_boost disabled (was boosting expert_kl from 0.3 to 2.0)
#   3. log_std schedule disabled but log_std_init NOT forced
# Changed from V7:
#   - actor_lr: 1e-5 (was 1.5e-5)
#   - expert_kl_coef: frozen at 0.3 (was decaying 0.3→0.05)
set -e
cd "$(dirname "$0")/.."

source /home/uavlab/miniconda3/etc/profile.d/conda.sh
conda activate sim
export PYTHONPATH="${PYTHONPATH:-}"
source setup_conda_env.sh 2>/dev/null || true

CKPT="checkpoints/HideAndSeek_20260419_013651/checkpoint_196804608.pt"
RUN_NAME="v8_fixed_kl0.3_lr1e-5"

echo "============================================"
echo "  V8 Fixed Training"
echo "  Fixes: log_std_min restored, mse_boost off"
echo "  Checkpoint: $CKPT"
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
  task.v_drone=1.5 \
  task.v_prey=1.0 \
  \
  algo.use_TP_net=1 \
  algo.train_every=64 \
  algo.clip_param=0.02 \
  \
  algo.actor.lr=0.00001 \
  algo.actor.lr_scheduler_kwargs.eta_min=0.000001 \
  algo.actor.log_std_min=-4.605170 \
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
  algo.warmstart.expert_kl.decay_after_eval_success.enabled=false \
  algo.warmstart.expert_kl.mse_boost_enabled=false \
  algo.warmstart.kl_reg.enabled=true \
  algo.warmstart.kl_reg.target_kl=0.03 \
  \
  total_frames=2000000000 \
  save_interval=100 \
  2>&1 | tee "logs/train_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
