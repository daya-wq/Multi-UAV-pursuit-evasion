#!/bin/bash
# Record evaluation videos for the v7 best checkpoint (peak 85.4% eval success)
# Single Isaac Sim session, multiple seeds
cd "$(dirname "$0")/.."

source /home/uavlab/miniconda3/etc/profile.d/conda.sh
conda activate sim
export PYTHONPATH="${PYTHONPATH:-}"
source setup_conda_env.sh 2>/dev/null || true

CKPT="checkpoints/HideAndSeek_20260419_013651/checkpoint_196804608.pt"
VIDEO_DIR="analysis/rl_v7_best_videos"
LOG_FILE="analysis/record_v7_best.log"

echo "========================================================"
echo "  Recording v7 BEST checkpoint evaluation videos"
echo "  Checkpoint: $CKPT"
echo "  Output dir: $VIDEO_DIR"
echo "========================================================"

python scripts/record_rl_videos.py \
  task=HideAndSeek \
  headless=true \
  wandb.mode=disabled \
  model_dir="$CKPT" \
  algo.use_TP_net=1 \
  algo.actor.bc_aux.enabled=true \
  algo.actor.bc_aux.hidden_dim=256 \
  algo.actor.bc_aux.condition_action_on_aux=true \
  algo.actor.bc_aux.condition_hidden_dim=256 \
  algo.actor.log_std_init=-2.0 \
  algo.actor.log_std_min=-2.0 \
  algo.actor.log_std_max=-1.0 \
  algo.train_every=64 \
  task.env.num_envs=1 \
  task.v_prey=1.5 \
  task.v_drone=1.5 \
  seed=42 \
  "+video_seeds=[42,100,200,300,400,500,789,1024,2026,3000]" \
  "+video_dir=$VIDEO_DIR" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "Done! Videos in: $VIDEO_DIR"
ls -lah "$VIDEO_DIR"
