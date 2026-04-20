#!/bin/bash
# Combined: dynamics + hi-fps video (original & tilt50) + trajectory plots
set -e
cd "$(dirname "$0")/.."

source /home/uavlab/miniconda3/etc/profile.d/conda.sh
conda activate sim
export PYTHONPATH="${PYTHONPATH:-}"
source setup_conda_env.sh 2>/dev/null || true

CKPT="checkpoints/HideAndSeek_20260419_013651/checkpoint_196804608.pt"

COMMON="task=HideAndSeek headless=true wandb.mode=disabled \
  model_dir=$CKPT \
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
  task.v_drone=1.5"

echo "============================================"
echo "  PHASE 1: Original (default target) dynamics + hi-fps video"
echo "============================================"
python scripts/dynamics_analysis.py $COMMON \
  seed=100 "+out_dir=analysis/dynamics" \
  2>&1 | grep -E "Episode|saved" || true

python scripts/record_rl_videos.py $COMMON \
  seed=100 \
  "+video_seeds=[100]" \
  "+video_dir=analysis/hifps_original" \
  "+video_fps=24" "+video_frame_skip=4" \
  2>&1 | grep -E "saved|SUCCESS|GOAL" || true

echo ""
echo "============================================"
echo "  PHASE 2: Tilt50 (rep1.5, tilt50) dynamics + hi-fps video"  
echo "============================================"
python scripts/dynamics_analysis.py $COMMON \
  task.target_repulsion_coef=1.5 \
  task.target_uav_max_tilt_deg=50.0 \
  seed=42 "+out_dir=analysis/dynamics" \
  2>&1 | grep -E "Episode|saved" || true

python scripts/record_rl_videos.py $COMMON \
  task.target_repulsion_coef=1.5 \
  task.target_uav_max_tilt_deg=50.0 \
  seed=42 \
  "+video_seeds=[42]" \
  "+video_dir=analysis/hifps_tilt50" \
  "+video_fps=24" "+video_frame_skip=4" \
  2>&1 | grep -E "saved|SUCCESS|GOAL" || true

echo ""  
echo "============================================"
echo "  PHASE 3: Smart target dynamics + hi-fps video"
echo "============================================"
python scripts/dynamics_analysis.py $COMMON \
  task.target_repulsion_coef=1.5 \
  task.target_accel_limit=3.5 \
  task.target_uav_max_tilt_deg=40.0 \
  task.target_velocity_damping=0.15 \
  seed=42 "+out_dir=analysis/dynamics" \
  2>&1 | grep -E "Episode|saved" || true

# Rename to distinguish
if [ -f analysis/dynamics/dynamics_seed42.json ]; then
  cp analysis/dynamics/dynamics_seed42.json analysis/dynamics/dynamics_smart_seed42.json
fi

python scripts/record_rl_videos.py $COMMON \
  task.target_repulsion_coef=1.5 \
  task.target_accel_limit=3.5 \
  task.target_uav_max_tilt_deg=40.0 \
  task.target_velocity_damping=0.15 \
  seed=42 \
  "+video_seeds=[42]" \
  "+video_dir=analysis/hifps_smart" \
  "+video_fps=24" "+video_frame_skip=4" \
  2>&1 | grep -E "saved|SUCCESS|GOAL" || true

echo ""
echo "============================================"
echo "  PHASE 4: Draw trajectory plots"
echo "============================================"
python scripts/plot_trajectories.py analysis/trajectory_plots

echo ""
echo "============================================"
echo "  ALL DONE!"
echo "============================================"
ls -lah analysis/trajectory_plots/*.png 2>/dev/null
ls -lah analysis/hifps_*/*.mp4 2>/dev/null
