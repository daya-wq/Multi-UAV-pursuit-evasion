#!/bin/bash
# Test: tilt50 + repulsion1.5 — 512 eval + 5 videos + dynamics
set -e
cd "$(dirname "$0")/.."

source /home/uavlab/miniconda3/etc/profile.d/conda.sh
conda activate sim
export PYTHONPATH="${PYTHONPATH:-}"
source setup_conda_env.sh 2>/dev/null || true

CKPT="checkpoints/HideAndSeek_20260419_013651/checkpoint_196804608.pt"
LOG_DIR="analysis/batch_eval"
VIDEO_DIR="analysis/rl_v7_tilt50_videos"
DYN_DIR="analysis/dynamics"
mkdir -p "$LOG_DIR" "$VIDEO_DIR" "$DYN_DIR"

LABEL="tilt50_rep1.5"

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
  task.v_prey=1.5 \
  task.v_drone=1.5 \
  task.target_repulsion_coef=1.5 \
  task.target_uav_max_tilt_deg=50.0"

echo "============================================"
echo "  PHASE 1: 512-episode eval (4x128)"
echo "  target_repulsion_coef=1.5"
echo "  target_uav_max_tilt_deg=50"
echo "  (accel_limit=2.0, damping=0.25 = default)"
echo "============================================"

for i in 1 2 3 4; do
  SEED=$((41 + i))
  echo ""
  echo "--- Round $i/4: seed=$SEED ---"
  python scripts/batch_eval.py $COMMON \
    task.env.num_envs=128 seed=$SEED \
    "+eval_label=${LABEL}_r${i}" \
    2>&1 | grep -vE "TF_PYTHON|pxr/base" || true
  echo "  Round $i done."
  sleep 5
done

echo ""
echo "============================================"
echo "  PHASE 1 RESULTS"
echo "============================================"
python3 -c "
import json,glob,os
files=sorted(glob.glob('$LOG_DIR/eval_${LABEL}_r*.json'))
total_s=total_g=total_o=total_t=total_n=0
for f in files:
    d=json.load(open(f));n=d['num_envs'];r=d['results']
    total_n+=n;total_s+=int(r['success_rate']*n);total_g+=int(r['goal_reached_rate']*n)
    total_o+=int(r['out_of_arena_rate']*n);total_t+=int(r['timeout_rate']*n)
    print(f'  {os.path.basename(f)}: success={r[\"success_rate\"]:.1%}')
print(f'\n  AGGREGATE ({total_n}):')
print(f'    Success:      {total_s/total_n:.1%} ({total_s}/{total_n})')
print(f'    Goal reached: {total_g/total_n:.1%}')
print(f'    Out of arena: {total_o/total_n:.1%}')
print(f'    Timeout:      {total_t/total_n:.1%}')
"

echo ""
echo "============================================"
echo "  PHASE 2: Recording 5 videos"
echo "============================================"
python scripts/record_rl_videos.py $COMMON \
  task.env.num_envs=1 seed=42 \
  "+video_seeds=[42,100,200,400,789]" \
  "+video_dir=$VIDEO_DIR" \
  2>&1 | grep -E "Video|saved|SUCCESS|OUT_OF_ARENA|GOAL_ZONE"

echo ""
echo "============================================"
echo "  PHASE 3: Dynamics analysis (seed=42)"
echo "============================================"
# Find a SUCCESS seed from the videos
python scripts/dynamics_analysis.py $COMMON \
  task.env.num_envs=1 seed=42 \
  "+out_dir=$DYN_DIR" \
  2>&1 | grep -vE "TF_PYTHON|pxr/base|Warning.*omni|Warning.*carb|ext:" || true

echo ""
echo "============================================"
echo "  ALL DONE!"
echo "============================================"
ls -lah "$VIDEO_DIR"/*.mp4 2>/dev/null
