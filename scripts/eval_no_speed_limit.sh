#!/bin/bash
# Test V7 checkpoint with NO pursuer speed limit
# v_drone=100 (pursuer PhysX limit = 100, effectively unlimited)
# v_prey=0.015 (target speed = 100*0.015 = 1.5, same as original)
set -e
cd "$(dirname "$0")/.."

source /home/uavlab/miniconda3/etc/profile.d/conda.sh
conda activate sim
export PYTHONPATH="${PYTHONPATH:-}"
source setup_conda_env.sh 2>/dev/null || true

CKPT="checkpoints/HideAndSeek_20260419_013651/checkpoint_196804608.pt"
OUT_DIR="analysis/eval_no_speed_limit"
mkdir -p "$OUT_DIR"

COMMON_ARGS="task=HideAndSeek headless=true wandb.mode=disabled \
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
  task.v_drone=100.0 \
  task.v_prey=0.015"

echo "============================================"
echo "  Test: NO pursuer speed limit"
echo "  v_drone=100 (PhysX limit=100, effectively unlimited)"
echo "  v_prey=0.015 (target speed=100*0.015=1.5, unchanged)"
echo "  Checkpoint: $CKPT"
echo "============================================"

echo ""
echo "--- Phase 1: Batch eval 500 episodes ---"
python scripts/batch_eval.py $COMMON_ARGS \
  task.env.num_envs=64 \
  seed=42 \
  "+eval_episodes=500" \
  "+out_path=$OUT_DIR/eval_no_limit_500.json" \
  2>&1 | tee "$OUT_DIR/eval_log.txt" | tail -20

echo ""
echo "--- Phase 2: Record 5 videos ---"
python scripts/record_rl_videos.py $COMMON_ARGS \
  task.env.num_envs=1 \
  seed=42 \
  "+video_seeds=[42,100,200,300,500]" \
  "+video_dir=$OUT_DIR/videos" \
  "+video_fps=24" "+video_frame_skip=4" \
  2>&1 | tail -10

echo ""
echo "--- Results ---"
python3 -c "
import json, os
f = '$OUT_DIR/eval_no_limit_500.json'
if os.path.exists(f):
    d=json.load(open(f))
    print(f'  Success rate: {d[\"success_rate\"]*100:.1f}%')
    print(f'  Goal reached: {d.get(\"goal_reached_rate\",0)*100:.1f}%')
    print(f'  Out of arena: {d.get(\"out_of_arena_rate\",0)*100:.1f}%')
    print(f'  Timeout:      {d.get(\"timeout_rate\",0)*100:.1f}%')
    print(f'  Collision:    {d.get(\"collision_rate\",0)*100:.1f}%')
    print(f'  Total eps:    {d.get(\"total_episodes\",\"?\")}')
else:
    print('  ERROR: result file not found')
"
echo "Done! All results in: $OUT_DIR"
