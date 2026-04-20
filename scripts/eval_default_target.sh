#!/bin/bash
# Test V7 best checkpoint with DEFAULT target params (no speed advantage)
# v_prey=1.0 → target_speed = 1.5 * 1.0 = 1.5 m/s = same as pursuer
set -e
cd "$(dirname "$0")/.."

source /home/uavlab/miniconda3/etc/profile.d/conda.sh
conda activate sim
export PYTHONPATH="${PYTHONPATH:-}"
source setup_conda_env.sh 2>/dev/null || true

CKPT="checkpoints/HideAndSeek_20260419_013651/checkpoint_196804608.pt"
OUT_DIR="analysis/eval_default_target"
mkdir -p "$OUT_DIR"

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
  task.v_drone=1.5"

echo "============================================"
echo "  Phase 1: Batch eval 500 episodes"
echo "  Target: DEFAULT (v_prey=1.0 → speed=1.5)"
echo "  All target params: YAML defaults"
echo "============================================"

python scripts/batch_eval.py $COMMON \
  task.v_prey=1.0 \
  task.env.num_envs=64 \
  seed=42 \
  "+eval_episodes=500" \
  "+out_path=$OUT_DIR/eval_default_500.json" \
  2>&1 | tail -20

echo ""
echo "============================================"
echo "  Phase 2: Record videos (5 seeds)"
echo "============================================"

python scripts/record_rl_videos.py $COMMON \
  task.v_prey=1.0 \
  task.env.num_envs=1 \
  seed=42 \
  "+video_seeds=[42,100,200,300,500]" \
  "+video_dir=$OUT_DIR/videos" \
  "+video_fps=24" "+video_frame_skip=4" \
  2>&1 | tail -20

echo ""
echo "============================================"
echo "  Done! Results in: $OUT_DIR"
echo "============================================"
python3 -c "
import json
try:
    d=json.load(open('$OUT_DIR/eval_default_500.json'))
    print(f'Success: {d[\"success_rate\"]*100:.1f}%')
    print(f'Goal reached: {d.get(\"goal_reached_rate\",0)*100:.1f}%')
    print(f'Out of arena: {d.get(\"out_of_arena_rate\",0)*100:.1f}%')
    print(f'Timeout: {d.get(\"timeout_rate\",0)*100:.1f}%')
except: pass
"
