#!/bin/bash
# Batch eval: 4 independent Isaac Sim launches × 128 envs = 512 total
set -e
cd "$(dirname "$0")/.."

source /home/uavlab/miniconda3/etc/profile.d/conda.sh
conda activate sim
export PYTHONPATH="${PYTHONPATH:-}"
source setup_conda_env.sh 2>/dev/null || true

CKPT="checkpoints/HideAndSeek_20260419_013651/checkpoint_196804608.pt"
LOG_DIR="analysis/batch_eval"
mkdir -p "$LOG_DIR"

for i in 1 2 3 4; do
  SEED=$((41 + i))
  echo ""
  echo "========================================================"
  echo "  Round $i/4: seed=$SEED (128 envs)"
  echo "========================================================"
  
  # Each round is a separate process to avoid Isaac Sim re-init crash
  python scripts/batch_eval.py \
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
    task.env.num_envs=128 \
    task.v_prey=1.5 \
    task.v_drone=1.5 \
    task.target_repulsion_coef=3.0 \
    seed=$SEED \
    "+eval_label=repulsion_3x_r${i}" \
    2>&1 | grep -vE "TF_PYTHON|pxr/base" || true
    
  echo "  Round $i exited. Waiting 5s for cleanup..."
  sleep 5
done

echo ""
echo "========================================================"
echo "  All rounds complete! Aggregating..."
echo "========================================================"

# Aggregate results
python3 -c "
import json, glob, os
files = sorted(glob.glob('$LOG_DIR/eval_repulsion_3x_r*.json'))
if not files:
    print('  No result files found!')
    exit(1)
total_success = 0
total_goal = 0
total_ooa = 0
total_timeout = 0
total_envs = 0
all_steps = []
for f in files:
    d = json.load(open(f))
    n = d['num_envs']
    r = d['results']
    total_envs += n
    total_success += int(r['success_rate'] * n)
    total_goal += int(r['goal_reached_rate'] * n)
    total_ooa += int(r['out_of_arena_rate'] * n)
    total_timeout += int(r['timeout_rate'] * n)
    all_steps.append(r['avg_steps'])
    print(f'  {os.path.basename(f)}: success={r[\"success_rate\"]:.1%} ({int(r[\"success_rate\"]*n)}/{n})')

print()
print(f'  AGGREGATE ({total_envs} episodes):')
print(f'    Success:      {total_success/total_envs:.1%} ({total_success}/{total_envs})')
print(f'    Goal reached: {total_goal/total_envs:.1%} ({total_goal}/{total_envs})')
print(f'    Out of arena: {total_ooa/total_envs:.1%} ({total_ooa}/{total_envs})')
print(f'    Timeout:      {total_timeout/total_envs:.1%} ({total_timeout}/{total_envs})')
print(f'    Avg steps:    {sum(all_steps)/len(all_steps):.0f}')
"
