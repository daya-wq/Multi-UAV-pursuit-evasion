#!/bin/bash
# Test "smart target" parameters: moderate evasion, still catchable
# target_repulsion_coef=1.5, accel_limit=3.5, max_tilt=40, damping=0.15
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
  echo "  Round $i/4: seed=$SEED (128 envs) — Smart Target"
  echo "========================================================"
  
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
    task.target_repulsion_coef=1.5 \
    task.target_accel_limit=3.5 \
    task.target_uav_max_tilt_deg=40.0 \
    task.target_velocity_damping=0.15 \
    seed=$SEED \
    "+eval_label=smart_target_r${i}" \
    2>&1 | grep -vE "TF_PYTHON|pxr/base" || true
    
  echo "  Round $i done."
  sleep 5
done

echo ""
echo "========================================================"
echo "  All rounds complete!"
echo "========================================================"

python3 -c "
import json, glob, os
files = sorted(glob.glob('$LOG_DIR/eval_smart_target_r*.json'))
if not files:
    print('  No result files found!')
    exit(1)
total_s = total_g = total_o = total_t = total_n = 0
for f in files:
    d = json.load(open(f))
    n = d['num_envs']
    r = d['results']
    total_n += n
    total_s += int(r['success_rate'] * n)
    total_g += int(r['goal_reached_rate'] * n)
    total_o += int(r['out_of_arena_rate'] * n)
    total_t += int(r['timeout_rate'] * n)
    print(f'  {os.path.basename(f)}: success={r[\"success_rate\"]:.1%}')
print()
print(f'  AGGREGATE ({total_n} episodes):')
print(f'    Success:      {total_s/total_n:.1%} ({total_s}/{total_n})')
print(f'    Goal reached: {total_g/total_n:.1%}')
print(f'    Out of arena: {total_o/total_n:.1%}')
print(f'    Timeout:      {total_t/total_n:.1%}')
"
