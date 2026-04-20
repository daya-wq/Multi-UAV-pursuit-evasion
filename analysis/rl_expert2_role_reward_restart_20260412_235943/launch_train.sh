#!/usr/bin/env bash
set -eo pipefail

source /home/uavlab/miniconda3/etc/profile.d/conda.sh
export PYTHONPATH="${PYTHONPATH:-}"
conda activate sim
cd /data/uavlab/multi-uav-pursuit2

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

python3 scripts/train.py \
  model_dir=checkpoints/expert2_antcoll_dagger_1024x50_resume3_20260411_161648/dagger_best.pt \
  load_actor_tp_only=true \
  task.env.num_envs=1024 \
  task.env.max_episode_length=1200 \
  task.curriculum.start_stage=1 \
  task.curriculum.pursuer_spawn_mode=goal_side_arc \
  task.curriculum.eval_uses_fixed_layout=false \
  total_frames=250000000 \
  eval_interval=25 \
  save_interval=25 \
  wandb.mode=disabled \
  wandb.run_name=expert2_role_reward_restart_fixedstd015 \
  headless=true \
  task.sim.device=cuda:0 \
  task.sim.active_gpu=0 \
  task.sim.physics_gpu=0
