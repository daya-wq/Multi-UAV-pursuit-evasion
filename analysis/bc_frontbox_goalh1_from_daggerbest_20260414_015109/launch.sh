#!/usr/bin/env bash
set -euo pipefail

cd /data/uavlab/multi-uav-pursuit2

export TIMESTAMP=20260414_015109
export RUN_COLLECT=true
export RUN_BC=true
export RUN_DAGGER=false

export PRED_MODE=tp_net
export STRATEGY_VARIANT=expert2
export ENABLE_GOAL_MODE=false
export ENABLE_CLOSE_MODE=true
export ENABLE_RUSH_MODE=false
export EXPERT2_FRONT_LAYOUT=symmetric

export GENERIC_BATCH_ENVS=1024
export NUM_WAVES=50
export GENERIC_SEED_BASE=2026041401
export V_DRONE_TEST=1.5
export V_PREY_TEST=1.5
export V_PREY_SCHEDULE=1.5
export EPISODE_LENGTH=1200
export EVAL_GPU=0

export DATASET_NAME=expert2_frontbox_goalh1_v15_1024x50_${TIMESTAMP}
export MIN_SUCCESS_STEPS=25
export DATASET_DTYPE=float16

export BC_SAVE_TAG=expert2_frontbox_goalh1_bc_from_daggerbest_1024x50
export BC_MODEL_DIR=checkpoints/expert2_antcoll_dagger_1024x50_resume3_20260411_161648/dagger_best.pt
export BC_EPOCHS=6
export BC_BATCH_SIZE=4096
export BC_LR=0.0001
export BC_ACTION_MSE_COEF=1.0
export BC_LOG_PROB_COEF=0.0
export BC_AUX_WAYPOINT_COEF=0.02
export BC_AUX_ASSIGNMENT_COEF=0.05
export BC_FRONT_WEIGHT_ALPHA=0.5
export BC_KEEP_PREFIX_STEPS=-1
export BC_KEEP_SUFFIX_STEPS=64

bash scripts/run_expert2_actor_imitation_pipeline.sh
