#!/usr/bin/env bash
set -euo pipefail

cd /data/uavlab/multi-uav-pursuit2

export RUN_COLLECT=false
export RUN_BC=true
export RUN_DAGGER=true
export START_FROM_SCRATCH=false

export TIMESTAMP=20260415_1346
export DATASET_ROOT=/data/uavlab/multi-uav-pursuit2/expert_datasets
export DATASET_NAME=expert2_step5_aligned_1024x20_20260415_034910

export PRED_MODE=tp_net
export IMITATION_USE_TP_NET=true
export BC_TP_MODEL_DIR=/data/uavlab/multi-uav-pursuit2/checkpoints/tp_supervised_20260415_005910/tp_only_20260415_010548.pt
export DAGGER_TP_WEIGHT="${BC_TP_MODEL_DIR}"

export BC_MODEL_DIR=/data/uavlab/multi-uav-pursuit2/checkpoints/expert2_step5_aligned_bc_resume_epoch3_20260415_130535_20260415_130613/bc_epoch_001.pt
export BC_SAVE_TAG=expert2_step5_lowlr_from_epoch1
export BC_EPOCHS=12
export BC_LR=0.00005
export BC_BATCH_SIZE=4096
export BC_FRONT_WEIGHT_ALPHA=0.5
export BC_KEEP_PREFIX_STEPS=-1
export BC_KEEP_SUFFIX_STEPS=64
export BC_EVAL_EVERY=1
export BC_N_EVAL=1024
export BC_EVAL_BATCH_ENVS=1024
export BC_EVAL_SUCCESS_THRESHOLD=0.60

export V_DRONE_TEST=1.5
export V_PREY_TEST=1.5
export V_PREY_SCHEDULE=1.5
export EPISODE_LENGTH=1200

export STRATEGY_VARIANT=expert2
export ENABLE_GOAL_MODE=false
export ENABLE_CLOSE_MODE=true
export ENABLE_RUSH_MODE=false
export EXPERT_INTERCEPT_PRED_STEP=5
export EXPERT_INTERCEPT_USE_DIRECT_PRED=false

exec bash scripts/run_expert2_actor_imitation_pipeline.sh
