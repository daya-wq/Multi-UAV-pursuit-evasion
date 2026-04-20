# Expert2 Actor Imitation Run Summary

Date: 2026-04-11

## Setup

- Expert strategy: `expert2`
- Modes: `goal_mode=false`, `close_mode=true`, `rush_mode=false`
- Front layout: `symmetric`
- Parallel envs: 1024
- Target speed schedule: `1.2, 1.35, 1.5, 1.65`
- Action label: normalized PIDRate, aligned with actor output shape `(3, 4)`
- Observation fields: actor `agents/observation` plus `prev_action`
- GPU: GPU0, NVIDIA GeForce RTX 4090

## Expert Dataset

- Path: `expert_datasets/expert2_antcoll_1024x50_20260411_143000`
- Size: 6.7G
- Chunks: 50
- Total successful episodes: 42,776
- Total successful steps: 15,804,748

Speed split:

- `v_prey=1.20`: 13 chunks, 12,708 episodes, 5,009,960 steps
- `v_prey=1.35`: 13 chunks, 11,733 episodes, 4,657,052 steps
- `v_prey=1.50`: 12 chunks, 9,594 episodes, 3,386,167 steps
- `v_prey=1.65`: 12 chunks, 8,741 episodes, 2,751,569 steps

## BC Training

- Checkpoint dir: `checkpoints/expert2_antcoll_bc_1024x50_20260411_152846`
- Final checkpoint: `checkpoints/expert2_antcoll_bc_1024x50_20260411_152846/bc_final.pt`
- Epochs: 8
- Final train MSE: 0.002652

## DAgger Training

- Online dataset path: `expert_datasets/dagger_expert2_antcoll_1024x50_20260411_143000`
- Online dataset size: 3.6G
- Online chunks: 50
- Total successful episodes: 28,933
- Total successful steps: 8,467,590

Speed split:

- `v_prey=1.20`: 13 chunks, 8,629 episodes, 2,826,367 steps
- `v_prey=1.35`: 13 chunks, 7,565 episodes, 2,374,802 steps
- `v_prey=1.50`: 12 chunks, 6,353 episodes, 1,665,006 steps
- `v_prey=1.65`: 12 chunks, 6,386 episodes, 1,601,415 steps

Final DAgger checkpoints:

- Final: `checkpoints/expert2_antcoll_dagger_1024x50_resume3_20260411_161648/dagger_final.pt`
- Best eval: `checkpoints/expert2_antcoll_dagger_1024x50_resume3_20260411_161648/dagger_best.pt`
- Best eval capture rate: 53.1%

Evaluation snapshots:

- Wave 10: capture 43.1%, goal 41.4%, landed 0.0%, timeout 15.5%
- Wave 20: capture 46.9%, goal 33.4%, landed 0.0%, timeout 19.7%
- Wave 30: capture 52.1%, goal 35.4%, landed 0.0%, timeout 12.5%
- Wave 40: capture 46.9%, goal 34.6%, landed 0.0%, timeout 18.6%
- Wave 50: capture 53.1%, goal 36.6%, landed 0.0%, timeout 10.3%

## Notes

- The initial DAgger run OOMed because rollout was building autograd graphs through actor/env stepping. `torch.no_grad()` was added around actor rollout and env stepping, and online BC buffer flushing was chunked.
- After the fix, GPU0 memory stayed around 7.8-8.0GB during the 1024-env resume run.

## Actor-Only Random Evaluation

Checkpoint: `checkpoints/expert2_antcoll_dagger_1024x50_resume3_20260411_161648/dagger_best.pt`

Config:

- `n_eval=1024`
- `batch_envs=1024`
- `episode_length=1000`
- `v_prey=1.5`
- `v_drone=1.5`
- `random_init=true`
- `deterministic=true`
- No gradient updates; rollout runs under `torch.no_grad()`

Results:

- Capture: 62% (640/1024)
- Goal zone: 12% (120/1024)
- Landed: 0% (0/1024)
- Timeout: 26% (264/1024)
- Capture steps: mean 289.8, std 205.8, min 0, max 981
- Timeout avg: mindist 0.530, dmin -0.055, ahead 2.66, team 0.129, spread 0.089, cap_prog 0.0025
- Capture avg: mindist 0.406, dmin -0.094, ahead 2.85, team 0.147, spread 0.136, cap_prog 0.0380
