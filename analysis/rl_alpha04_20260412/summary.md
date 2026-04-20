# Expert2 Alpha04 RL Run Summary
## Artifacts
- Log: `formal_expert2_alpha04_rl_20260412_011130.log`
- TensorBoard run: `runs/HideAndSeek_20260412_011138`
- Checkpoints: `checkpoints/HideAndSeek_20260412_011138`
- Eval CSV: `analysis/rl_alpha04_20260412/eval_metrics.csv`

## Run Setup
- Start checkpoint: `checkpoints/expert2_antcoll_dagger_1024x50_resume3_20260411_161648/dagger_best.pt`
- Environments: 1024 parallel Isaac environments on GPU0
- Rollout length per PPO update: 64 sim steps
- Frames per update: 65,536
- Completed updates: 3,051
- Final logged frames: 199,950,336
- Max episode length: 1000 steps
- Eval interval: every 25 updates = 1,638,400 frames
- Save interval: every 25 updates = 1,638,400 frames
- Eval points with success metric: 123
- Eval episodes represented: 123 x 1024 = 125,952 vectorized episodes

## Main Overrides
- `reward_profile=expert2_minimal`, `catch_reward_coef=70.0`, `goal_penalty_coef=40.0`, `collision_coef=20.0`, `spread_reward_coef=0.20`, `capture_progress_coef=0.02`, `time_penalty_coef=0.01`
- `landed_z_threshold=0.01`, `target_min_z=0.2`
- `ppo_epochs=1`, `num_minibatches=16`, `actor.lr=1e-6`, `critic.lr=5e-5`, entropy disabled
- Actor freeze: first 5,000,000 frames
- Actor log_std schedule: frames `[0, 5000000, 20000000, 80000000, 140000000]`, values `[-4.5, -4.5, -3.4, -3.2, -4.2]`
- Anchor loss: `2.0 -> 0.5` over 120M frames
- Expert KL alpha: `0.4 -> 0.03` over 120M frames, `loss_type=nll`, expert2 close-only symmetric layout, oracle target position

## Eval Results
- Best eval: step 57,409,536, success 99.61%, checkpoint `checkpoints/HideAndSeek_20260412_011138/checkpoint_57409536.pt`
- Final normal eval: step 199,950,336, success 71.58%, landed 0.00%, first_capture_step 881.7

Top evals by success:

| rank | step | success | landed | first_capture_step | collision | pursuer_collisions_count | target_pred_err | checkpoint |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 57,409,536 | 99.61% | 0.00% | 603.8 | 0.000606 | 0.491 | 0.0349 | `checkpoint_57409536.pt` |
| 2 | 50,855,936 | 99.51% | 0.00% | 681.4 | 0.013902 | 3.815 | 0.0345 | `checkpoint_50855936.pt` |
| 3 | 62,324,736 | 99.51% | 0.00% | 516.1 | 0.020398 | 11.954 | 0.0317 | `checkpoint_62324736.pt` |
| 4 | 167,182,336 | 99.12% | 0.00% | 770.8 | 0.000164 | 0.134 | 0.0166 | `checkpoint_167182336.pt` |
| 5 | 55,771,136 | 98.83% | 0.00% | 847.9 | 0.003813 | 1.229 | 0.1778 | `checkpoint_55771136.pt` |
| 6 | 3,342,336 | 97.07% | 0.00% | 240.9 | 0.022053 | 14.337 | 0.0955 | `checkpoint_3342336.pt` |
| 7 | 13,172,736 | 97.07% | 0.00% | 244.0 | 0.016858 | 12.831 | 0.0601 | `checkpoint_13172736.pt` |
| 8 | 54,132,736 | 97.07% | 0.00% | 575.2 | 0.000173 | 0.170 | 0.0819 | `checkpoint_54132736.pt` |
| 9 | 1,703,936 | 96.88% | 0.00% | 252.2 | 0.024766 | 15.165 | 0.1278 | `checkpoint_1703936.pt` |
| 10 | 14,811,136 | 95.90% | 0.00% | 268.8 | 0.025313 | 18.167 | 0.0626 | `checkpoint_14811136.pt` |
| 11 | 16,449,536 | 95.80% | 0.00% | 247.4 | 0.020341 | 14.577 | 0.0499 | `checkpoint_16449536.pt` |
| 12 | 8,257,536 | 95.12% | 0.00% | 267.2 | 0.022959 | 14.809 | 0.0721 | `checkpoint_8257536.pt` |
| 13 | 11,534,336 | 95.12% | 0.00% | 269.8 | 0.020407 | 15.501 | 0.0730 | `checkpoint_11534336.pt` |
| 14 | 49,217,536 | 95.02% | 0.00% | 523.4 | 0.004093 | 1.246 | 0.0436 | `checkpoint_49217536.pt` |
| 15 | 4,980,736 | 94.43% | 0.00% | 276.2 | 0.031436 | 21.710 | 0.0759 | `checkpoint_4980736.pt` |

## Eval Initial Conditions
- Eval used `task.use_eval=1` and `curriculum.eval_uses_fixed_layout=true`, so eval did **not** use the training curriculum random spawn sampler.
- Base fixed eval target position before noise: `[-1.6, 0.0, goal_z]`, with `goal_z=2.5`.
- Base fixed pursuer positions before noise for 3 agents: `[1.2, 0.0, 2.5]`, `[1.5, 0.6, 2.5]`, `[1.5, -0.6, 2.5]`.
- Eval position noise on every reset: x/y uniform `[-0.15, +0.15]` m, z uniform `[-0.10, +0.10]` m; target z is clamped to `target_min_z=0.2`, pursuer z clamped to `[0.2, max_height-0.1]`.
- Initial drone roll/pitch/yaw in eval is zero; initial drone and target velocities are zero.
- Good evals in the log all report curriculum stats as `stage=1`, `pursuer_speed=0.8`, `target_speed=0.5`.
- Important limitation: the run did not log per-environment sampled initial positions/seeds for each eval episode, so we can reconstruct the distribution and fixed base layout, but not the exact 1024 sampled initial states for a particular good eval without rerunning with reset-state logging enabled.

## Interpretation
- `alpha=0.4` improved early/mid training retention: several evals are above 95%, with the best at 99.61%.
- The final checkpoint is weaker than the best checkpoint: final success is 71.58%, so `checkpoint_final.pt` should not be treated as the best policy.
- Landed stayed at 0.00% in eval, so this run is not failing through the old land issue.
- The remaining issue is late PPO drift after alpha decays and actor fully unfreezes; use `checkpoint_57409536.pt` or validate `checkpoint_167182336.pt` rather than final.
