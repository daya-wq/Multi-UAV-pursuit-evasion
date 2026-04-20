# RL Stage2 曲线复盘

- 日志: `analysis/rl_stage2_from54132736_20260412/train_20260412_182837.log`
- train 记录数: 1734
- eval 记录数: 70
- 全程最高 eval success: 68.26% @ 0.07M frames
- 解冻后最高 eval success: 55.18% @ 6.62M frames
- 最新 eval success: 3.32% @ 113.12M frames, goal_reached=96.58%
- 最新 train success: 0.00% @ 114.00M frames, goal_reached=0.00%

## 曲线

![00_success_train_eval](00_success_train_eval.png)

![01_eval_outcome](01_eval_outcome.png)

![02_eval_collisions](02_eval_collisions.png)

![03_train_outcome](03_train_outcome.png)

![04_train_rewards](04_train_rewards.png)

![05_train_return_failure](05_train_return_failure.png)

![06_optimizer_losses](06_optimizer_losses.png)

![07_ppo_stability](07_ppo_stability.png)

![08_schedule_expert](08_schedule_expert.png)

![09_entropy_action](09_entropy_action.png)

![10_geometry](10_geometry.png)

