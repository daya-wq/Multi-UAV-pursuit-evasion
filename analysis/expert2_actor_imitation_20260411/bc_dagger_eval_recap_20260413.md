# BC + DAgger Actor 评估结果回顾

整理时间：2026-04-13

## 主线权重

- BC 最终权重：`checkpoints/expert2_antcoll_bc_1024x50_20260411_152846/bc_final.pt`
- DAgger 最佳权重：`checkpoints/expert2_antcoll_dagger_1024x50_resume3_20260411_161648/dagger_best.pt`
- DAgger 最终权重：`checkpoints/expert2_antcoll_dagger_1024x50_resume3_20260411_161648/dagger_final.pt`
- DAgger 使用 TP：`checkpoints/HideAndSeek_20260403_001241/tp_only_1690959872.pt`

## BC 阶段

- 专家数据集：`expert_datasets/expert2_antcoll_1024x50_20260411_143000`
- 成功轨迹：42,776 episodes
- 成功步数：15,804,748 steps
- 速度分布：`v_prey=1.20/1.35/1.50/1.65`，追捕方 `v_drone=1.50`
- BC 训练：8 epochs，`action_mse_coef=1.0`，`aux_waypoint_coef=0.02`，`aux_assignment_coef=0.05`
- 最终训练 MSE：0.002652
- 没有找到当时单独对 `bc_final.pt` 做纯 actor rollout 的评估日志。

## DAgger 训练条件

- 初始模型：`bc_final.pt`
- 策略：`expert2`，`goal_mode=false`，`close_mode=true`，`rush_mode=false`，`front_layout=symmetric`
- 并行环境：1024
- 每局最大步数：1000
- 目标速度 schedule：`1.2, 1.35, 1.5, 1.65`
- 追捕方速度：`1.5`
- 专家混合概率：从 0.50 线性降到 0.10
- 只保存在线 DAgger 中成功的专家标注轨迹：`online_success_only=True`

## 纯 actor 评估快照

这些是 `DAgger eval @wave X`，不是在线混合采样。评估时 actor 使用 deterministic 行为，且不再混入专家动作。

| 权重 | 目标速度 | 捕获率 | 目标进守区 | 坠机 | 超时 | 平均捕获步 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dagger_wave_010.pt` | 1.35 | 43.1% | 41.4% | 0.0% | 15.5% | 236.3 |
| `dagger_wave_020.pt` | 1.65 | 46.9% | 33.4% | 0.0% | 19.7% | 213.8 |
| `dagger_wave_030.pt` | 1.35 | 52.1% | 35.4% | 0.0% | 12.5% | 272.7 |
| `dagger_wave_040.pt` | 1.65 | 46.9% | 34.6% | 0.0% | 18.6% | 238.6 |
| `dagger_wave_050.pt` / `dagger_best.pt` | 1.35 | 53.1% | 36.6% | 0.0% | 10.3% | 314.0 |

说明：这条 DAgger 脚本设置了 `--n_eval=1024` 和 `--eval_batch_envs=1024`，但代码没有为 eval 单独重建环境，所以实际并行环境沿用训练时的 `batch_envs=1024`。`DAgger eval @wave X` 会继承刚完成 wave 的目标速度，例如 wave 50 的评估速度是 `v_prey=1.35`。

## 后续随机初始化复评

`dagger_best.pt` 后来单独做过一次 actor-only 随机初始化评估：

- `n_eval=1024`
- `batch_envs=1024`
- `episode_length=1000`
- `v_prey=1.5`
- `v_drone=1.5`
- `random_init=true`
- `deterministic=true`
- 不做梯度更新，rollout 在 `torch.no_grad()` 下执行

结果：

- 捕获：62%（640/1024）
- 目标进守区：12%（120/1024）
- 坠机：0%（0/1024）
- 超时：26%（264/1024）
- 平均首次抓捕步：289.8

## 更难距离复评

2026-04-12 又把初始距离固定到 `[2.5, 3.5]m`、速度双方都为 1.5、episode length 1200：

| 组合 | 捕获率 | 目标进守区 | 坠机 | 超时 | 平均首次抓捕步 | TP 预测误差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dagger_best` actor + 自带 TP | 48.9%（501/1024） | 48.1%（493/1024） | 0.0% | 2.9%（30/1024） | 242.9 | 0.2538 |
| `dagger_best` actor + RL TP `checkpoint_113115136.pt` | 45.6%（467/1024） | 49.8%（510/1024） | 0.0% | 4.6%（47/1024） | 227.3 | 0.0576 |

结论：之前 60%+ 的结果对应较容易/默认随机初始化条件；当初始距离固定到 `[2.5, 3.5]m` 正面对抗后，抓捕率降到约 46%-49%。
