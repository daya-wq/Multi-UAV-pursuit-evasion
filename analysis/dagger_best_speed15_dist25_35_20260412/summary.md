# DAgger Best 在 1.5m/s、2.5-3.5m 初始距离下的评估

评估时间：2026-04-12 21:25-21:29  
评估目的：回到 `dagger_best.pt`，在双方速度均为 `1.5m/s`、追捕方与目标初始距离为 `[2.5, 3.5]m` 的条件下，测试抓捕率，并比较不同 TP 权重对结果的影响。

## 评估设置

- Actor 权重：`checkpoints/expert2_antcoll_dagger_1024x50_resume3_20260411_161648/dagger_best.pt`
- 并行环境数：`1024`
- 评估 episode 数：`1024`
- 每局最大步数：`1200`
- 追捕方速度：`1.5`
- 目标速度：`1.5`
- 初始距离：`[2.5, 3.5]`
- curriculum stage：脚本实际使用 `stage 4`，并临时覆盖 stage 4 的距离和目标速度
- 初始位置模式：`goal_side_arc`
- seed：`20260412`
- 策略执行：deterministic

## 结果

| 组合 | 抓捕率 | 到达守区率 | 坠机率 | 超时率 | 平均首次抓捕步数 | TP 预测误差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dagger_best` actor + 最新 RL TP (`checkpoint_113115136.pt`) | 45.6% (467/1024) | 49.8% (510/1024) | 0.0% | 4.6% (47/1024) | 227.3 | 0.0576 |
| `dagger_best` actor + 自带 TP | 48.9% (501/1024) | 48.1% (493/1024) | 0.0% | 2.9% (30/1024) | 242.9 | 0.2538 |

## 初步结论

这次结果低于之前记忆里的 60%+，主要不是因为坠机或动作崩溃：两组坠机率都是 `0%`。更可能的原因是本次评估条件更难，初始距离被固定在 `[2.5, 3.5]m`，且使用 `goal_side_arc` 的正面对抗/拦截分布，目标有更多时间进入守区。

一个反直觉但重要的现象是：最新 RL TP 的预测误差明显更低，但抓捕率反而比 `dagger_best` 自带 TP 低约 `3.3` 个百分点。这说明对当前 DAgger actor 来说，“预测更准”不一定直接带来更高抓捕率；actor 可能已经适配了训练时的 TP/观测分布，直接替换 TP 会造成输入分布偏移。

## 产物

- 最新 RL TP 覆盖评估日志：`analysis/dagger_best_speed15_dist25_35_20260412/eval_dagger_best_tp113115136_speed15_dist25_35_1024_20260412_212545.log`
- DAgger 自带 TP 评估日志：`analysis/dagger_best_speed15_dist25_35_20260412/eval_dagger_best_builtin_tp_speed15_dist25_35_1024_20260412_212743.log`
- 本次为了支持单独 TP 覆盖，扩展了 `scripts/eval_policy_batch.py` 的 `--tp_model_dir` 参数，并增加了实际 curriculum stage/距离/速度和 TP 预测误差打印。
