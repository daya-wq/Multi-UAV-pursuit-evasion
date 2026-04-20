# 远距离 Expert2 BC 微调结果

时间：2026-04-14

## 训练设置

- 专家策略：`expert2`
- 模式：`goal_mode=false`，`close_mode=true`，`rush_mode=false`
- 初始分布：`front_box`
- 目标初始位置：`x=[-3.0, -2.5]`，`y=[-1.0, 1.0]`
- 追捕方初始位置：`x=[1.0, 2.0]`，`y=[-1.0, 1.0]`
- 追捕方最小初始间距：`0.7m`
- 双方速度：`v_drone=1.5`，`v_prey=1.5`
- 守区高度：`goal_region_height=1.0m`
- episode length：`1200`
- BC 初始权重：`checkpoints/expert2_antcoll_dagger_1024x50_resume3_20260411_161648/dagger_best.pt`
- BC 学习率：`1e-4`
- BC epoch：`6`

## 专家数据收集

- 总 rollout：`1024 * 50 = 51200` 局
- 成功专家轨迹：`28316` 局
- 专家成功轨迹步数：`11455339`
- 专家数据成功率：`28316 / 51200 = 55.3%`
- 数据目录：`expert_datasets/expert2_frontbox_goalh1_v15_1024x50_20260414_015109`

## BC 训练曲线

| epoch | loss | action MSE |
| --- | ---: | ---: |
| 1 | 0.010959 | 0.000951 |
| 2 | 0.009165 | 0.000683 |
| 3 | 0.008724 | 0.000657 |
| 4 | 0.008540 | 0.000644 |
| 5 | 0.008416 | 0.000633 |
| 6 | 0.008295 | 0.000624 |

最终权重：`checkpoints/expert2_frontbox_goalh1_bc_from_daggerbest_1024x50_20260414_030754/bc_final.pt`

## 同条件 Actor 评估

评估条件：`1024` 随机初始化环境，`front_box`，`v_drone=1.5`，`v_prey=1.5`，`goal_region_height=1.0m`，deterministic actor。

| 权重 | 捕获率 | 目标进守区 | 坠机 | 超时 | 平均捕获步 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原始 `dagger_best.pt` | 47.0% (481/1024) | 10.4% (107/1024) | 0.0% | 42.6% (436/1024) | 308.7 |
| 微调后 `bc_final.pt` | 27.6% (283/1024) | 25.3% (259/1024) | 0.0% | 47.1% (482/1024) | 373.4 |

## 结论

这次 BC 训练的监督误差确实下降了，但 closed-loop 策略表现明显退化。说明它在专家成功状态分布上更像专家动作了，但 rollout 时更容易偏离原本 DAgger actor 的稳定状态分布。

当前不建议使用这个 `bc_final.pt` 作为后续 RL warmstart。更安全的选择是继续使用原始 `dagger_best.pt`，或者只对前几个 epoch 做早停评估，寻找是否存在没有明显破坏闭环稳定性的微调点。
