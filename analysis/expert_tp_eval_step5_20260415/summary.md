# 第5步预测点作为 intercept_seed 的专家策略评测

## 测试设置

- TP 权重：`checkpoints/tp_supervised_20260415_005910/tp_only_20260415_010548.pt`
- 专家策略：`expert2`
- 预测模式：`tp_net`
- `goal_mode=false`
- `close_mode=true`
- `rush_mode=false`
- 前向布局：`symmetric`
- `intercept_pred_step=5`
- 追捕方速度：`1.5 m/s`
- 目标速度：`1.5 m/s`
- 并行环境：`1024`
- 总回合数：`1024`
- 单回合步长：`1200`
- 初始化：`front_box` 正面对抗随机初始化
- 环境内旧 TP 分支：关闭（`algo.use_TP_net=0`）

## 结果

- 捕获率：`914 / 1024 = 89.26%`
- 目标到达守区：`10 / 1024 = 0.98%`
- landed：`5 / 1024 = 0.49%`
- timeout：`95 / 1024 = 9.28%`
- 任意碰撞回合：`404 / 1024 = 39.45%`
- 机间碰撞回合：`58 / 1024 = 5.66%`
- 首次捕获平均步数：`396.1`

## 与默认第1步版本对比

- 默认 `intercept_pred_step=1`：`88.18%`
- 当前 `intercept_pred_step=5`：`89.26%`
- 提升：`+1.08 个百分点`

## 备注

日志里的 `Expert pred pos_err / next_err` 变大是正常的。  
因为这里 `target_pos_pred` 已经不再表示“预测下一步位置”，而是“预测第5步位置”，所以拿它去和当前 / 下一步真值比，误差当然会更大。这个指标在这次实验里不能直接和第1步版本横向比较。

