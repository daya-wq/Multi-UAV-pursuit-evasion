# TP预测网络监督训练说明

## 1. 数据采集口径
- 数据目录：`/data/uavlab/multi-uav-pursuit2/tp_datasets/tp_expert2_oraclenext_frontbox_1024x5_20260415_005910`
- 波次数：`5`
- 并行环境：`1024`
- 历史长度：`10` 步
- 预测长度：`5` 步
- window_step：`1`
- 追捕方速度：`1.5` m/s
- 目标速度：`1.5` m/s
- 仿真步长：`0.01` s
- 单回合步长上限：`1200`
- 位置归一化：`x,y / arena_size`，`z -> z/max_height*2-1`

## 2. 网络结构
- 直接复用仓库里的 `omni_drones.learning.TP_net`
- 输入：`[history_step, input_dim] = [10, 19]`
- 主干：`1层 LSTM(hidden_dim=64)`
- 输出：`future_step * 3 = 5 * 3`
- 输出激活：`tanh`，对应归一化后的未来 5 步目标位置

## 3. 损失函数
- 主损失：归一化坐标上的 MSE
- 形式：`Loss = mean((TP_net(TP_input) - TP_future.reshape(B, -1))^2)`
- 评估指标：`val_loss`、`ADE(m)`、`FDE(m)`、每个预测步的平均位置误差

## 4. 训练参数
- batch size：`8192`
- epochs：`20`
- learning rate：`0.0001`
- weight decay：`0.0`
- val_ratio：`0.1`
- num_workers：`4`
- device：`cuda:0`

## 5. 当前最佳结果
- best_epoch：`20`
- val_loss：`0.005066`
- val_ADE：`0.0401 m`
- val_FDE：`0.0405 m`
