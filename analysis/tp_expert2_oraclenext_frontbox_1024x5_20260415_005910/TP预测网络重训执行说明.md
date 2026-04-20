# TP预测网络重训执行说明

## 1. 这次要解决的问题

你现在的目标动力学模型已经改了，所以以前的 TP 权重不再可信。  
这次我们不复用旧 TP，而是重新采一套和当前目标动力学一致的监督数据，再重新训练 TP 网络。

核心思路：

1. 让追捕方继续执行当前专家策略去追目标。
2. 在 Isaac Sim 里并行跑 `1024` 个环境，连续跑 `5` 个 wave，总共 `5120` 个 episode。
3. 在每个 episode 里，不是收 actor 数据，而是专门收 TP 监督数据：
   - 输入：目标前 `10` 步历史窗口
   - 标签：目标后 `5` 步真实轨迹
4. 用这套真值窗口数据重新训练 TP 网络。

## 2. 这次的采集设置

- 采集脚本：[expert_isaac_eval.py](/data/uavlab/multi-uav-pursuit2/scripts/expert_isaac_eval.py)
- 启动脚本：[run_tp_retrain_pipeline.sh](/data/uavlab/multi-uav-pursuit2/scripts/run_tp_retrain_pipeline.sh)
- 训练脚本：[train_tp_supervised.py](/data/uavlab/multi-uav-pursuit2/scripts/train_tp_supervised.py)

本次启动参数：

- 专家策略：`expert2`
- 预测模式：`oracle_next`
- `goal_mode=false`
- `close_mode=true`
- `rush_mode=false`
- 前向布局：`symmetric`
- 追捕方速度：`1.5 m/s`
- 目标速度：`1.5 m/s`
- 并行环境数：`1024`
- wave 数：`5`
- 总 episode 数：`5120`
- 单回合上限：`1200` 步
- 初始化方式：当前环境配置下的 `front_box` 正面对抗随机初始化

这里让专家用 `oracle_next` 的原因很直接：

- 我们想让追捕行为尽量稳定，避免因为旧 TP 预测误差把追捕过程本身搞乱。
- 这次 TP 的监督标签必须来自真实目标轨迹，不应该再被旧预测网络污染。

## 3. TP 数据是怎么构造的

环境里每一步都会形成一条 TP 输入帧，结构和你后面在线使用 TP 时保持一致。

### 3.1 单步 TP 输入

当前 TP 输入维度是 `19`，每一帧包含：

1. 当前回合进度 `progress_buf / max_episode_length`
2. 当前目标位置（归一化后）
3. 当前目标速度（归一化后）
4. 追捕方位置展开后的向量（归一化后）

然后把最近 `10` 帧堆起来，得到：

- `TP_input.shape = [10, 19]`

### 3.2 监督标签

标签不是“下一步单点”，而是“后 5 步完整轨迹”：

- `TP_future.shape = [5, 3]`

也就是：

- 第 1 个点：`t+1`
- 第 2 个点：`t+2`
- 第 3 个点：`t+3`
- 第 4 个点：`t+4`
- 第 5 个点：`t+5`

### 3.3 窗口筛选规则

不是每一步都能拿来训 TP。我们只保留合法窗口：

1. 当前这一步之后，未来至少还剩 `5` 步。
2. 这 `5` 步未来窗口中间不能跨 episode 终止边界。
3. 终止后那种“补出来的无意义尾巴”不会进数据集。

所以这次数据集不是“把所有 step 全存下来”，而是：

- 先跑完整个 wave
- 再从每个 episode 里切出所有合法的 `10 -> 5` 监督样本

## 4. 预测网络设计

这次不改 TP 网络本体，直接复用仓库现有定义 [mappo.py](/data/uavlab/multi-uav-pursuit2/omni_drones/learning/mappo.py) 里的 `TP_net`。

结构如下：

```text
输入: [B, 10, 19]
   ↓
1层 LSTM
   hidden_dim = 64
   num_layers = 1
   ↓
取最后一个时间步隐藏状态
   ↓
Linear(64 -> 15)
   ↓
tanh
   ↓
输出: [B, 15] -> reshape 为 [B, 5, 3]
```

解释一下：

- `10`：历史长度
- `19`：每步输入特征维度
- `15 = 5 * 3`：未来 5 步，每步 3 维位置
- `tanh`：因为标签是归一化到有限范围内的

## 5. 位置归一化方式

训练时使用归一化坐标，和环境里 TP 分支一致：

- `x_norm = x / arena_size`
- `y_norm = y / arena_size`
- `z_norm = z / max_height * 2 - 1`

所以网络输出的每一个未来点都是归一化坐标。  
评估时再把它反归一化回米制坐标，计算实际误差。

## 6. 损失函数

这次主损失就是最直接的监督 MSE：

```text
pred = TP_net(TP_input)                  # [B, 15]
target = TP_future.reshape(B, 15)        # [B, 15]
Loss = mean((pred - target)^2)
```

也就是说：

- 不加 KL
- 不加额外正则
- 不混 reward
- 就是纯监督地逼近未来 5 步真值轨迹

## 7. 训练参数

当前默认训练参数是：

- optimizer：`Adam`
- learning rate：`1e-4`
- weight decay：`0.0`
- batch size：`8192`
- epochs：`20`
- 验证集比例：`10%`
- 训练设备：`cuda:0`
- 数据存储精度：`float16`
- 训练时计算精度：`float32`

## 8. 训练时看的指标

除了 `train_loss` / `val_loss`，这次我还额外记三类误差：

1. `ADE`
   - 未来 5 步平均位置误差，单位米
2. `FDE`
   - 第 5 步终点误差，单位米
3. `h1 ~ h5`
   - 每一个未来步各自的平均位置误差，单位米

这样你能直接看出：

- 是所有预测步都偏了
- 还是只是越往后越飘

## 9. 启动顺序

这次实际执行顺序就是：

1. 启动 `expert_isaac_eval.py`
2. 专家策略在 `1024` 并行环境里跑 `5` 个 wave
3. 每个 wave 结束后，保存一个 `tp_wave_0000x.pt`
4. 全部采完后，启动 `train_tp_supervised.py`
5. 训练过程中每个 epoch 保存 checkpoint
6. 同时保存 best TP 权重，命名为 `tp_only_时间戳.pt`

## 10. 本次运行目录

- 运行目录：[tp_expert2_oraclenext_frontbox_1024x5_20260415_005910](/data/uavlab/multi-uav-pursuit2/analysis/tp_expert2_oraclenext_frontbox_1024x5_20260415_005910)
- 采集日志：[collect_tp.log](/data/uavlab/multi-uav-pursuit2/analysis/tp_expert2_oraclenext_frontbox_1024x5_20260415_005910/collect_tp.log)
- 训练日志：[train_tp.log](/data/uavlab/multi-uav-pursuit2/analysis/tp_expert2_oraclenext_frontbox_1024x5_20260415_005910/train_tp.log)

采集完成后我会继续把这份文档补上：

- 实际采到多少个 TP 样本
- 训练最优 epoch
- 最优 `val_loss / ADE / FDE`
- 最终保存下来的 TP 权重路径
