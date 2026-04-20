# 当前这次行为模仿与 DAgger 的详细流程说明

本文档对应当前这次从零开始的模仿学习流水线：

- run 时间戳：`20260415_001304`
- 专家预测模式：`oracle_next`
- 专家策略：`expert2 + goal_mode=false + close_mode=true + rush_mode=false + front_layout=symmetric`
- 追捕方速度：`1.5`
- 目标速度：`1.5`
- 单局最大步数：`1200`
- 采集 / DAgger 并行环境数：`1024`
- BC 评测环境数：`1024`

对应主脚本：

- 流水线总控：[scripts/run_expert2_actor_imitation_pipeline.sh](/data/uavlab/multi-uav-pursuit2/scripts/run_expert2_actor_imitation_pipeline.sh)
- 专家采集：[scripts/expert_isaac_eval.py](/data/uavlab/multi-uav-pursuit2/scripts/expert_isaac_eval.py)
- BC 训练：[scripts/train_expert_bc.py](/data/uavlab/multi-uav-pursuit2/scripts/train_expert_bc.py)
- DAgger 训练：[scripts/train_actor_dagger.py](/data/uavlab/multi-uav-pursuit2/scripts/train_actor_dagger.py)
- BC 损失实现：[omni_drones/learning/mappo.py](/data/uavlab/multi-uav-pursuit2/omni_drones/learning/mappo.py)

## 1. 这条流水线到底在做什么

整体分 3 个阶段：

1. 先用专家策略在 Isaac Sim 里并行跑很多局，收集“专家成功轨迹”数据集。
2. 用这些专家成功轨迹做 BC，先把 actor 从零训练到至少能模仿出一个可用策略。
3. 只有当 BC 的评测捕获率达到 `60%` 以上，才进入 DAgger；DAgger 会继续收集“当前 actor 在在线环境里的成功轨迹”，再和原始专家数据按比例混合继续训练。

一句话理解：

- BC 是“先照着老师抄”
- DAgger 是“边跑边纠正，但永远不让在线垃圾数据把专家数据冲掉”

## 2. 当前这次的关键配置

当前 run 里最重要的参数来自 [scripts/run_expert2_actor_imitation_pipeline.sh](/data/uavlab/multi-uav-pursuit2/scripts/run_expert2_actor_imitation_pipeline.sh)：

- `START_FROM_SCRATCH=true`
- `PRED_MODE=oracle_next`
- `GENERIC_BATCH_ENVS=1024`
- `NUM_WAVES=50`
- `EPISODE_LENGTH=1200`
- `BC_EPOCHS=8`
- `BC_BATCH_SIZE=4096`
- `BC_LR=5e-4`
- `BC_EVAL_EVERY=1`
- `BC_N_EVAL=1024`
- `BC_EVAL_SUCCESS_THRESHOLD=0.60`
- `DAGGER_WAVES=50`
- `DAGGER_BATCH_ENVS=1024`
- `DAGGER_LR=1e-4`
- `DAGGER_REPLAY_UPDATES_PER_WAVE=4`
- `DAGGER_REPLAY_BATCH_SIZE=4096`
- `DAGGER_ONLINE_SUCCESS_ONLY=true`
- `DAGGER_ONLINE_MIN_WAVE_CAPTURE_RATE=0.60`
- `DAGGER_ONLINE_RATIO_INIT=0.10`
- `DAGGER_ONLINE_RATIO_WARMUP_WAVES=10`
- `DAGGER_ONLINE_RATIO_STEP=0.05`
- `DAGGER_ONLINE_RATIO_MAX=0.40`

这里有两个很重要的“从零开始”含义：

1. actor 不加载任何旧的 BC / DAgger 权重。
2. 但 BC 阶段的 `TP_net` 仍然会加载最新的 `tp_only_*.pt`，因为 actor 的观测管线里本身还会用到环境里的目标预测模块。也就是说：
   - actor 从零开始
   - critic 从零开始
   - TP 模块不是从零开始

这里要特别强调一个风险：

- 这只是“当前代码路径”的真实行为
- 不代表这是“当前任务定义下正确的行为”

如果目标动力学模型已经改过，而旧的 `tp_only_*.pt` 是按旧目标动力学训练出来的，那么这个 TP 权重就已经失配了。  
这种情况下，虽然代码会自动把它加载进来，但逻辑上它已经不该再被当作“可用预测模块”。

换句话说：

- 代码现在会自动加载旧 TP
- 但在你已经改过目标动力学之后，这件事本身是有问题的
- 更严重的是，只要环境 observation 还在使用 TP 预测，那么连专家采集下来的 `obs` 也会被过期 TP 污染
- 也就是说，问题不是只发生在 BC / DAgger，而是从“专家数据采集阶段”就已经开始了

## 3. 专家数据是怎么收集的

### 3.1 采集入口

专家采集由 [scripts/run_expert2_actor_imitation_pipeline.sh](/data/uavlab/multi-uav-pursuit2/scripts/run_expert2_actor_imitation_pipeline.sh) 调用 [scripts/expert_isaac_eval.sh](/data/uavlab/multi-uav-pursuit2/scripts/expert_isaac_eval.sh)，最终启动 [scripts/expert_isaac_eval.py](/data/uavlab/multi-uav-pursuit2/scripts/expert_isaac_eval.py)。

当前这次会跑：

- `50` 个 wave
- 每个 wave `1024` 个并行环境
- 每个环境 1 局
- 所以总 episode 数 = `50 * 1024 = 51200`

### 3.2 专家到底用什么做标签

当前这次你要求的是“使用真实数值去训练”，这里落实为两层含义：

1. 专家预测模式是 `oracle_next`
   - 即专家不是用 `tp_net` 预测目标下一时刻位置
   - 而是直接用目标的真实下一时刻位置
   - 这相当于给专家一个理想预测上界
2. actor 的监督标签是连续实数动作
   - 存储字段叫 `action_raw`
   - 类型是 `pidrate_normalized`
   - 含义是 4 维连续控制量：`[omega_x, omega_y, omega_z, thrust]`
   - 这 4 个量已经和你训练时的 `PIDRate` 控制链严格对齐

所以这不是离散 imitation，也不是只学一个 waypoint，而是直接学连续控制动作。

### 3.3 每一局里记录了什么

在 [scripts/expert_isaac_eval.py](/data/uavlab/multi-uav-pursuit2/scripts/expert_isaac_eval.py) 里，只有“成功回合”才会被整理成数据集。

每个保存下来的 `expert_success_wave_XXXXX.pt` 里主要有这些字段：

- `obs`
  - `state_self`
  - `state_others`
  - `cooperation`
- `expert_aux`
  - `vel_cmd`
  - `assignment`
  - `waypoint`
  - `target_pos_pred`
  - `forward_dir`
  - `trap_mode`
- `prev_action`
- `action_raw`
- `action_label_type = pidrate_normalized`
- `episode_lengths`
- `episode_meta`
- `num_success_episodes`
- `num_success_steps`
- `dropped_too_short`
- `storage_dtype`

也就是说，一条成功轨迹里不只是“状态 -> 动作”，还额外存了专家的角色分配、waypoint、预测点、前向方向等辅助监督信息。

## 4. 专家数据是怎么筛选的

### 4.1 第一层筛选：只保留成功轨迹

在 [scripts/expert_isaac_eval.py](/data/uavlab/multi-uav-pursuit2/scripts/expert_isaac_eval.py) 中，只有 `ep_info["success"] == True` 的 episode 才会进入保存列表。

这意味着：

- 失败回合不会进入专家数据集
- 到达守区、超时、坠地都不会被当作专家训练样本

### 4.2 第二层筛选：太短的成功轨迹也丢掉

当前 `MIN_SUCCESS_STEPS=25`。

也就是说，即使这一局最后算成功了，只要：

- 成功步数 `< 25`

它也会被记为 `dropped_too_short`，不会写进训练数据。

这样做的目的是避免一些“刚开局就贴脸成功”的极短轨迹把状态分布搞得太窄。

### 4.3 当前专家采集阶段不会保留失败数据

这点很关键：

- 目前的离线专家库是“纯成功库”
- 不会保存失败状态
- 不会保存中途接近成功但最终超时的轨迹

因此当前 BC 的目标不是“学会区分成功和失败”，而是“尽量复刻成功策略”

## 5. BC 阶段到底怎么训练

### 5.1 BC 的初始化方式

BC 训练由 [scripts/train_expert_bc.sh](/data/uavlab/multi-uav-pursuit2/scripts/train_expert_bc.sh) 调 [scripts/train_expert_bc.py](/data/uavlab/multi-uav-pursuit2/scripts/train_expert_bc.py)。

当前这次：

- actor 不加载任何旧 checkpoint
- critic 不加载任何旧 checkpoint
- 但 `policy.TP_net` 会自动加载最新的 TP 权重

所以更准确地说，当前代码下的 BC 不是“完全全随机一切都从零”，而是“策略从零开始，但默认会把环境里的 TP 模块初始化成旧权重”。  
在目标动力学已经变化的前提下，这一句不能再写成“预测模块保持可用”，因为这里的 TP 很可能已经失效。

### 5.2 BC 用的是哪些数据

BC 会把专家采集目录下所有 `expert_success_wave_*.pt` 都读进来。

然后再做一次可选过滤：

- `min_episode_len`
- `max_episode_len`
- `keep_prefix_steps`
- `keep_suffix_steps`

但当前这次的默认值是：

- `keep_prefix_steps = -1`
- `keep_suffix_steps = 64`

这在当前代码里等价于：`整条轨迹都保留`。

原因是 `keep_prefix_steps <= 0` 时，prefix 会被设为整条 episode 长度，所以并不会只截后 64 步。

### 5.3 BC 的采样方式

BC 不是按“整条轨迹”为一个 batch 去训，而是：

1. 读取一个 wave chunk
2. 把这个 chunk 里的所有步打平
3. 随机打乱步索引
4. 按 `batch_size=4096` 取 batch 更新

所以 BC 的最小训练单元是“单步监督”，不是整局监督。

### 5.4 BC 的样本权重

当前 BC 里 `front_weight_alpha = 0.5`。

这意味着：

- 一条轨迹前半段的步权重更高
- 后半段权重稍低
- 最终再归一化到平均权重约为 1

直观理解：

- 早期接近、围堵、角色展开阶段更重要
- 不希望训练只记住最后几步收口动作

### 5.5 BC 的损失函数

真实损失在 [omni_drones/learning/mappo.py](/data/uavlab/multi-uav-pursuit2/omni_drones/learning/mappo.py) 的 `update_actor_bc()` 里。

当前 BC 总损失可以写成：

```text
L_BC
= 1.0 * 动作 MSE
+ 0.02 * waypoint 辅助损失
+ 0.05 * assignment 辅助损失
+ 0.0 * vel_cmd 辅助损失
+ 0.0 * trap 辅助损失
+ 0.0 * log_prob 项
+ 0.0 * entropy bonus
```

其中：

- 动作 MSE：actor 输出动作和专家 `action_raw` 的均方误差
- waypoint 辅助损失：actor 的辅助头去回归专家 waypoint
- assignment 辅助损失：actor 的辅助头去预测专家分配的角色类别

所以当前 BC 不只是学动作，还在学：

- 我该去哪个 waypoint
- 我当前是什么角色分配

### 5.6 BC 训练多少轮，怎么评测

当前 BC 设置：

- `epochs = 8`
- 每个 epoch 结束后都做一次评测
- 每次评测 `1024` 局
- 评测时 actor 是 deterministic

评测指标包括：

- `capture_rate`
- `goal_rate`
- `landed_rate`
- `timeout_rate`
- `capture_steps_mean`

保存规则：

- 每个 epoch 存一个 `bc_epoch_XXX.pt`
- 最优评测模型存为 `bc_best.pt`
- 指标写到：
  - `bc_eval_history.json`
  - `bc_eval_history.csv`
  - `bc_best_metrics.json`
  - `bc_summary.json`

### 5.7 BC 什么时候才允许进入 DAgger

门槛非常明确：

- 如果 `BC best capture_rate < 0.60`
- 则整条流水线直接 `skip DAgger`

只有当：

- `BC best capture_rate >= 60%`

才会把 `bc_best.pt` 作为 DAgger 的初始策略继续往下跑。

## 6. DAgger 阶段到底怎么跑

### 6.1 DAgger 的起点

DAgger 不是从零开始。

它的起点是：

- `BC best checkpoint`

也就是先让 actor 有一个基本模仿能力，然后再做在线纠偏。

### 6.2 一轮 DAgger wave 会发生什么

每个 DAgger wave 做一次完整在线 rollout：

- 并行环境数：`1024`
- 每局最长：`1200` 步
- actor 用 deterministic 动作
- 同时每一步都会重新计算专家动作

每一步同时会得到两套动作：

- `actor_action`
- `expert_action`

然后用于两个目的：

1. 决定环境里实际执行谁的动作
2. 收集监督样本，后面继续训练 actor

### 6.3 rollout 时谁在控制无人机

这里有一个容易混淆的“mix_prob”。

当前代码里 rollout 的实际执行动作是：

- 默认执行 actor_action
- 但有一部分环境会被 `mix_prob` 替换成 `expert_action`

当前 `mix_prob` 不是固定值，而是从：

- `0.50 -> 0.10`

随着 DAgger wave 线性下降。

它的作用是：

- 前期更多依赖专家接管，避免 actor 完全失控
- 后期逐渐放手，让 actor 自己承担更多控制

注意：这不是你说的“90% 专家数据 + 10% 在线数据”那个比例。  
这是 rollout 执行动作时的“专家接管概率”。

### 6.4 DAgger 在线数据怎么收

当前设置 `online_success_only=true`。

所以在线 rollout 里：

- 只有成功 episode 才会被加入在线数据候选集合
- 失败 episode 不会写入在线成功库

并且每个成功 episode 保存的内容和离线专家库格式基本一致：

- `obs`
- `expert_aux`
- `prev_action`
- `action_raw`
- `episode_lengths`
- `episode_meta`

这里很重要的一点是：

- 在线数据里存的 supervision 仍然是专家动作，不是 actor 自己的动作
- 也就是说，在线数据本质上仍然是“状态 -> 专家标签”

## 7. DAgger 的双重过滤机制

### 7.1 第一个过滤：只要成功轨迹

如果某个 episode 最终没抓到目标：

- 它不会进入在线成功数据集

这和你现在的要求完全一致。

### 7.2 第二个过滤：整轮 wave 成功率必须大于 60%

当前还有一个更强的过滤：

- 如果这一整个 wave 的 `capture_rate <= 0.60`
- 那这一轮所有在线数据全部作废
- `effective_online_ratio = 0`

也就是说，在线数据要同时满足：

1. 单条轨迹本身成功
2. 这一整轮 actor 的整体水平也不能太差

这样做的目的就是避免“偶尔几条碰巧成功的垃圾波次”把在线数据污染进去。

## 8. 你要求的 90/10、再逐步 +5%，在代码里是怎么实现的

这里说的是“训练时 expert replay 和 online data 的混合比例”，不是 rollout 接管比例。

混合比例函数在 [scripts/train_actor_dagger.py](/data/uavlab/multi-uav-pursuit2/scripts/train_actor_dagger.py) 里是：

```text
前 10 轮:
  online_ratio = 0.10
  expert_ratio = 0.90

第 11 轮开始:
  online_ratio = 0.10 + (wave - 10) * 0.05

然后截断到最大 0.40
```

所以实际就是：

- wave 1~10：`90% 专家 + 10% 在线`
- wave 11：`85% 专家 + 15% 在线`
- wave 12：`80% 专家 + 20% 在线`
- wave 13：`75% 专家 + 25% 在线`
- wave 14：`70% 专家 + 30% 在线`
- wave 15：`65% 专家 + 35% 在线`
- wave 16 及以后：`60% 专家 + 40% 在线`

不会出现：

- `50/50`
- `100% online`
- `只用在线数据`

因为代码里明确要求：

- 没有 expert replay chunk，就不允许做 mixed update

## 9. DAgger 真正更新 actor 时怎么混合

每个 DAgger wave 结束后，会做 mixed updates。

当前设置：

- `replay_updates_per_wave = 4`
- `replay_batch_size = 4096`

每次 mixed update 的构造方式是：

1. 先从离线专家 replay 里采样 `expert_batch`
2. 再按 `effective_online_ratio` 从在线成功样本里采样 `online_batch`
3. 把两者拼接起来
4. 再调用和 BC 完全一样的 `update_actor_bc()` 去更新 actor

所以 DAgger 不是 PPO，也不是 RL 微调。  
它仍然是监督学习，只不过监督数据来源从“纯离线专家”变成了“离线专家 + 在线成功状态上的专家标注”。

## 10. 在线样本还有哪些额外权重

当前 DAgger 在线监督样本还有两个额外加权：

- `online_goal_weight_alpha = 0.5`
  - 越接近守区的状态，监督权重越高
- `online_disagreement_weight_alpha = 2.0`
  - actor 和 expert 动作差异越大的状态，监督权重越高

直观理解：

- 越危险的状态，更该认真学
- 越偏离专家的状态，更该重点纠正

所以 DAgger 不是简单把在线数据混进去，而是会优先“纠正偏得最厉害、最关键的位置”。

## 11. DAgger 里有哪些数据不会被立即用于训练

当前 `defer_online_updates=true`。

意思是：

- 在线 rollout 时不会边跑边立即更新 actor
- 而是先把这一轮的在线成功数据缓存起来
- wave 结束后再统一和专家 replay 混起来更新

这样更稳定，也更容易统计每一轮的真实表现。

## 12. 评测、保存和 best checkpoint 是怎么管理的

### 12.1 BC 阶段

保存：

- `bc_epoch_XXX.pt`
- `bc_best.pt`
- `bc_final.pt`

统计文件：

- `bc_eval_history.json`
- `bc_eval_history.csv`
- `bc_best_metrics.json`
- `bc_summary.json`

### 12.2 DAgger 阶段

保存：

- `dagger_wave_XXX.pt`
- `dagger_best.pt`
- `dagger_final.pt`

统计文件：

- `dagger_eval_history.json`
- `dagger_eval_history.csv`
- `dagger_best_metrics.json`
- `dagger_summary.json`

在线数据集：

- 每轮合格在线数据会保存成
  - `expert_success_wave_dagger_XXXXX.pt`

## 13. 当前这条链路的一句话总结

当前这套流程可以概括为：

1. 用 `oracle_next` 专家在 Isaac GPU 并行环境里收集成功轨迹。
2. 只保留成功且长度至少 25 步的轨迹，形成离线专家库。
3. 用离线专家库做 BC，从零训练 actor。
4. BC 每个 epoch 都做 1024 局评测，只有超过 60% 才进入 DAgger。
5. DAgger 以 `bc_best.pt` 为起点，在线跑 actor，但每一步都同时计算专家标签。
6. 在线数据必须同时满足“单条成功”和“整轮成功率 > 60%”才会被保留。
7. 训练时始终保留专家 replay 主体，前 10 轮 `90/10`，之后每轮在线数据 `+5%`，最多到 `40%`。
8. 整个 DAgger 过程本质上仍然是监督学习，不是 PPO 强化学习。

## 14. 你最容易混淆的三个点

### 14.1 `oracle_next` 和 `TP_net` 不是一回事

- `oracle_next`：专家拿到目标真实下一时刻位置
- `TP_net`：专家拿到预测网络输出

当前这次专家标签是 `oracle_next`，不是 `TP_net`。

### 14.2 “90/10 数据比例”不是 rollout 接管比例

- `online_ratio`：控制训练 batch 里 expert / online 的混合比例
- `expert_mix_prob`：控制 rollout 时有多少环境直接执行专家动作

这两个量不是一回事。

### 14.3 DAgger 仍然是在学专家，不是在学 actor 自己

在线样本保存的是：

- 在线状态
- 对应的专家动作标签

不是把 actor 自己跑出来的动作直接当真值继续训。

所以当前 DAgger 的本质是：

- 扩大状态分布
- 但监督老师仍然是专家
