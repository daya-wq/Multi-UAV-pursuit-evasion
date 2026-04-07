
# Expert Dataset + BC 交接文档（2026-04-04）

## 1. 目标

这次做的事情，不是继续直接训 RL，而是走一条“专家数据预训练 actor”的路线：

1. 用当前已经调好的专家策略，在 Isaac Sim 里直接控制追捕者。
2. 只收集成功 episode 的轨迹，做成专家经验库。
3. 用这些成功轨迹做 BC（Behavior Cloning）训练。
4. 再把训出来的 actor 放回 Isaac 里做真实 rollout，检查它到底有没有学会专家行为。

大白话说就是：

- 专家自己会追。
- 先让专家跑，存下“好样本”。
- 再让神经网络模仿这些“好样本”。
- 最后检查网络是不是真的学会了。


## 2. 这条链路里新增/修改了什么

### 2.1 成功轨迹采集

在 [scripts/expert_isaac_eval.py](/data/uavlab/multi-uav-pursuit2/scripts/expert_isaac_eval.py#L1015) 到 [scripts/expert_isaac_eval.py](/data/uavlab/multi-uav-pursuit2/scripts/expert_isaac_eval.py#L1229) 里，新增了 batched episode 采集逻辑：

- batched 专家策略并行控制多个 Isaac 环境
- 只保留 `success=True` 的 episode
- 把成功 episode 的所有 step 都拼平后存盘

真正落盘的位置在 [scripts/expert_isaac_eval.py](/data/uavlab/multi-uav-pursuit2/scripts/expert_isaac_eval.py#L1488) 到 [scripts/expert_isaac_eval.py](/data/uavlab/multi-uav-pursuit2/scripts/expert_isaac_eval.py#L1510)。

### 2.2 BC 训练

在 [scripts/train_expert_bc.py](/data/uavlab/multi-uav-pursuit2/scripts/train_expert_bc.py#L68) 到 [scripts/train_expert_bc.py](/data/uavlab/multi-uav-pursuit2/scripts/train_expert_bc.py#L267) 里，新增了专家数据训练入口：

- 读取 `expert_success_wave_*.pt`
- 按 chunk 逐块加载
- 从专家 obs 重建 actor 输入
- 用专家动作监督 actor
- 每个 epoch 存一次 checkpoint
- 最后额外存一个 `bc_final.pt`

### 2.3 BC 损失函数

实际 BC 更新逻辑在 [omni_drones/learning/mappo.py](/data/uavlab/multi-uav-pursuit2/omni_drones/learning/mappo.py#L342) 到 [omni_drones/learning/mappo.py](/data/uavlab/multi-uav-pursuit2/omni_drones/learning/mappo.py#L399)。

当前这版 BC 的核心损失是：

```text
bc_loss = - mean(log_prob(expert_action))
```

另外还会额外记录一个：

```text
bc_action_mse = MSE(deterministic_pred_action, expert_action)
```

但注意：

- `bc_action_mse` 只是记录指标
- 当前不会参与反向传播
- 真正更新 actor 的仍然只是 `-log_prob`

这个点后来被证明很关键，因为它几乎就是这次 BC 失败的核心原因。

### 2.4 一键流水线

在 [scripts/run_expert_bc_pipeline.sh](/data/uavlab/multi-uav-pursuit2/scripts/run_expert_bc_pipeline.sh#L1) 到 [scripts/run_expert_bc_pipeline.sh](/data/uavlab/multi-uav-pursuit2/scripts/run_expert_bc_pipeline.sh#L77) 里，新增了自动流水线：

1. 先跑专家轨迹采集
2. 自动识别新生成的数据集目录
3. 再自动启动 BC 训练


## 3. 专家采集时的实际参数

这次正式大规模采集，实际走的是：

- batched Isaac generic eval
- 不录视频
- 使用当前专家策略
- 只保留成功 episode

流水线默认参数来自 [scripts/run_expert_bc_pipeline.sh](/data/uavlab/multi-uav-pursuit2/scripts/run_expert_bc_pipeline.sh#L6) 到 [scripts/run_expert_bc_pipeline.sh](/data/uavlab/multi-uav-pursuit2/scripts/run_expert_bc_pipeline.sh#L23)：

- `PRED_MODE=tp_net`
- `V_PREY_TEST=1.5`
- `V_DRONE_TEST=1.5`
- `EPISODE_LENGTH=1000`
- `GENERIC_BATCH_ENVS=2048`
- `NUM_WAVES=100`
- `DATASET_DTYPE=float16`
- `BC_EPOCHS=10`
- `BC_BATCH_SIZE=4096`
- `BC_LR=5e-4`
- `BC_DEVICE=cuda:0`
- `BC_ENTROPY_BONUS_COEF=0.0`

注意一点：

- 脚本默认 `MIN_SUCCESS_STEPS=1`
- 但这次正式跑的时候，我手动把它改成了 `2`

也就是说，这次正式采集实际用的是：

- `MIN_SUCCESS_STEPS=2`

大白话说：

- `step=0` 或 `step=1` 这种“出生点太近就秒抓”的样本，不进专家库
- 这样数据会更干净一点

专家评测/采集脚本里和环境相关的关键设置在 [scripts/expert_isaac_eval.py](/data/uavlab/multi-uav-pursuit2/scripts/expert_isaac_eval.py#L1265) 到 [scripts/expert_isaac_eval.py](/data/uavlab/multi-uav-pursuit2/scripts/expert_isaac_eval.py#L1386)：

- `num_envs = 2048`
- `episode_length = 1000`
- `random_init = true`
- `cfg.task.use_eval = 0`
- `cfg.task.v_drone = 1.5`
- target curriculum stage 被强制切到最后一阶段
- `current_target_speed = 1.5`
- `target_velocity_scale = 1.5`

所以正式采集时，双方速度都是：

- 追捕者 `1.5 m/s`
- 目标 `1.5 m/s`

仿真步长沿用环境默认值：

- `dt = 0.01 s`

TP 预测模式是：

- `tp_net`

实际使用的 TP 权重是：

- [tp_only_1690959872.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/HideAndSeek_20260403_001241/tp_only_1690959872.pt)


## 4. 专家数据集到底存了什么

每个 wave 会保存一个：

- `expert_success_wave_XXXXX.pt`

单个 chunk 的保存结构来自 [scripts/expert_isaac_eval.py](/data/uavlab/multi-uav-pursuit2/scripts/expert_isaac_eval.py#L1218) 到 [scripts/expert_isaac_eval.py](/data/uavlab/multi-uav-pursuit2/scripts/expert_isaac_eval.py#L1227)。

字段包括：

- `obs`
- `action_raw`
- `episode_lengths`
- `episode_meta`
- `num_success_episodes`
- `num_success_steps`
- `dropped_too_short`
- `storage_dtype`
- `wave`
- `pred_mode`
- `v_prey_test`
- `v_drone_test`
- `episode_length`
- `batch_envs`
- `dt`

其中最关键的是：

- `obs`
  - 当前包含 3 类输入
  - `cooperation`
  - `state_self`
  - `state_others`
- `action_raw`
  - 形状是 `[num_success_steps, 3, 4]`
  - 存的是专家输出后映射成训练动作空间的原始动作
  - 也就是追捕方训练时用的 `PIDrate` 动作
  - 不是电机 RPM，也不是后处理后的推力

一个实际 chunk 的例子：

- wave 1 中 `num_success_episodes = 1378`
- `num_success_steps = 433393`
- `action_raw` 形状是 `(433393, 3, 4)`
- `storage_dtype = torch.float16`


## 5. 正式采集结果

正式专家数据集目录：

- [HideAndSeek_expert_20260404_012945](/data/uavlab/multi-uav-pursuit2/expert_datasets/HideAndSeek_expert_20260404_012945)

主日志：

- [expert_eval_20260404_012906.log](/data/uavlab/multi-uav-pursuit2/expert_eval_20260404_012906.log)

最终统计结果是：

- 总 episode 数：`2048 * 100 = 204800`
- 成功 episode：`135833`
- 成功率：`66.3%`
- 成功 step 总数：`42461198`
- 被 `MIN_SUCCESS_STEPS=2` 过滤掉的过短成功局：`486`
- 数据集总大小：约 `14.74 GB`
- 数据 chunk 数：`100`

大白话解释：

- 专家自己是有用的
- 它不是完美专家，但成功率不低
- 成功样本量也足够大

所以这次后面 BC 训练如果效果差，优先怀疑：

- 不是专家库太小
- 也不是专家完全不会
- 更像是 BC 方法本身没把专家动作学进去


## 6. BC 训练时的实际参数

BC 训练入口脚本：

- [scripts/train_expert_bc.sh](/data/uavlab/multi-uav-pursuit2/scripts/train_expert_bc.sh#L13)
- [scripts/train_expert_bc.py](/data/uavlab/multi-uav-pursuit2/scripts/train_expert_bc.py#L28)

这次正式 BC 的实际参数是：

- 数据集目录：
  - [HideAndSeek_expert_20260404_012945](/data/uavlab/multi-uav-pursuit2/expert_datasets/HideAndSeek_expert_20260404_012945)
- `epochs = 10`
- `batch_size = 4096`
- `bc_lr = 5e-4`
- `entropy_bonus_coef = 0.0`
- `device = cuda:0`
- `shuffle_chunks = true`
- `max_chunks = -1`
- `save_every = 1`
- `save_tag = expert_bc_2048x100`

初始化方式是：

- 没有传 `model_dir`
- 所以不是从旧 actor 继续训
- 会自动寻找最新 `tp_only_*.pt`
- 然后只加载 TP 网络

对应逻辑在 [scripts/train_expert_bc.py](/data/uavlab/multi-uav-pursuit2/scripts/train_expert_bc.py#L159) 到 [scripts/train_expert_bc.py](/data/uavlab/multi-uav-pursuit2/scripts/train_expert_bc.py#L171)。

这次自动加载到的 TP 权重就是：

- [tp_only_1690959872.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/HideAndSeek_20260403_001241/tp_only_1690959872.pt)

所以这次 BC 的真实含义是：

- actor 从随机开始学
- critic 也不是重点
- TP 沿用已经训好的预测网络


## 7. BC 训练输出位置

正式 BC checkpoint 目录：

- [expert_bc_2048x100_20260404_040607](/data/uavlab/multi-uav-pursuit2/checkpoints/expert_bc_2048x100_20260404_040607)

最终 checkpoint：

- [bc_final.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/expert_bc_2048x100_20260404_040607/bc_final.pt)

TensorBoard run：

- [expert_bc_2048x100_20260404_040607](/data/uavlab/multi-uav-pursuit2/runs/expert_bc_2048x100_20260404_040607)


## 8. BC 训练指标结果

从 TensorBoard 看，这次 BC 呈现出一个很明显的特征：

- 前 1 到 2 个 epoch 有变化
- 后面基本完全平台

关键指标如下：

### 8.1 `bc_log_prob`

- epoch 1：`2.7880`
- epoch 2：`4.3242`
- epoch 3-10：基本维持 `4.3242`

### 8.2 `bc_loss`

- epoch 1：`-11.1522`
- epoch 2：`-17.2970`
- epoch 3-10：基本维持 `-17.2970`

### 8.3 `bc_action_mse`

- epoch 1：`0.0856567`
- epoch 10：`0.0856591`

几乎没变。

### 8.4 `bc_grad_norm`

- epoch 1：`3.0724`
- epoch 2-10：`0.0`

这个现象非常重要，说明：

- 训练在第 2 轮后几乎就不再更新了

### 8.5 `epoch_steps`

- 每个 epoch 都扫了 `42461200` 个 step

所以这次不是“训练量不够”。  
每个 epoch 实际上已经把整套专家数据完整扫了一遍。


## 9. BC 后的真实 rollout 评测

为了验证 `bc_final.pt` 到底有没有学会专家行为，我新增了批量评测脚本：

- [scripts/eval_policy_batch.py](/data/uavlab/multi-uav-pursuit2/scripts/eval_policy_batch.py#L153)

这个脚本会：

- 加载训练出来的策略 checkpoint
- 放进 Isaac Sim
- 在随机初始化下做真实 deterministic rollout
- 统计 `Capture / Goal zone / Landed / Timeout`

### 9.1 大 batch 评测遇到的问题

用 `512` 和 `1024` 并行评测这版 `bc_final.pt` 时，出现了显存不足：

- 1024 batch 评测：Isaac 收尾段错误
- 512 batch 评测：明确报 `torch.cuda.OutOfMemoryError`

这说明当前这版 actor 评测时的 LSTM 开销不小，大 batch rollout 不太稳。

### 9.2 最终可信的 rollout 结果

我最终用 `256` 并行做了稳定评测，日志在：

- [bc_policy_eval_20260404_256.log](/data/uavlab/multi-uav-pursuit2/bc_policy_eval_20260404_256.log)

评测设置是：

- `n_eval = 256`
- `batch_envs = 256`
- `episode_length = 1000`
- `v_drone = 1.5`
- `v_prey = 1.5`
- `random_init = true`

最终结果：

- `Capture = 0 / 256 = 0%`
- `Goal zone = 5 / 256 = 2%`
- `Landed = 251 / 256 = 98%`
- `Timeout = 0 / 256 = 0%`

大白话翻译：

- 网络几乎没有学会抓捕
- 也不是主要输在 goal
- 而是几乎一上来就摔


## 10. 现在该怎么理解这个结果

### 10.1 专家库本身不是主要问题

证据：

- 专家自己成功率大约 `66.3%`
- 数据量也很大，`4246 万` 成功 step
- 数据里只保留成功局
- 还过滤掉了一部分过短 trivial 成功局

所以这次不是“专家太差，没法学”。

### 10.2 当前 BC 目标设计有明显问题

这次 BC 更新逻辑真正优化的是：

```text
- mean(log_prob(expert_action))
```

但我们真正更关心的是：

```text
deterministic actor 输出，能不能贴近 expert_action
```

而这件事当前只是记录了 `bc_action_mse`，并没有纳入损失。

结果就变成：

- `bc_log_prob` 变好
- 说明“专家动作在分布里不算离谱了”
- 但 deterministic 均值动作没明显贴近专家
- 所以一做真实 rollout，就直接崩成 `98% landed`

一句最直白的话：

**这次网络更像学会了“嘴上说自己会”，但手上并没有真的把专家动作做出来。**

### 10.3 训练平台太早

`bc_grad_norm` 第 2 个 epoch 后几乎就是 `0`，这说明：

- 不是 10 个 epoch 太少
- 而是按当前损失定义，它在第 2 轮就基本卡死了

也就是说，就算机械地继续把 epoch 从 `10` 加到 `20`、`30`，收益大概率也不会太大。

### 10.4 rollout 里的主失败模式已经很清楚

当前 BC actor 的失败模式不是：

- 超时太多
- 或者总是差一点点没抓到

而是：

- `98% landed`

这说明它输出的动作一放进真实物理里，闭环就不稳定。


## 11. 当前最可信的结论

这次完整流程已经证明了 3 件事：

1. 专家策略可以在 Isaac 里稳定产生大量成功数据。
2. 成功专家库已经成功采集出来了，规模足够大。
3. 当前这版 BC 训练方法，并没有把 actor 真正训成专家。

所以现在的阻塞点不是：

- 缺数据
- 或者专家完全没用

而是：

- **当前 BC loss 不对路**


## 12. 最推荐的下一步

如果继续往前推进，最建议改的是 BC，而不是重新采数据。

优先级建议如下：

1. 把 `bc_action_mse` 真正纳入优化目标  
   也就是把当前纯 `NLL/log_prob`，改成类似：

   ```text
   total_loss = action_mse + alpha * (-log_prob)
   ```

   或者干脆先用 `MSE` 做主损失。

2. 检查 LSTM actor 的 BC 训练和 rollout 时 hidden state 处理  
   因为当前 rollout 的“几乎一上来就摔”，也可能和时序状态处理不一致有关。

3. BC 改好后，不需要重采专家库  
   现有这套专家数据完全可以继续用。


## 13. 本次关键文件一览

### 专家采集

- [scripts/expert_isaac_eval.py](/data/uavlab/multi-uav-pursuit2/scripts/expert_isaac_eval.py)
- [scripts/expert_isaac_eval.sh](/data/uavlab/multi-uav-pursuit2/scripts/expert_isaac_eval.sh)

### BC 训练

- [scripts/train_expert_bc.py](/data/uavlab/multi-uav-pursuit2/scripts/train_expert_bc.py)
- [scripts/train_expert_bc.sh](/data/uavlab/multi-uav-pursuit2/scripts/train_expert_bc.sh)
- [omni_drones/learning/mappo.py](/data/uavlab/multi-uav-pursuit2/omni_drones/learning/mappo.py#L342)

### 自动流水线

- [scripts/run_expert_bc_pipeline.sh](/data/uavlab/multi-uav-pursuit2/scripts/run_expert_bc_pipeline.sh)

### rollout 评测

- [scripts/eval_policy_batch.py](/data/uavlab/multi-uav-pursuit2/scripts/eval_policy_batch.py)

### 结果产物

- [HideAndSeek_expert_20260404_012945](/data/uavlab/multi-uav-pursuit2/expert_datasets/HideAndSeek_expert_20260404_012945)
- [expert_bc_2048x100_20260404_040607](/data/uavlab/multi-uav-pursuit2/checkpoints/expert_bc_2048x100_20260404_040607)
- [bc_final.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/expert_bc_2048x100_20260404_040607/bc_final.pt)
- [expert_eval_20260404_012906.log](/data/uavlab/multi-uav-pursuit2/expert_eval_20260404_012906.log)
- [bc_policy_eval_20260404_256.log](/data/uavlab/multi-uav-pursuit2/bc_policy_eval_20260404_256.log)


## 14. 一句话总结

这次流程里：

- 专家数据采成功了
- 专家库规模也够大
- 但当前 BC 只是把“专家动作的概率”学高了
- 没有把“专家动作本身”真正学会
- 所以真实 rollout 里几乎全摔了

下一步最该改的是 **BC 损失和时序训练方式**，不是重新怀疑专家库。


## 15. 后续 DAgger / Recurrent 调试进展（2026-04-04 夜）

在上面那版纯离线 BC 之后，我继续沿着“端到端输出推力和角速度不变，但让 actor 学会时序纠偏”这条线往下做了 DAgger 和 recurrent 调试。

### 15.1 先确认了一件大事

问题**不是**动作空间错了。

- 专家给 actor 的监督标签，依然是训练同一套 `PIDrate raw action`
- 也就是专家先算 `t + omega`
- 再映射成 actor / RL 真正输出的 4 维原始动作

所以后面 actor 学不好，不是“专家在教速度、网络在学推力角速度”这种错位。

### 15.2 纯 DAgger 的最好结果

我先前把 actor 从纯离线 BC 的 `18% capture`，拉到了大约 `40%` 左右。

其中老的最好外测基线是：

- [dagger_geom_wave1_eval.log](/data/uavlab/multi-uav-pursuit2/dagger_geom_wave1_eval.log)
- `Capture 42% / Goal 25% / Landed 20% / Timeout 13%`

这说明：

- 在线纠偏是有效的
- 但还远没到专家本体的 `66%`

### 15.3 关键根因的新判断

继续排查后，我认为纯 feedforward actor 卡住的主要原因是：

- actor 会学到“当前这一帧该怎么打动作”
- 但专家策略里其实有明显的**时序记忆/角色连续性**
- 比如 anchor 分配和 `prev_assignment` 带来的“不要乱换角色”

大白话说：

- 专家不是每一帧都临时拍脑袋
- 它是“上一秒你是 blocker，这一秒大概率还应该继续当 blocker”
- 而 feedforward actor 很容易学成“这一帧看起来像谁就临时切谁”

### 15.4 我补了什么代码

为了验证这个判断，我把 recurrent actor 这条线真正接通了：

- [omni_drones/learning/mappo.py](/data/uavlab/multi-uav-pursuit2/omni_drones/learning/mappo.py)
  - 修了 shared actor + RNN 时 `agent` 维和 `time` 维的 `vmap` 接法
  - 保存 checkpoint 时会额外记录 `actor_has_rnn` 和 `actor_rnn_hidden_size`
  - `update_actor_bc()` 允许在 recurrent BC/DAgger 里缺省 `actor_rnn_state`
- [scripts/train_actor_dagger.py](/data/uavlab/multi-uav-pursuit2/scripts/train_actor_dagger.py)
  - 新增了 `--enable_actor_rnn`
  - 新增了按 episode 切固定长度 sequence chunk 的逻辑
  - rollout 时会把 `actor_rnn_state` 显式从当前步传到下一步
  - 修了低空 rescue / success-only 序列拼接时的一些工程 bug
- [scripts/eval_policy_batch.py](/data/uavlab/multi-uav-pursuit2/scripts/eval_policy_batch.py)
  - 支持自动识别 recurrent actor checkpoint
  - rollout 时同样显式传递 `actor_rnn_state`

### 15.5 recurrent probe 的结果

我先做了一个最小 GRU probe：

- 起点：`success-only` 的老最好 checkpoint
- 配置：`GRU hidden=128`，`train_seq_len=8`
- 结果：
  - [dagger_gru_probe.log](/data/uavlab/multi-uav-pursuit2/dagger_gru_probe.log)
  - `Capture 40.6% / Goal 23.4% / Landed 32.8% / Timeout 3.1%`

这个结果说明：

- 只要把 GRU 随便接上，不会自动飞升
- 但它至少证明这条 recurrent 路已经真实跑通了

### 15.6 当前最好结果

真正有提升的是下面这条：

- 起点：`rescuefix` 基线 checkpoint
- 做法：`GRU + 2 waves DAgger`
- 结果：
  - [dagger_gru_rescuefix_2w.log](/data/uavlab/multi-uav-pursuit2/dagger_gru_rescuefix_2w.log)
  - `Capture 45.3% / Goal 18.8% / Landed 25.0% / Timeout 10.9%`

这已经超过了之前 `41%~42%` 的旧平台。

然后我又从这个 best checkpoint 往下做了一轮“低空 rescue continuation”：

- checkpoint：
  - [dagger_best.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/dagger_gru_rescue_continue_20260404_231730/dagger_best.pt)
- 日志：
  - [dagger_gru_rescue_continue.log](/data/uavlab/multi-uav-pursuit2/dagger_gru_rescue_continue.log)
- 结果：
  - `Capture 48.4% / Goal 28.1% / Landed 17.2% / Timeout 6.2%`

这是**目前 actor 的最好外测结果**。

### 15.7 一个重要对照

我还试了把 expert 混合率再降一点，看看 actor 少靠老师扶着会不会更强：

- [dagger_gru_rescue_continue2.log](/data/uavlab/multi-uav-pursuit2/dagger_gru_rescue_continue2.log)
- 结果反而退化成：
  - `Capture 40.6% / Goal 32.8% / Landed 17.2% / Timeout 9.4%`

这说明当前阶段：

- 老师扶着太多当然不行
- 但扶得太少也不行
- 当前最好的点，反而是在一个中间强度

### 15.8 现在最可信的新结论

到这一步，已经可以比前面更准确地下结论：

1. actor 学不会，不是因为“专家动作空间和网络输出空间不一致”。
2. actor 学不会，也不只是单纯 `MSE` 或 `logprob` 权重没调好。
3. 真正有效的提升，来自：
   - **在线 DAgger**
   - **recurrent actor**
   - **在危险低空状态下让专家短时接管，再把这些纠偏状态教回 actor**

### 15.9 当前数值结论

如果只看目前 actor 的最好外测成绩：

- 旧纯离线 BC：`18% capture`
- 老 feedforward DAgger 最好：`约 42% capture`
- 当前最好 recurrent + rescue continuation：`48.4% capture`

所以现在这条线已经证明：

- actor 不是完全学不会专家
- 只是要想学得更像专家，**必须教它“自己跑歪以后怎么被拉回来”**
- 而不是只喂离线成功轨迹

### 15.10 当前还没做到的事

虽然已经拉到 `48.4%`，但还没到用户想要的 `60%+`。

所以接下来最值得继续做的，不是推翻这条路，而是继续在这条路上调：

- recurrent actor 的时序学习
- rescue / success-only / expert mix 的平衡
- 让 `goal` 和 `landed` 之间的 tradeoff 再往更优的方向走

### 15.11 2026-04-05 凌晨继续对照后的最新结论

在 `48.4%` 这版最好 checkpoint 基础上，我又继续做了 3 轮很有针对性的 continuation probe。

#### A. 近 goal rescue

- 日志：
  - [dagger_gru_goalrescue.log](/data/uavlab/multi-uav-pursuit2/dagger_gru_goalrescue.log)
- 配置要点：
  - 在 `48.4%` 最优版基础上，额外加 `goal_rescue_radius=1.20`
  - 仍保留 `altitude_rescue_threshold=0.35`
- 结果：
  - `Capture 42.2%`
  - `Goal 34.4%`
  - `Landed 17.2%`
  - `Timeout 6.2%`

解释：

- 这刀没有砍对
- 训练内看起来很猛，`wave capture` 甚至到 `51.6%`
- 但外测明显退了
- 说明它更像是在教 actor “依赖专家兜底近 goal 状态”，而不是自己真正学会守 goal

#### B. 完全沿用当前最好设置，再滚一轮

- 日志：
  - [dagger_gru_rescue_same.log](/data/uavlab/multi-uav-pursuit2/dagger_gru_rescue_same.log)
- 做法：
  - 从 `48.4%` 最优版继续 1 wave
  - 保持和当前最好设置基本一致
- 结果：
  - `Capture 43.8%`
  - `Goal 28.1%`
  - `Landed 21.9%`
  - `Timeout 6.2%`

解释：

- 这说明当前最好设置继续往下滚，并不会自然继续涨
- 更像已经碰到一个局部平台

#### C. 增大 GRU 训练序列长度

- 日志：
  - [dagger_gru_seq16.log](/data/uavlab/multi-uav-pursuit2/dagger_gru_seq16.log)
- 做法：
  - 把 `actor_rnn_train_seq_len` 从 `8` 提到 `16`
- 结果：
  - `Capture 42.2%`
  - `Goal 32.8%`
  - `Landed 14.1%`
  - `Timeout 10.9%`

解释：

- 更长时间窗确实让 `landed` 更低了
- 但同时 `goal` 和 `timeout` 变差，最终 `capture` 没提升
- 说明“更长记忆”本身不是当前瓶颈的直接答案

### 15.12 截至目前最可信的最好结果

到 2026-04-05 凌晨这一步为止，actor 的最好外测结果仍然是：

- [dagger_gru_rescue_continue.log](/data/uavlab/multi-uav-pursuit2/dagger_gru_rescue_continue.log)
- [dagger_best.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/dagger_gru_rescue_continue_20260404_231730/dagger_best.pt)

对应指标：

- `Capture 48.4%`
- `Goal 28.1%`
- `Landed 17.2%`
- `Timeout 6.2%`

我还额外把它单独保存在这里，防止后续实验覆盖：

- [dagger_best_preserved_20260405_gru48_goal28_landed17.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/dagger_best_preserved_20260405_gru48_goal28_landed17.pt)

### 15.13 截至目前的最新判断

现在已经能更清楚地说：

1. recurrent actor 这条线是对的，因为它把 actor 从 `42%` 左右继续推到了 `48.4%`。
2. 但当前最主要的剩余短板，已经不是单纯 `landed`，而是：
   - `goal` 仍然偏高
   - 一旦为压 `goal` 直接上强 rescue，又容易让外测退化
3. 所以当前最优点，处在一个比较微妙的平衡：
   - 够 aggressive，能抓到更多
   - 但还没 aggressive 到既能抓又能稳稳守住 goal

一句更直白的话：

**现在 actor 已经从“不会学专家”走到了“学到一大半”，但离专家的最后一截差距，主要卡在“关键时刻到底该更狠去堵，还是更稳别摔”这个 tradeoff 上。**

### 15.14 2026-04-05 凌晨补做的两组“温和加权”对照

为了确认是不是还能靠小改权重继续往上推，我又补了两组更温和的 continuation：

#### D. near-goal weighting（只放大靠近守区样本的监督权重）

- 日志：
  - [dagger_gru_goalweight.log](/data/uavlab/multi-uav-pursuit2/dagger_gru_goalweight.log)
- 关键设置：
  - `online_goal_weight_alpha=2.0`
  - `online_goal_weight_radius=2.0`
  - 保留 `altitude_rescue_threshold=0.35`
- 结果：
  - `Capture 42.2%`
  - `Goal 31.2%`
  - `Landed 20.3%`
  - `Timeout 6.2%`

结论：

- 这条线没有超过当前最好结果
- 说明“只靠把 near-goal 状态权重调大”并不能自动学出更好的守 goal 行为

#### E. disagreement weighting（只放大 actor 和 expert 分歧大的样本）

- 日志：
  - [dagger_gru_disagree.log](/data/uavlab/multi-uav-pursuit2/dagger_gru_disagree.log)
- 关键设置：
  - `online_disagreement_weight_alpha=4.0`
  - 其他保持当前最好版附近的设置
- 结果：
  - `Capture 43.8%`
  - `Goal 29.7%`
  - `Landed 17.2%`
  - `Timeout 9.4%`

结论：

- 这条线也没有超过 `48.4%`
- 说明“优先学 actor 与 expert 差得最远的状态”在当前阶段也不是决定性提升点

### 15.15 到这一步的更严格结论

截至现在，已经可以更严格地说：

- `GRU + success-only + 低空 rescue continuation` 是目前已验证最有效的主线
- 但在这个主线附近继续做“小权重微调”，已经开始明显进入收益递减
- 当前最好 checkpoint 仍然是：
  - [dagger_best_preserved_20260405_gru48_goal28_landed17.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/dagger_best_preserved_20260405_gru48_goal28_landed17.pt)
- 最好指标仍然是：
  - `Capture 48.4% / Goal 28.1% / Landed 17.2% / Timeout 6.2%`

这意味着下一步如果还想明显往 `60%+` 推，可能需要：

- 不是继续磨同类小系数
- 而是引入一个更有信息量、但仍不破坏“端到端输出推力和角速度”这个论文主线的小结构改动

## 15. 后续排查更新：为什么“输出就是推力和角速度”还是学不出来

这部分是 2026-04-04 当天下午继续深入排查后得到的新结论。

### 15.1 已确认：动作空间不是主问题

专家数据里存的不是别的东西，而是已经映射到训练动作空间里的 `PIDrate raw action`。

也就是：

- 专家先算出 `t + omega`
- 再经过 `t_omega_to_pidrate_raw(...)`
- 最后存进 `action_raw`

这个动作和 actor 在 RL/BC 里输出的动作空间是一致的。

所以当前问题不是：

- “专家输出和 actor 输出不是一个空间”
- 或者“专家是速度控制，actor 是推力角速度控制”

这点已经对齐。

### 15.2 真正的核心问题：分布偏移，不是标签空间错位

后面我继续做了两组非常关键的对比：

1. 离线 BC 在专家成功轨迹上的拟合误差  
   全量 `MSE-only BC` 训练到最后，`bc_action_mse ≈ 0.00088`。

2. actor 自己 rollout 时，它访问到的状态上，再用专家给标签做在线监督  
   第一轮在线 DAgger 时，actor-state 上的 `mse` 直接变成了大约 `0.037`。

这两个数字放在一起，说明了很关键的一点：

- actor 在“专家本来就走得很顺”的状态上，其实已经能学得挺像
- 但 actor 一旦自己 rollout 稍微偏掉，就会进入一堆**离线专家数据里很少出现的状态**
- 一进入这些状态，actor 和专家动作的差距就立刻放大一个数量级

所以现在最核心的根因不是：

- 专家动作空间不对
- actor 完全没法表示专家动作

而是：

- **离线 BC 只学会了专家状态分布**
- **没有学会 actor 自己跑歪之后该怎么纠偏**

### 15.3 新增的在线 DAgger 训练链

为了解这个问题，我新增了一个在线纠偏脚本：

- [scripts/train_actor_dagger.py](/data/uavlab/multi-uav-pursuit2/scripts/train_actor_dagger.py)

它做的事是：

1. 先加载当前 actor checkpoint
2. 让 actor 自己在 Isaac 里跑
3. 每一步都同时让 batched expert 在**同一个当前状态**上给出标准动作
4. 用这些 actor 真正访问到的状态做在线模仿更新

大白话说：

- 以前是“只看老师的标准录像”
- 现在是“让学生自己上场，老师在场边实时纠错”

### 15.4 DAgger 里做过的工程修复

为了把这条在线链跑稳，我还修了两个坑：

1. checkpoint 加载设备错位  
   在 [omni_drones/learning/mappo.py](/data/uavlab/multi-uav-pursuit2/omni_drones/learning/mappo.py#L795) 里，把 `load_state_dict()` 改成会把加载进来的 `actor_params` 自动搬到当前策略设备，避免 CPU/GPU 参数混杂。

2. 在线缓存 observation 时的隐式共享存储问题  
   在 [scripts/train_actor_dagger.py](/data/uavlab/multi-uav-pursuit2/scripts/train_actor_dagger.py#L179) 里，对缓存进 BC buffer 的 observation/aux/action 都做了显式深拷贝，避免 Isaac 后续 step 原地改写底层 buffer，触发 autograd 的 inplace 报错。

### 15.5 在线 DAgger 的结果

下面这些结果都是 **纯 actor rollout 的真实 Isaac batch 评测**，不是混入专家动作的训练波结果。

#### 基线：全量 MSE-only BC

- [bc_mse_full_eval_128.log](/data/uavlab/multi-uav-pursuit2/bc_mse_full_eval_128.log)
- `Capture 18%`
- `Goal zone 55%`
- `Landed 7%`
- `Timeout 20%`

#### 第一版 DAgger（偏激进）

训练产物：

- [dagger_final.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/dagger_train_128x2_20260404_152741/dagger_final.pt)

评测：

- [dagger_train_128x2_eval.log](/data/uavlab/multi-uav-pursuit2/dagger_train_128x2_eval.log)
- `Capture 30%`
- `Goal zone 18%`
- `Landed 42%`
- `Timeout 10%`

解释：

- 这版已经证明 **DAgger 方向是对的**
- 因为成功率从 `18%` 明显抬到了 `30%`
- 但它学得太猛，摔机大幅增加

#### 第二版 DAgger（更保守，过滤低空状态）

训练产物：

- [dagger_final.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/dagger_conservative_128x2_20260404_153038/dagger_final.pt)

评测：

- [dagger_conservative_128x2_eval.log](/data/uavlab/multi-uav-pursuit2/dagger_conservative_128x2_eval.log)
- `Capture 34%`
- `Goal zone 27%`
- `Landed 27%`
- `Timeout 12%`

解释：

- 这版比上一版更平衡
- 把 landed 从 `42%` 压回到了 `27%`
- 同时 capture 还继续升到 `34%`

#### 第三版 DAgger（继续保守微调）

训练产物：

- [dagger_final.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/dagger_conservative_continue_128x2_20260404_153238/dagger_final.pt)

评测：

- [dagger_conservative_continue_eval.log](/data/uavlab/multi-uav-pursuit2/dagger_conservative_continue_eval.log)
- `Capture 40%`
- `Goal zone 20%`
- `Landed 23%`
- `Timeout 16%`

这是到目前为止**最好的 actor 结果**。

#### 第四版 DAgger（再往下滚两轮）

训练产物：

- [dagger_final.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/dagger_conservative_more_128x2_20260404_153434/dagger_final.pt)

评测：

- [dagger_conservative_more_eval.log](/data/uavlab/multi-uav-pursuit2/dagger_conservative_more_eval.log)
- `Capture 38%`
- `Goal zone 27%`
- `Landed 22%`
- `Timeout 12%`

解释：

- 和第三版相比略有回落
- 说明这条“同一种小步 DAgger”继续往下滚，已经开始接近平台，不再是稳步线性增长

### 15.6 到目前为止最可信的新结论

现在可以更有把握地说：

1. **专家策略本身没问题**
   - 专家成功率大约 `66%`
   - 这点之前已经验证过

2. **动作空间对齐也没问题**
   - 专家给 actor 的标签就是训练动作空间里的 `PIDrate raw action`

3. **离线 BC 学不会，不是因为“推力角速度学不会”**
   - 而是因为只学了专家状态分布
   - 没学会 student 自己 rollout 偏离之后的恢复

4. **DAgger 已经证明这个根因判断是对的**
   - 因为一上 DAgger，capture 从 `18%` 提到了 `40%`

5. **当前新的瓶颈变成了 stability / recovery**
   - 也就是：
   - 已经更会追、更会堵了
   - 但还没把“越追越稳”学到能接近专家 `66%` 的程度

### 15.7 当前最推荐的下一步

现在最值得继续做的，不是回到老版纯离线 BC，而是沿着下面这条线：

1. 保留 `DAgger` 主框架  
2. 保留更保守的在线设置：
   - 小学习率
   - 较高 expert mixing
   - 过滤贴地状态
3. 再加一层“稳定性锚定”
   - 最值得尝试的是 **在线 DAgger + 离线专家成功轨迹 replay 混合训练**
   - 也就是每次在线纠偏时，再穿插一部分原始成功专家数据
   - 这样可以减少 actor 漂到不稳定控制区域

一句最直白的话：

**现在已经基本确认：真正缺的不是“更好的专家动作”，而是“让 actor 在自己会跑歪的状态上也能持续被专家拉回来”的训练机制。**

### 15.8 新进展：上一时刻动作 conditioning（2026-04-05 凌晨）

这轮继续调试时，围绕“端到端输出 `推力 + 角速度` 不变”的前提，又试了一条更贴近低层控制连续性的路：

- 给 actor 增加 `prev_action` 输入，也就是“上一时刻实际执行的动作”
- 但不是直接把它和 obs 拼到 encoder 输入里重训整网
- 而是在 actor 末端加一个**零初始化的 residual conditioner**
- 让它只学“基于上一时刻动作，对当前动作分布做一点低层修正”

对应代码改动主要在：
- [omni_drones/learning/mappo.py](/data/uavlab/multi-uav-pursuit2/omni_drones/learning/mappo.py)
- [scripts/train_actor_dagger.py](/data/uavlab/multi-uav-pursuit2/scripts/train_actor_dagger.py)
- [scripts/eval_policy_batch.py](/data/uavlab/multi-uav-pursuit2/scripts/eval_policy_batch.py)

具体接法是：
- 新增 `("agents", "prev_action")`
- reset 时置零
- 每步 step 后，把**真正执行的动作**写回 `next.prev_action`
- checkpoint 里新增：
  - `actor_has_prev_action_conditioning`
  - `actor_prev_action_condition_hidden_dim`

#### 15.8.1 直接整网继续训：失败

我先试了最直接的版本：
- 从当前最好 checkpoint 继续
- 开启 `prev_action_conditioning`
- 其余设置基本和 `48.4%` 那版一致

日志：
- [dagger_gru_prevact_probe.log](/data/uavlab/multi-uav-pursuit2/dagger_gru_prevact_probe.log)

结果：
- `Capture 35.9%`
- `Goal 42.2%`
- `Landed 18.8%`
- `Timeout 3.1%`

这个结果说明：
- “上一时刻动作”这条信息本身不是坏的
- 但**整网一起继续训**，会把原来已经学到的追捕几何能力带偏
- 表现上更像“更保守、更跟手了”，但更容易漏 goal

#### 15.8.2 只训练新增的小分支：有效

接着我做了一个更保守的版本：
- 还是开 `prev_action_conditioning`
- 但**冻结原来的 actor/GRU 主体**
- 只训练新加的 `prev_action_conditioner`

这个开关在 [scripts/train_actor_dagger.py](/data/uavlab/multi-uav-pursuit2/scripts/train_actor_dagger.py) 里叫：
- `--train_prev_action_only true`

这轮会打印出：
- `trainable=33536`
- `frozen=643224`

说明它确实是在“只让一个小低层修正分支学习”，而不是重写整个策略。

第一次 probe 的结果明显变好了：
- 日志：[dagger_gru_prevact_frozen_probe.log](/data/uavlab/multi-uav-pursuit2/dagger_gru_prevact_frozen_probe.log)
- checkpoint 目录：[dagger_gru_prevact_frozen_probe_20260405_022420](/data/uavlab/multi-uav-pursuit2/checkpoints/dagger_gru_prevact_frozen_probe_20260405_022420)
- 内置 eval 结果：
  - `Capture 51.6%`
  - `Goal 23.4%`
  - `Landed 17.2%`
  - `Timeout 7.8%`

和上一版最好结果相比：
- 旧最好：[dagger_best_preserved_20260405_gru48_goal28_landed17.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/dagger_best_preserved_20260405_gru48_goal28_landed17.pt)
  - `48.4 / 28.1 / 17.2 / 6.2`
- 新最好：[dagger_best_preserved_20260405_gru52_goal23_landed17.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/dagger_best_preserved_20260405_gru52_goal23_landed17.pt)
  - `51.6 / 23.4 / 17.2 / 7.8`

大白话总结：
- `landed` 没变差
- `goal` 明显下降
- `capture` 往上推了一截

这说明：
**上一时刻动作这条信息是有价值的，但更适合作为“低层补丁”，不适合一上来把整网一起带着跑。**

#### 15.8.3 再往下推：开始回落

我又试了两轮 continuation：

1. 同学习率继续滚一轮  
日志：[dagger_gru_prevact_frozen_continue1.log](/data/uavlab/multi-uav-pursuit2/dagger_gru_prevact_frozen_continue1.log)  
结果：`Capture 48.4% / Goal 25.0% / Landed 20.3% / Timeout 6.2%`

2. 从 `51.6%` 那版出发，把学习率降到 `5e-5`  
日志：[dagger_gru_prevact_frozen_lowlr.log](/data/uavlab/multi-uav-pursuit2/dagger_gru_prevact_frozen_lowlr.log)  
结果：`Capture 46.9% / Goal 25.0% / Landed 20.3% / Timeout 7.8%`

所以目前这个方向的阶段结论是：
- 第一次引入这个小分支，确实有提升
- 但继续沿完全同样的办法往下滚，很快又开始回落

#### 15.8.4 当前最可信的新结论

现在可以把新的判断写得更具体一些：

1. **端到端输出 `推力 + 角速度` 这条论文主线没有问题**
   - 新提升仍然是在这条主线内部得到的
   - 没有换成 waypoint policy

2. **actor 学不会专家，不是因为“低层动作本身学不了”**
   - 否则 `prev_action` 这条低层修正支路不可能带来 `48.4% -> 51.6%` 的提升

3. **更像是 actor 缺少“动作连续性 / 控制手感”的一部分信息**
   - 专家策略本来就是连续决策
   - 单纯只看当前 obs，actor 很难完全还原这种低层闭环手感

4. **但这个信息更适合被“局部使用”**
   - 直接拿它去重训整网，会伤到原来的追捕几何能力
   - 只让小分支学这个修正，效果更好

5. **当前最优 checkpoint 更新为：**
   - [dagger_best_preserved_20260405_gru52_goal23_landed17.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/dagger_best_preserved_20260405_gru52_goal23_landed17.pt)

#### 15.8.5 补充说明：独立 batch eval 仍有 Isaac 侧不稳定

我尝试对这版 `51.6%` checkpoint 再跑一次独立的 [eval_policy_batch.py](/data/uavlab/multi-uav-pursuit2/scripts/eval_policy_batch.py)：
- 日志：[dagger_gru_prevact_frozen_probe_eval128.log](/data/uavlab/multi-uav-pursuit2/dagger_gru_prevact_frozen_probe_eval128.log)

这次没有拿到可靠摘要，因为 Isaac/USD 层在启动后早期报错并段错误退出。  
所以**目前最可信的评测数字，仍然是训练脚本内置 eval 的 `64` 局结果**，而不是这次独立 batch eval。

### 15.9 `prev_action` 小分支后的进一步 continuation

在拿到 `51.6%` 这个新最好点之后，我又继续试了几种“沿同一条线往前推”的小改法，结果都没有再超过它。

#### 15.9.1 同设置继续滚一轮

日志：
- [dagger_gru_prevact_frozen_continue1.log](/data/uavlab/multi-uav-pursuit2/dagger_gru_prevact_frozen_continue1.log)

结果：
- `Capture 48.4%`
- `Goal 25.0%`
- `Landed 20.3%`
- `Timeout 6.2%`

结论：
- `51.6%` 这版不是“多滚几轮就继续涨”
- 更像是第一次把低层修正支路接上时有一波增益
- 再沿完全同样设置往前推，就会开始回落

#### 15.9.2 更小学习率

日志：
- [dagger_gru_prevact_frozen_lowlr.log](/data/uavlab/multi-uav-pursuit2/dagger_gru_prevact_frozen_lowlr.log)

设置变化：
- `bc_lr: 1e-4 -> 5e-5`

结果：
- `Capture 46.9%`
- `Goal 25.0%`
- `Landed 20.3%`
- `Timeout 7.8%`

结论：
- 这条线的回落不是单纯因为“学习率太大”
- 降学习率也没把 `51.6%` 保住

#### 15.9.3 近 goal rescue

日志：
- [dagger_gru_prevact_goalrescue.log](/data/uavlab/multi-uav-pursuit2/dagger_gru_prevact_goalrescue.log)

设置变化：
- 在 `prev_action + train_prev_action_only` 的基础上增加：
  - `goal_rescue_radius = 1.0`

结果：
- `Capture 46.9%`
- `Goal 28.1%`
- `Landed 15.6%`
- `Timeout 9.4%`

结论：
- 近 goal 的专家接管虽然把 `landed` 压低了一点
- 但没有把 `goal` 压下去，反而把 `capture` 拉低了
- 所以当前这条线不值得继续往这个方向加码

#### 15.9.4 disagreement weighting

日志：
- [dagger_gru_prevact_disagree.log](/data/uavlab/multi-uav-pursuit2/dagger_gru_prevact_disagree.log)

设置变化：
- 在 `prev_action + train_prev_action_only` 的基础上增加：
  - `online_disagreement_weight_alpha = 1.0`

结果：
- `Capture 46.9%`
- `Goal 25.0%`
- `Landed 21.9%`
- `Timeout 6.2%`

结论：
- 让小分支更关注“和专家差得大的状态”没有带来额外收益
- 当前这版 `prev_action` 补丁更像需要一小步对齐，而不是再做强重加权

#### 15.9.5 当前最新阶段判断

现在这一阶段可以比较明确地总结成：

1. `prev_action` 小分支是目前**唯一一个把 actor 从 `48.4%` 再往上推到 `51.6%`** 的新结构改动
2. 但这条线当前也已经出现明显的平台
3. 继续沿同一套参数做 continuation、low-lr、goal rescue、disagreement weighting，都没有超过 `51.6%`
4. 因此当前最优 checkpoint 仍然是：
   - [dagger_best_preserved_20260405_gru52_goal23_landed17.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/dagger_best_preserved_20260405_gru52_goal23_landed17.pt)

## 16. RL Warm-Start 接训进展（2026-04-05 晚）

### 16.1 为什么开始转 RL

到 `51.6%` 这一版之后，纯模仿学习已经明显进入平台：

- 最好的 actor 已经不是“乱飞”，而是已经学会了任务的大框架
- 但继续磨 BC / DAgger 小权重，收益开始很小
- 这时候更适合让 RL 去学剩下那部分：
  - rollout 偏离后的纠偏
  - 临门一脚的收口
  - 真实闭环里的稳定性

所以后面开始转成：

- 用当前最好 actor 做 warm-start
- 再接 PPO / MAPPO 微调

### 16.2 RL 接训前修掉的关键问题

在真正起 RL 之前，代码里其实有几个会直接把 warm-start 卡死的问题。

#### 16.2.1 `train.py` 不能自动重建对的 actor 结构

当前最好 checkpoint 不是普通 actor，它带了：

- `actor_rnn_hidden_size = 128`
- `prev_action_conditioning`
- `bc_aux`

但原来的 [scripts/train.py](/data/uavlab/multi-uav-pursuit2/scripts/train.py) 不会根据 checkpoint 自动把这些结构重新打开，容易出现：

- checkpoint 里有这些模块
- 训练时却按普通 actor 构建
- 最后 `load_state_dict` 虽然表面能跑，但结构实际是错的

现在已经补上自动识别逻辑：

- `checkpoint_has_bc_aux(...)`
- `checkpoint_actor_rnn_hidden_size(...)`
- `checkpoint_prev_action_condition_hidden_size(...)`

并在创建 policy 前自动打开对应配置。

#### 16.2.2 recurrent PPO 在 warm-start actor 上真的有 bug

原来一接 GRU actor 做 PPO 更新，就会在 `is_init` 的维度扩展上炸掉。

大白话说就是：

- actor 是按 `[batch, time, agent, ...]` 在跑
- 但 `is_init` 被扩成了 `[batch, time, agent]`
- 某些地方其实需要的是 `[batch, time, agent, 1]`

结果一到 recurrent PPO update 就会报维度错。

这个现在已经在 [omni_drones/learning/mappo.py](/data/uavlab/multi-uav-pursuit2/omni_drones/learning/mappo.py) 里修掉了。

#### 16.2.3 recurrent actor 的熵路径也会在 PPO 里卡住

另一个更隐蔽的问题是：

- `Actor.forward(..., eval_action=True)` 原来在 recurrent 分支里还返回显式 entropy
- 这会和 `vmap + TensorDictModule` 的组合发生不兼容

现在的改法是：

- recurrent PPO update 这条路不再强依赖 actor 显式返回 `action_entropy`
- 直接用 `log_probs_new` 去构造 entropy bonus

这样 recurrent PPO 才真正能稳定更新。

#### 16.2.4 关闭 curriculum 后，目标速度之前其实没真的对齐

这个点很关键，也很容易误会。

原来代码里即使：

- `task.curriculum.enabled=false`

环境初始化时还是会把 stage 切到最后一阶段，并把：

- `current_target_speed`

偷偷设成 stage 里的默认值，也就是 `1.0`

所以你表面上以为自己关了 curriculum、把双方速度设成了 `1.5`

但实际上目标仍然可能按 `1.0` 在跑。

现在已经改成：

- 如果 `curriculum_enabled = false`
- 那 `current_target_speed = base_target_speed`

这样显式覆盖的速度才真的生效。

### 16.3 对齐设置到底是什么

RL warm-start 这轮对齐的是“专家/DAgger 最后一阶段的困难随机分布”，所以参数用了：

- `task.use_eval = 0`
- `task.curriculum.enabled = false`
- `task.env.max_episode_length = 1000`
- `task.env.num_envs = 512`
- `task.v_drone = 1.5`
- `task.v_prey = 1.0`

这里最容易困惑的是最后两项。

当前环境里：

```text
base_target_speed = task.v_drone * task.v_prey
```

所以：

- `task.v_drone = 1.5`
- `task.v_prey = 1.0`

在这版代码下，实际含义是：

- 追捕者速度上限 `1.5 m/s`
- 目标速度上限也对齐成 `1.5 m/s`

也就是说，这里 `task.v_prey` 现在还是“相对系数”，不是直接的绝对速度。

### 16.4 已经成功跑通的 aligned smoke test

短烟雾测试已经跑通，输出目录在：

- [HideAndSeek_20260405_170406](/data/uavlab/multi-uav-pursuit2/checkpoints/HideAndSeek_20260405_170406)

日志在：

- [outputs/2026-04-05/17-03-59/train.log](/data/uavlab/multi-uav-pursuit2/outputs/2026-04-05/17-03-59/train.log)

这轮 smoke 的设置是：

- 从 [dagger_best_preserved_20260405_gru52_goal23_landed17.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/dagger_best_preserved_20260405_gru52_goal23_landed17.pt) warm-start
- `total_frames = 262144`
- `task.use_eval = 0`
- `task.curriculum.enabled = false`
- `task.v_drone = 1.5`
- `task.v_prey = 1.0`
- `task.env.max_episode_length = 1000`
- `task.env.num_envs = 512`

这轮已经确认：

- 能正常进入 rollout
- 能正常进入 PPO update
- 能保存中间 checkpoint
- 能正常保存 `checkpoint_final.pt`

对应产物：

- [checkpoint_32768.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/HideAndSeek_20260405_170406/checkpoint_32768.pt)
- [checkpoint_final.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/HideAndSeek_20260405_170406/checkpoint_final.pt)

### 16.5 正式 RL fine-tune 已经起跑

正式 run 已经拉起来，当前目录是：

- checkpoint 目录：[HideAndSeek_20260405_171726](/data/uavlab/multi-uav-pursuit2/checkpoints/HideAndSeek_20260405_171726)
- TensorBoard run：[HideAndSeek_20260405_171726](/data/uavlab/multi-uav-pursuit2/runs/HideAndSeek_20260405_171726)
- 训练日志：[outputs/2026-04-05/17-17-19/train.log](/data/uavlab/multi-uav-pursuit2/outputs/2026-04-05/17-17-19/train.log)

这轮正式设置是：

- warm-start checkpoint：
  - [dagger_best_preserved_20260405_gru52_goal23_landed17.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/dagger_best_preserved_20260405_gru52_goal23_landed17.pt)
- `total_frames = 1000000000`
- `eval_interval = 50`
- `save_interval = 50`
- `task.use_eval = 0`
- `task.curriculum.enabled = false`
- `task.v_drone = 1.5`
- `task.v_prey = 1.0`
- `task.env.max_episode_length = 1000`
- `task.env.num_envs = 512`

目前它已经确认越过：

- `Eval at 32768 steps`
- 并保存了 [checkpoint_32768.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/HideAndSeek_20260405_171726/checkpoint_32768.pt)

### 16.6 当前正式 RL 的最早期信号

我已经直接从 TensorBoard event 文件里读到了这轮正式 run 的早期值。

在大约 `1048576` steps 时：

- `train/stats.success = 0.00287`
- `train/stats.any_landed = 0.9971`
- `train/stats.return = -26.37`
- `train/stats.goal_reached = 0.0`
- `drone/entropy = 5.75`
- `drone/actor_grad_norm = 1.45`
- `drone/value_loss = 0.0846`

第一次 eval (`32768` steps) 时：

- `eval/stats.success = 0.0`
- `eval/stats.any_landed = 0.0137`
- `eval/stats.return = -4.90`

大白话解释：

- 训练 rollout 这时候还比较粗糙，`landed` 仍然很高
- 但 deterministic eval 没有一上来全摔
- 说明 warm-start 起点是活的，不是“动作一放进 PPO 就完全坏掉”

#### 16.6.1 现在该怎么理解这个早期结果

不要把这 `1M` steps 左右的信号看成“RL 已经失败”。

更合理的理解是：

- imitation / DAgger 给了一个不错的初始结构
- PPO 刚接手时，策略分布和 critic 还在重新适应
- 早期 rollout 变粗糙是正常现象

当前最重要的是：

- 这条 RL warm-start 链已经能稳定跑
- 后面才值得继续看它会不会逐步把 `goal` 和 `landed` 往下压、把 `success` 往上拉

### 16.7 第一轮正式 RL 为什么后来发散了

第一轮正式 RL run 是：

- [HideAndSeek_20260405_171726](/data/uavlab/multi-uav-pursuit2/checkpoints/HideAndSeek_20260405_171726)

它一开始能正常训练，但后来在大约 `22.97M` steps 之后开始明显数值发散。

关键证据是：

- `drone/actor_log_std_mean` 从接近 `0` 一路涨到接近上限
- `drone/entropy` 后期直接炸到异常量级
- `drone/actor_grad_norm` 后期变成超大值甚至 `inf`
- `train/stats.return` 后期明显变差
- `eval/stats.success` 全程没起来

更准确地说，这次不是“奖励突然坏了”，而是 PPO 稳定性先坏了。

#### 16.7.1 根因

这轮后来定位到两个关键问题叠在一起：

1. recurrent PPO 里的 entropy bonus 用的是不合适的近似  
   之前为了绕开 recurrent `TensorDict/vmap` 的兼容问题，代码把 PPO 里的 entropy bonus 近似成了基于 `log_prob` 的项。  
   这在 warm-start actor 上会让熵相关量失真，并放大数值不稳定。

2. warm-start actor 的探索强度太大  
   原来正式 RL 用的是：
   - `entropy_schedule.enabled = true`
   - `hold_coef = 0.01`
   - `actor.lr = 5e-4`
   - `ppo_epochs = 4`
   - `actor.log_std_max = 0.5`

   对从 DAgger checkpoint 接上的 actor 来说，这套更新太猛了，容易把专家学出来的策略分布迅速冲散。

#### 16.7.2 这次的最小修复

现在已经做了两类最小修复：

1. 熵 bonus 改成从 actor 的 `log_std` 直接精确计算  
   对当前这类 `DiagGaussian` actor，动作分布熵本来就可以直接从 `log_std` 算出来，没必要继续用那个会失真的近似。

2. warm-start RL 改成更保守的稳定化配置  
   新的稳定化实验参数是：
   - `algo.actor.lr = 1e-4`
   - `algo.ppo_epochs = 2`
   - `algo.max_grad_norm = 5.0`
   - `algo.entropy_schedule.enabled = false`
   - `algo.entropy_coef = 0.001`
   - `algo.actor.log_std_max = 0.2`

大白话说就是：

- 学习率更小
- 每个 batch 少改几次
- 熵奖励更小
- 动作方差上限更低

这样做的目的不是“让 RL 一上来就变强”，而是先保证：

- 不把专家策略冲散
- 不再出现那种后期数值爆炸

### 16.8 新的稳定化 smoke 已通过

在上述修复后，新的稳定化 smoke run 已经跑通：

- [HideAndSeek_20260405_182135](/data/uavlab/multi-uav-pursuit2/checkpoints/HideAndSeek_20260405_182135)

日志：

- [outputs/2026-04-05/18-21-27/train.log](/data/uavlab/multi-uav-pursuit2/outputs/2026-04-05/18-21-27/train.log)

它已经正常完成并保存：

- [checkpoint_32768.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/HideAndSeek_20260405_182135/checkpoint_32768.pt)
- [checkpoint_final.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/HideAndSeek_20260405_182135/checkpoint_final.pt)

这轮 smoke 的早期信号是：

- `train/stats.success ≈ 0.0057`
- `train/stats.any_landed ≈ 0.9943`
- `train/stats.return ≈ -26.55`
- `drone/entropy ≈ 5.68`
- `drone/actor_log_std_mean ≈ 0.00055`
- `drone/actor_log_std_max ≈ 0.00178`
- `drone/effective_entropy_coef = 0.001`

这说明：

- 熵没有再像上一轮那样一路往上飙
- `log_std` 也没有再很快冲向上限
- 新配置至少在“最容易炸的前几步”里是稳的

### 16.9 新的正式稳定化 RL 已重启

当前新的正式稳定化 RL run 是：

- checkpoint 目录：[HideAndSeek_20260405_182329](/data/uavlab/multi-uav-pursuit2/checkpoints/HideAndSeek_20260405_182329)
- 日志：[outputs/2026-04-05/18-23-22/train.log](/data/uavlab/multi-uav-pursuit2/outputs/2026-04-05/18-23-22/train.log)
- TensorBoard run：[HideAndSeek_20260405_182329](/data/uavlab/multi-uav-pursuit2/runs/HideAndSeek_20260405_182329)

这轮用的还是：

- [dagger_best_preserved_20260405_gru52_goal23_landed17.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/dagger_best_preserved_20260405_gru52_goal23_landed17.pt)

但 RL 参数改成了上面那套更保守的稳定化设置。

当前它已经顺利越过：

- `Eval at 32768 steps`
- 并保存了 [checkpoint_32768.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/HideAndSeek_20260405_182329/checkpoint_32768.pt)

所以现在最新的判断是：

- 第一轮 RL 失败，主要是 PPO 稳定性问题，不是专家 actor 完全没用
- 现在已经做了最小必要修复
- 新的稳定化正式 RL 已经重新起跑
