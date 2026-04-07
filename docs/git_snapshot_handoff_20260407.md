# Git 快照交接文档（2026-04-07）

## 1. 这次快照是什么

这是一次围绕“多无人机追逃 + 专家策略 + 行为克隆/DAgger + RL warm-start”主线整理出来的代码快照。

这次快照不是单点小修，而是一个阶段性的工作集，主要覆盖了：

- 守区防守环境与奖励重构
- 专家策略设计与 Isaac 评测
- 专家数据采集与行为克隆（BC）
- DAgger 在线纠偏训练
- warm-start RL 接续训练与失败分析
- 一批配套说明文档

一句话说：

**这是“从专家策略到 BC/DAgger，再到 RL warm-start”的完整阶段版本。**

---

## 2. 这次快照的主线成果

### 2.1 专家策略

专家策略已经成型，并能在 Isaac Sim 下以真实追捕者动力学做 batched 评测。

你现在的专家策略不是随机追，而是：

- 先预测目标未来位置
- 再做三机角色分配
- 再摆出三角围捕阵型
- 最后在近距离切到收网模式

对应说明文档：

- [expert_strategy_explanation.md](/data/uavlab/multi-uav-pursuit2/docs/expert_strategy_explanation.md)

关键脚本：

- [expert_strategy_test.py](/data/uavlab/multi-uav-pursuit2/scripts/expert_strategy_test.py)
- [expert_isaac_eval.py](/data/uavlab/multi-uav-pursuit2/scripts/expert_isaac_eval.py)
- [expert_isaac_eval.sh](/data/uavlab/multi-uav-pursuit2/scripts/expert_isaac_eval.sh)

---

### 2.2 行为克隆 / DAgger

这条线已经完成了从：

- 专家成功轨迹采集
- 到离线 BC
- 再到在线 DAgger

的完整闭环。

目前最重要的认识是：

- 纯 BC 不足以把低层动作完整学会
- 在线 DAgger 比纯 BC 有明显提升
- 加了 `prev_action` 小分支的 recurrent DAgger，是目前最好的 imitation 结果

对应说明文档：

- [behavior_cloning_explanation.md](/data/uavlab/multi-uav-pursuit2/docs/behavior_cloning_explanation.md)
- [expert_bc_handoff_20260404.md](/data/uavlab/multi-uav-pursuit2/docs/expert_bc_handoff_20260404.md)

关键脚本：

- [train_expert_bc.py](/data/uavlab/multi-uav-pursuit2/scripts/train_expert_bc.py)
- [train_expert_bc.sh](/data/uavlab/multi-uav-pursuit2/scripts/train_expert_bc.sh)
- [train_actor_dagger.py](/data/uavlab/multi-uav-pursuit2/scripts/train_actor_dagger.py)
- [run_expert_bc_pipeline.sh](/data/uavlab/multi-uav-pursuit2/scripts/run_expert_bc_pipeline.sh)
- [eval_policy_batch.py](/data/uavlab/multi-uav-pursuit2/scripts/eval_policy_batch.py)

---

### 2.3 RL warm-start

这次已经验证了：

- 从 DAgger 最优 actor 接 PPO/MAPPO 是可以跑通的
- 但当前 warm-start RL 这条线**效果不理想**
- 最主要的问题是：PPO 把已有的 imitation 行为冲散了

对应分析文档：

- [rl_warmstart_failure_for_claude_20260407.md](/data/uavlab/multi-uav-pursuit2/docs/rl_warmstart_failure_for_claude_20260407.md)
- [rl_warmstart_analysis.md](/data/uavlab/multi-uav-pursuit2/docs/rl_warmstart_analysis.md)

---

## 3. 当前最重要的结果结论

### 3.1 当前最好的 imitation actor

当前最好 actor 是：

- [dagger_best_preserved_20260405_gru52_goal23_landed17.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/dagger_best_preserved_20260405_gru52_goal23_landed17.pt)

它的代表性结果大约是：

- `Capture 51.6%`
- `Goal 23.4%`
- `Landed 17.2%`
- `Timeout 7.8%`

这说明：

- 端到端输出推力/角速度这条 imitation 主线不是完全不行
- 但还没有推到 `60%+`

为了对比，也保留了一个稍早的最好点：

- [dagger_best_preserved_20260405_gru48_goal28_landed17.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/dagger_best_preserved_20260405_gru48_goal28_landed17.pt)

---

### 3.2 BC 的最终认识

最开始那版纯 `log_prob` BC 没学好，后来改成 `MSE` 为主以后明显更合理。

目前对 BC 的核心认识是：

- **只学专家最终动作，不够**
- actor 更难学的是“自己 rollout 偏离后如何回到专家行为”
- 所以才需要 DAgger

---

### 3.3 RL warm-start 的最终认识

从最好 DAgger actor 接 PPO 之后，目前最重要的结论不是“RL 完全没法用”，而是：

- 这条 warm-start RL 目前没有把成功率继续往上推
- 反而把原来的 imitation 能力冲淡了

当前最关键的失败 run 是：

- [HideAndSeek_20260405_183521](/data/uavlab/multi-uav-pursuit2/checkpoints/HideAndSeek_20260405_183521)
- 最终权重：[checkpoint_final.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/HideAndSeek_20260405_183521/checkpoint_final.pt)

这个权重也被纳入这次快照，原因不是它好，而是：

- **它是 warm-start RL 失败分析的关键证据**

---

## 4. 这次快照里最关键的代码改动模块

下面这些文件是下次回溯最值得优先看的。

### 4.1 环境与奖励

- [HideAndSeek.yaml](/data/uavlab/multi-uav-pursuit2/cfg/task/HideAndSeek.yaml)
- [hideandseek.py](/data/uavlab/multi-uav-pursuit2/omni_drones/envs/hide_and_seek/hideandseek.py)
- [hideandseek_deploy.py](/data/uavlab/multi-uav-pursuit2/omni_drones/envs/hide_and_seek/hideandseek_deploy.py)
- [hideandseek_envgen.py](/data/uavlab/multi-uav-pursuit2/omni_drones/envs/hide_and_seek/hideandseek_envgen.py)

这里面主要涉及：

- 守区防守任务定义
- curriculum
- landed 判定
- 奖励重配比
- 目标运动机制

---

### 4.2 actor / PPO / MAPPO

- [mappo.yaml](/data/uavlab/multi-uav-pursuit2/cfg/algo/mappo.yaml)
- [mappo.py](/data/uavlab/multi-uav-pursuit2/omni_drones/learning/mappo.py)
- [distributions.py](/data/uavlab/multi-uav-pursuit2/omni_drones/learning/modules/distributions.py)
- [networks.py](/data/uavlab/multi-uav-pursuit2/omni_drones/learning/modules/networks.py)

这里面主要涉及：

- actor/critic 结构
- recurrent actor
- BC 损失
- DAgger 需要的辅助输入
- PPO 熵项与稳定性
- raw action 记录与 log_std 监控

---

### 4.3 动作变换与控制链

- [transforms.py](/data/uavlab/multi-uav-pursuit2/omni_drones/utils/torchrl/transforms.py)
- [crazyflie.yaml](/data/uavlab/multi-uav-pursuit2/omni_drones/robots/assets/usd/crazyflie.yaml)
- [crazyflie_deploy.yaml](/data/uavlab/multi-uav-pursuit2/omni_drones/robots/assets/usd/crazyflie_deploy.yaml)

这里面最重要的是：

- actor raw action 到 `PIDrate` 的映射
- `torch.tanh(action)` 的 squash
- 推力和角速度的真实控制链

---

### 4.4 训练入口与评测

- [train.py](/data/uavlab/multi-uav-pursuit2/scripts/train.py)
- [eval.py](/data/uavlab/multi-uav-pursuit2/scripts/eval.py)
- [eval_policy_batch.py](/data/uavlab/multi-uav-pursuit2/scripts/eval_policy_batch.py)
- [train_deploy.py](/data/uavlab/multi-uav-pursuit2/scripts/train_deploy.py)
- [train_generator.py](/data/uavlab/multi-uav-pursuit2/scripts/train_generator.py)

这些文件决定了：

- 正式 RL 是怎么起跑的
- warm-start checkpoint 怎么载入
- eval 怎么统计
- video 怎么生成

---

### 4.5 专家策略与专家数据

- [expert_strategy_test.py](/data/uavlab/multi-uav-pursuit2/scripts/expert_strategy_test.py)
- [expert_isaac_eval.py](/data/uavlab/multi-uav-pursuit2/scripts/expert_isaac_eval.py)
- [expert_isaac_eval.sh](/data/uavlab/multi-uav-pursuit2/scripts/expert_isaac_eval.sh)
- [train_expert_bc.py](/data/uavlab/multi-uav-pursuit2/scripts/train_expert_bc.py)
- [train_actor_dagger.py](/data/uavlab/multi-uav-pursuit2/scripts/train_actor_dagger.py)

---

## 5. 这次一并入库的重要权重文件

因为 `.gitignore` 里默认忽略了 `*.pt`，这次是**有意识地强制把关键权重纳入版本控制**，方便以后回溯。

当前建议保留的关键权重有：

### 5.1 最佳 imitation / DAgger actor

- [dagger_best_preserved_20260405_gru52_goal23_landed17.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/dagger_best_preserved_20260405_gru52_goal23_landed17.pt)

用途：

- 这是目前最好的 warm-start actor 起点
- 后续所有 RL warm-start 对比，都应该优先以它为基线

### 5.2 次优 imitation actor（对照）

- [dagger_best_preserved_20260405_gru48_goal28_landed17.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/dagger_best_preserved_20260405_gru48_goal28_landed17.pt)

用途：

- 用来对比 `51.6%` 那版到底提升了什么
- 也是一个可回退的稳定参考点

### 5.3 TP 预测网络

- [tp_only_1690959872.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/HideAndSeek_20260403_001241/tp_only_1690959872.pt)

用途：

- 专家策略和后续 BC/DAgger 都反复使用了这个 TP 权重
- 如果以后要重训 actor，但不想重训 TP，这个文件很关键

### 5.4 关键失败 RL final checkpoint

- [checkpoint_final.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/HideAndSeek_20260405_183521/checkpoint_final.pt)

用途：

- 不是“最好模型”
- 而是 warm-start RL 失败分析的重要对照样本

---

## 6. 哪些重要东西没有入库

为了避免仓库被运行产物污染，这次**不会**把下面这些内容全部推上去：

- `runs/`
- `eval_videos/`
- 大量 `outputs/.../train.log`
- 大量中间 checkpoints
- `backups/`
- 本地工具目录，例如 `.agents/`、`.codex/`、`.manual_train_watch/`

但这些东西不是不重要，而是太大、太杂，不适合当 Git 主体版本内容。

如果你下次回溯，需要重点记住这些“仓库外的重要路径”：

### 6.1 关键 warm-start RL 失败 run

- `runs/HideAndSeek_20260405_183521`
- `outputs/2026-04-05/18-35-15/train.log`

### 6.2 关键 DAgger 成功记录

- `dagger_gru_prevact_frozen_probe.log`

### 6.3 主要交接文档

- [expert_bc_handoff_20260404.md](/data/uavlab/multi-uav-pursuit2/docs/expert_bc_handoff_20260404.md)
- [rl_warmstart_failure_for_claude_20260407.md](/data/uavlab/multi-uav-pursuit2/docs/rl_warmstart_failure_for_claude_20260407.md)
- [behavior_cloning_explanation.md](/data/uavlab/multi-uav-pursuit2/docs/behavior_cloning_explanation.md)
- [expert_strategy_explanation.md](/data/uavlab/multi-uav-pursuit2/docs/expert_strategy_explanation.md)

---

## 7. 下次回溯时最推荐的阅读顺序

如果你下次很久以后回来，不想重新摸索，建议按这个顺序看：

### 第一步：先看当前阶段总结

- [git_snapshot_handoff_20260407.md](/data/uavlab/multi-uav-pursuit2/docs/git_snapshot_handoff_20260407.md)

### 第二步：看专家策略本身

- [expert_strategy_explanation.md](/data/uavlab/multi-uav-pursuit2/docs/expert_strategy_explanation.md)

### 第三步：看 BC / DAgger 的来龙去脉

- [behavior_cloning_explanation.md](/data/uavlab/multi-uav-pursuit2/docs/behavior_cloning_explanation.md)
- [expert_bc_handoff_20260404.md](/data/uavlab/multi-uav-pursuit2/docs/expert_bc_handoff_20260404.md)

### 第四步：看 warm-start RL 为什么失败

- [rl_warmstart_failure_for_claude_20260407.md](/data/uavlab/multi-uav-pursuit2/docs/rl_warmstart_failure_for_claude_20260407.md)

### 第五步：再看关键代码

- `scripts/expert_strategy_test.py`
- `scripts/train_expert_bc.py`
- `scripts/train_actor_dagger.py`
- `omni_drones/learning/mappo.py`
- `omni_drones/utils/torchrl/transforms.py`
- `scripts/train.py`

---

## 8. 当前最重要的未解决问题

当前最值得继续攻克的主问题有两个：

### 8.1 imitation 已经到平台，但还没到 60%+

目前最好 actor 约 `51.6%`，还没到你理想的 `60%+`。

说明：

- 端到端输出推力/角速度这条 imitation 主线不是走不通
- 但继续靠同一套小修小补，收益已经变小

### 8.2 PPO warm-start 还没有学会“接着变强”

现在最大的研究问题不是：

- 专家会不会
- BC/DAgger 能不能学一点

而是：

**为什么一个已经有 51.6% 成功率的 actor，一接 PPO 反而会被冲散。**

这也是接下来最值得继续深挖的地方。

---

## 9. 一句话总结

这次 Git 快照保留下来的不是一份“最终完美解”，而是一份**完整而关键的中间成果**：

- 专家策略已经成型
- BC / DAgger 这条线已经跑通并拿到不错起点
- warm-start RL 的失败也已经被明确定位到较具体的机制问题

所以下次回来看，这份版本最大的价值是：

**你不需要再从头摸索“之前做到哪了”，而是可以直接从“为什么 PPO 会冲掉 imitation 能力”这个真正关键的问题继续往下推进。**

