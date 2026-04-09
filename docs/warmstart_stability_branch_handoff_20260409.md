# Warm-Start Stability Branch 说明（2026-04-09）

这份文档说明当前分支 [exp/warmstart-stability](/data/uavlab/multi-uav-pursuit2) 和原始仓库相比，多了什么、为什么要加这些改动、当前应该怎么理解这条分支。

## 1. 先说结论

这个分支不是从零重写项目，而是在已经推到 `exp-baseline` 的“专家策略 + 行为克隆 + DAgger + RL warm-start 分析”基础上，再额外做了一轮 **RL warm-start 稳定性修正**。

大白话说：

- 原始仓库更偏“直接 RL 训练”
- `exp-baseline` 已经加上了专家策略、专家数据采集、BC、DAgger、批量评测、很多文档和关键权重
- **当前分支 `exp/warmstart-stability`** 则继续解决一个更具体的问题：
  - 为什么把一个已经通过 BC/DAgger 预训练好的 actor 接到 PPO 里以后，策略很快变差
  - 为什么训练时 `train any_landed` 很高、`action_norm` 会漂大、warm-start 能力保不住

所以这条分支可以理解成：

**“在 `exp-baseline` 的基础上，专门针对 RL warm-start 失稳问题做的实验分支。”**


## 2. 和一开始的原始仓库相比，加了什么

如果把原始仓库近似看成 [upstream/master](/data/uavlab/multi-uav-pursuit2)，当前项目已经多了下面几大块能力。

### 2.1 专家策略与专家评测

新增/扩展了：

- [scripts/expert_strategy_test.py](/data/uavlab/multi-uav-pursuit2/scripts/expert_strategy_test.py)
- [scripts/expert_isaac_eval.py](/data/uavlab/multi-uav-pursuit2/scripts/expert_isaac_eval.py)
- [scripts/expert_isaac_eval.sh](/data/uavlab/multi-uav-pursuit2/scripts/expert_isaac_eval.sh)

这部分的作用是：

- 写了一个可解释的专家追捕策略
- 可以在 Isaac Sim 里直接跑专家
- 可以批量测试专家成功率
- 可以录专家视频
- 可以把专家策略输出转换成训练 actor 使用的同一动作空间

也就是说，原始仓库里没有这么完整的“专家示教”链路，现在有了。

### 2.2 专家数据采集与行为克隆（BC）

新增/扩展了：

- [scripts/train_expert_bc.py](/data/uavlab/multi-uav-pursuit2/scripts/train_expert_bc.py)
- [scripts/train_expert_bc.sh](/data/uavlab/multi-uav-pursuit2/scripts/train_expert_bc.sh)
- [scripts/run_expert_bc_pipeline.sh](/data/uavlab/multi-uav-pursuit2/scripts/run_expert_bc_pipeline.sh)
- [scripts/eval_policy_batch.py](/data/uavlab/multi-uav-pursuit2/scripts/eval_policy_batch.py)

这部分做了几件事：

- 用专家策略批量采集成功轨迹
- 只保留成功局，构建专家经验库
- 用专家经验库去训 actor
- 再把 BC 后的 actor 放回 Isaac 做真实 rollout

原始仓库里并没有这条完整的 imitation learning 流程。

### 2.3 DAgger / 在线纠偏训练

新增：

- [scripts/train_actor_dagger.py](/data/uavlab/multi-uav-pursuit2/scripts/train_actor_dagger.py)

这部分是因为纯 BC 不够，后来进一步做了：

- actor 自己在环境里跑
- 专家在同一状态上给标签
- 用在线纠偏方式继续提升 actor

这条线最后产出了比较好的 actor 权重，比如：

- [dagger_best_preserved_20260405_gru52_goal23_landed17.pt](/data/uavlab/multi-uav-pursuit2/checkpoints/dagger_best_preserved_20260405_gru52_goal23_landed17.pt)

这版大约有 `51.6%` 的成功率，是后面 RL warm-start 的主要起点。

### 2.4 大量补充文档

原始仓库相对缺少一套“能让人回头看懂整个过程”的文档。现在已经补上了不少关键说明，例如：

- [docs/expert_strategy_explanation.md](/data/uavlab/multi-uav-pursuit2/docs/expert_strategy_explanation.md)
- [docs/behavior_cloning_explanation.md](/data/uavlab/multi-uav-pursuit2/docs/behavior_cloning_explanation.md)
- [docs/expert_bc_handoff_20260404.md](/data/uavlab/multi-uav-pursuit2/docs/expert_bc_handoff_20260404.md)
- [docs/当前网络结构说明.md](/data/uavlab/multi-uav-pursuit2/docs/当前网络结构说明.md)
- [docs/rl_warmstart_failure_for_claude_20260407.md](/data/uavlab/multi-uav-pursuit2/docs/rl_warmstart_failure_for_claude_20260407.md)
- [docs/git_snapshot_handoff_20260407.md](/data/uavlab/multi-uav-pursuit2/docs/git_snapshot_handoff_20260407.md)

所以和原始仓库相比，现在不仅多了功能，也多了比较系统的“怎么理解这个项目”的说明。


## 3. 当前分支相对于 `exp-baseline` 又额外加了什么

`exp-baseline` 可以理解成“第一阶段成果快照”；  
当前分支 `exp/warmstart-stability` 是在这个快照上继续做的一轮更聚焦的改动。

当前分支只额外修改了 5 个核心文件：

- [cfg/algo/mappo.yaml](/data/uavlab/multi-uav-pursuit2/cfg/algo/mappo.yaml)
- [omni_drones/learning/mappo.py](/data/uavlab/multi-uav-pursuit2/omni_drones/learning/mappo.py)
- [omni_drones/learning/modules/distributions.py](/data/uavlab/multi-uav-pursuit2/omni_drones/learning/modules/distributions.py)
- [omni_drones/utils/torchrl/transforms.py](/data/uavlab/multi-uav-pursuit2/omni_drones/utils/torchrl/transforms.py)
- [scripts/train.py](/data/uavlab/multi-uav-pursuit2/scripts/train.py)

这些改动都围绕同一个目标：

**让 warm-start actor 接 PPO 以后，别那么容易被探索噪声和动作饱和冲坏。**


## 4. 这 5 个文件具体改了什么

### 4.1 `cfg/algo/mappo.yaml`

这里把 actor 的动作参数化和初期探索强度收了一档：

- `actor.tanh: false -> true`
- `log_std_init: 0.0 -> -3.0`
- `log_std_min: -2.0 -> -5.0`
- `log_std_max: 0.5 -> -1.0`

大白话理解：

- 原来 actor 输出的是 raw Gaussian 动作，到了控制器里再 `tanh`
- 现在改成 actor 自己就输出 squash 后的动作
- 同时把初始动作噪声压得更小，避免 warm-start 一开始就乱飞

### 4.2 `omni_drones/learning/modules/distributions.py`

这里主要是补 `TanhIndependentNormalModule`，让它更适合现在这条训练链：

- 支持 `log_std_init / log_std_min / log_std_max`
- 非 state-dependent std 情况下，显式 clamp `log_std`
- 用 `fc_mean` 命名来兼容旧 checkpoint 的加载

这一步很重要，因为你已经有不少历史权重，如果名字完全不兼容，切换动作分布后旧权重就很难直接 warm-start。

### 4.3 `omni_drones/utils/torchrl/transforms.py`

这里给 `PIDRateController` 增加了 `actor_has_tanh` 开关。

逻辑变成：

- 如果 actor 本身已经是 `tanh` 输出
  - 那控制器里就**不要再多做一次 `torch.tanh(action)`**
  - 只做一个安全 clamp
- 如果 actor 还是 raw Gaussian
  - 那控制器里保留原来的 squash 逻辑

大白话说就是：

**避免“前面已经压过一次，后面又压一次”的动作链不一致问题。**

### 4.4 `scripts/train.py`

这里只做了一点很关键的接线：

- 在构建 `PIDRateController` transform 时，把 `actor_has_tanh` 传进去

不然上面在 `transforms.py` 里加的逻辑就不会生效。

### 4.5 `omni_drones/learning/mappo.py`

这里有两个重点。

第一个是构建 actor 时：

- 如果 `cfg.actor.tanh=true`
- 就用支持 `log_std_*` 配置的 `TanhIndependentNormalModule`

第二个是训练日志里对 `action_squash_sat_frac` 的统计逻辑做了区分：

- 如果 actor 已经是 tanh 输出
  - 就直接看输出是否接近 `±1`
- 如果 actor 还是 raw 输出
  - 就先过 `tanh` 再看是否接近 `±1`

这一步的作用是：

**以后看 TensorBoard 时，能更早分清楚“是 raw action 很大”还是“post-tanh 真的饱和了”。**


## 5. 为什么要做这批改动

因为前一轮 RL warm-start 虽然能跑，但表现出一个很典型的问题：

- actor 在 imitation / DAgger 阶段已经学出大约 `51.6%` 成功率
- 但一接 PPO，训练很快把这种能力冲掉

我们当时看到的症状是：

- `train any_landed` 很高
- `eval` 不一定全崩，但就是抓不到
- `raw action_norm` 会一路往大漂
- 动作分布和控制器动作链之间有参数化不一致

所以这批改动的核心动机是：

1. **降低 warm-start 初期探索噪声**
2. **让 actor 输出和控制器期望的动作语义更对齐**
3. **增加饱和监控，便于后面继续诊断**


## 6. 当前分支不是要证明什么

这一点也要说清楚。

当前分支并不是在证明：

- “问题已经完全解决”
- 或者“这套 PPO warm-start 一定成功”

它更像是：

**把 warm-start 的训练链从‘很容易被冲坏’推进到‘至少更稳、更可诊断’。**

也就是说，这条分支的意义是：

- 修动作参数化
- 修控制器接线
- 压初期噪声
- 给后续实验提供一个更靠谱的起点


## 7. 当前分支没有一起提交什么

为了保持 Git 历史干净，这次不会把下面这些本地产物一起推进去：

- `runs/`
- `eval_videos/`
- `backups/`
- `.agents/`
- `.codex`
- 零散本地脚本、临时日志、临时图片

也就是说：

**这次推上 GitHub 的重点是代码和说明，不是把所有运行产物都塞进仓库。**


## 8. 当前分支最适合怎么理解

一句最简短的话：

**如果 `exp-baseline` 是“专家模仿学习主线快照”，那 `exp/warmstart-stability` 就是“在此基础上，专门修 RL warm-start 稳定性”的实验分支。**

如果之后要继续做：

- PPO 稳定性
- warm-start 策略保护
- 动作饱和诊断
- tanh 动作链对齐

那么这条分支就是更合适的继续实验起点。
