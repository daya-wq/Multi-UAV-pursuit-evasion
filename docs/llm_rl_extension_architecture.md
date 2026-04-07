# 面向多无人机追捕任务的 LLM + RL 双层架构设计稿

## 1. 文档目的

这份文档不是训练记录，也不是代码变更日志，而是一份面向后续调研和可行性分析的设计稿。

目标是把当前项目的背景、现有痛点、我和你讨论过的后续拓展思路，以及一套适合这个任务场景的双层架构方案整理清楚，方便后续做三件事：

1. 调研相关论文和相近方法。
2. 判断这条路线在本项目中的技术可行性和实验代价。
3. 逐步把目前“人工分析一轮训练，再改奖励和代码”的流程，升级成结构化、可自动化、可复用的外环系统。

这份文档的核心立场是：

- **不要让大模型直接接管底层连续控制。**
- **要让大模型接入训练外环，负责经验提炼、奖励模块调度、课程调整和实验决策。**


## 2. 项目背景

### 2.1 任务定义

当前项目是一个多无人机追捕任务：

- 3 架 pursuer 无人机追捕 1 个 target。
- 环境主体在 [hideandseek.py](/data/uavlab/multi-uav-pursuit2/omni_drones/envs/hide_and_seek/hideandseek.py)。
- 训练算法主干是 MAPPO，入口在 [mappo.py](/data/uavlab/multi-uav-pursuit2/omni_drones/learning/mappo.py)。
- 主配置位于 [HideAndSeek.yaml](/data/uavlab/multi-uav-pursuit2/cfg/task/HideAndSeek.yaml) 和 [mappo.yaml](/data/uavlab/multi-uav-pursuit2/cfg/algo/mappo.yaml)。

当前主环境的关键特点：

- 连续控制、物理仿真、部分可观测。
- 多机协同，不是单机追踪。
- 成功信号稀疏，真正困难的是“围堵完成”，不是“朝目标方向飞”。
- 训练结果需要同时结合 TensorBoard 曲线、日志和评测视频判断，单看 reward 往往不够。


### 2.2 当前训练栈

当前系统实际上已经具备了一个比较完整的 RL 训练底座：

- **环境层**：
  - 多无人机追捕环境
  - 障碍物、地面、边界、碰撞逻辑
  - PIDrate action transform
- **感知/预测层**：
  - 已有 TP 网络，能预测未来目标位置
  - 预测结果已经进入 observation / state
- **策略优化层**：
  - MAPPO + GAE + value normalization
  - 当前已修复过 `action_raw` 与 transform 后命令的混淆问题
- **训练后处理层**：
  - TensorBoard 指标
  - checkpoint 评测
  - 视频导出
  - 人工分析 + 再调 reward / config / code

也就是说，你现在不是“从 0 开始想一个大模型系统”，而是在一个已经能跑的 RL 系统上，给它加一个更强的**训练外环**。


### 2.3 当前项目已经暴露出的典型问题

从我们前面的多轮分析看，当前项目的难点并不在“网络完全学不会任何东西”，而在于：

1. **会追近，但不会稳定围捕**
   - 距离奖励上升，不代表学会收网。
   - 当前距离奖励更像“局部逼近奖励”，不天然等价于“拦截奖励”。

2. **奖励容易诱导局部最优**
   - 例如尾随、保守跟跑、扎堆追逐、贴地摆烂。
   - 某些行为能在短期内规避大罚分，但并不通向真正捕获。

3. **团队协同目标不够明确**
   - 多机之间很容易学成同方向追逐，而不是角色分工和角度覆盖。

4. **训练过程需要多模态诊断**
   - 只看 `return` 不够。
   - 必须同时看：
     - reward 分项
     - PPO 优化状态
     - success / catch / collision / landed
     - 视频里的真实行为

5. **人工调参流程已经形成，但还没有系统化**
   - 你现在实际上已经在做“外环优化”：
     - 跑一个 run
     - 测最新或多个 checkpoint
     - 看 TensorBoard 和视频
     - 诊断失败模式
     - 改 reward / config / 代码
     - 再起下一轮
   - 这套流程已经很接近一个小型研究代理，只是目前还是手工和半手工。


## 3. 为什么值得把大模型接进这个任务

### 3.1 大模型适合解决什么

大模型在这个任务里最适合做的，不是实时飞控，而是下面这几件事：

1. **跨模态诊断**
   - 同时理解 reward 曲线、loss 曲线、日志、视频行为。

2. **跨 run 比较**
   - 当前 run 和旧 run 的参数、指标、视频差异对比。

3. **归纳失败模式**
   - 例如：
     - `tail-chasing`
     - `ground-collapse`
     - `collision-heavy swarming`
     - `early improvement, late regression`
     - `over-conservative pursuit`

4. **提出结构化的小改动**
   - 优先调 reward 权重、观测、终止条件、动作约束。
   - 而不是每次都从头发明新算法。

5. **积累经验库**
   - 形成“现象 -> 根因 -> 修改 -> 结果”的闭环记忆。


### 3.2 大模型不适合直接做什么

不建议让大模型直接在线输出底层动作，原因很明确：

1. 连续控制需要高频、低延迟、数值稳定。
2. 物理仿真和控制器闭环对精细数值非常敏感。
3. 大模型的推理开销和不确定性都不适合作为飞控器。
4. 当前任务的困难主要不在“单步动作求解”，而在“训练目标设计”和“实验外环决策”。

因此更合理的路线是：

- **内环**：继续用 RL 做低层连续控制。
- **外环**：让 LLM 做奖励设计、经验提炼、课程调度和实验编排。


## 4. 与相关工作的关系

### 4.1 可以借鉴 `Complementary RL` 的部分

从 [complementary rl.pdf](/data/uavlab/multi-uav-pursuit2/docs/paper/complementary%20rl.pdf) 能借的核心思想不是文字交互本身，而是：

- 不只让 actor 学。
- 还让“经验提炼器 / 经验库”也随着训练一起演化。

映射到本项目里，可以理解为：

- 当前策略不断训练。
- 外环同时维护一个经验库：
  - 哪种 reward 组合导致什么行为
  - 哪种失败模式最常见
  - 哪种修复在什么场景下有效
- 下次不是从空白重新分析，而是带着“项目内生经验”做判断。


### 4.2 可以借鉴 `RLAnything` 的部分

从 [RLAnything.pdf](/data/uavlab/multi-uav-pursuit2/docs/paper/RLAnything.pdf) 能借的核心思想是：

- reward 不是固定死的
- 环境难度也不是固定死的
- 可以存在一个外部模型，基于训练状态对 reward / curriculum 做慢速更新

映射到本项目里，可以理解为：

- reward 由一组模块组成，不再是完全手写死板的一整块公式。
- 外环可以根据失败模式决定：
  - 开哪些 reward 模块
  - 调哪些系数
  - 先在哪些环境上训
  - 什么时候从简单场景切换到复杂场景


### 4.3 不建议照搬的部分

这两类工作里有些做法不适合直接照搬到本项目：

1. 不建议让 LLM 自由生成任意 reward 代码。
2. 不建议让 LLM 每个 episode 都直接重写环境规则。
3. 不建议一开始就做完全在线、无人工审核的闭环。

本项目更适合：

- **结构化 reward 模块**
- **结构化经验库**
- **慢速外环更新**
- **人机混合决策**


## 5. 设计目标

我建议把这个系统的目标定义成下面 5 条：

1. **提升样本效率**
   - 更快从“只会追”进入“会围堵、会完成捕获”。

2. **降低人工调参成本**
   - 减少每轮都靠人工肉眼逐图对比。

3. **减少无效试错**
   - 避免每次改很多地方，结果无法归因。

4. **增强历史可复用性**
   - 过去训练结论要能沉淀，而不是靠聊天记录回忆。

5. **形成一个任务特化的研究框架**
   - 不只是“这次训练好了没”，而是为后续论文和系统化实验打底。


## 6. 总体架构

建议采用**双层架构**。

### 6.1 内环：Fast RL Layer

这一层负责高频控制和策略学习，保持现在的主干：

- Environment: `HideAndSeek`
- Actor-Critic: MAPPO
- TP prediction
- PPO-style policy update
- TensorBoard / logs / checkpoints / eval videos

这一层只做一件事：

**在给定任务定义、reward 模块配置和环境难度下，尽可能学好策略。**


### 6.2 外环：Slow LLM Layer

这一层负责慢速更新，不参与每一步动作，而是在一个更长的周期上做实验设计：

- 分析最近一轮或最近几个 checkpoint
- 对比历史 run
- 识别主失败模式
- 调整 reward 模块、课程难度、角色策略偏置
- 决定从哪个 checkpoint 继续
- 写入经验库

这一层的更新周期建议是：

- 每个 run 结束后
- 或每 `20M ~ 50M` frames 一个里程碑
- 或训练中断 / 停滞时


### 6.3 一张抽象结构图

```text
                 +----------------------------------+
                 |        Slow LLM Outer Loop       |
                 |----------------------------------|
                 |  Experience Extractor            |
                 |  Reward Designer / Reward Critic |
                 |  Curriculum Adapter              |
                 |  Role / Tactic Planner           |
                 |  Experiment Memory               |
                 +----------------+-----------------+
                                  |
                     structured decisions / configs
                                  |
                                  v
                 +----------------------------------+
                 |         Fast RL Inner Loop       |
                 |----------------------------------|
                 |  HideAndSeek Env                 |
                 |  TP Prediction                   |
                 |  MAPPO                           |
                 |  Checkpoints / Logs / Videos     |
                 +----------------+-----------------+
                                  |
                      metrics / artifacts / behavior
                                  |
                                  v
                 +----------------------------------+
                 |        Antigravity Workflow      |
                 |----------------------------------|
                 |  monitor -> collect -> eval      |
                 |  call Codex skill -> patch/run   |
                 +----------------------------------+
```


## 7. 外环的核心模块

### 7.1 Orchestrator / Workflow

职责：

- 监控训练何时结束、卡住或到达里程碑
- 定位最新 run、checkpoint、视频、日志
- 执行评测
- 触发 Codex skill
- 自动或半自动应用安全改动
- 启动下一轮训练

这部分已经有初步设计，见：

- [antigravity_workflow_design.md](/data/uavlab/multi-uav-pursuit2/docs/antigravity_workflow_design.md)
- [antigravity_autoloop_skill.md](/data/uavlab/multi-uav-pursuit2/docs/antigravity_autoloop_skill.md)


### 7.2 Experience Extractor

职责：

- 从当前 run 中抽取“结构化现象”，而不是只保留自然语言总结。

输入：

- TensorBoard 关键指标
- 日志
- 评测视频
- checkpoint 评测结果
- 当前配置
- 历史 run 对照

输出示例：

```json
{
  "run_id": "HideAndSeek_20260326_115824",
  "primary_failure_mode": "tail_chasing",
  "secondary_failure_modes": ["ground_collapse"],
  "evidence": {
    "distance_reward_trend": "up",
    "success_rate": "zero",
    "catch_reward": "zero",
    "collision_floor": "high",
    "video_behavior": ["follow_from_behind", "no_enclosure"]
  }
}
```

这一层本质上是把“人脑看图后的总结”变成稳定的机器接口。


### 7.3 Experiment Memory / Experience Bank

职责：

- 记录“现象 -> 根因 -> 修改 -> 结果”。
- 为下一次分析提供历史先验。

建议最少存这些字段：

```json
{
  "run_id": "...",
  "parent_run_id": "...",
  "checkpoint_source": "latest | earlier | scratch",
  "task_config_hash": "...",
  "algo_config_hash": "...",
  "reward_modules": {
    "dist_now_progress": 2.0,
    "encircle_reward": 5.0,
    "collision_penalty": 20.0
  },
  "observed_failure_mode": "tail_chasing",
  "proposed_root_cause": "distance reward encourages local pursuit, not interception",
  "patch_summary": "...",
  "post_patch_result": {
    "success": "still_zero",
    "collision_floor": "down",
    "return": "up"
  }
}
```

这个经验库对应 `Complementary RL` 里“经验提炼器”的思想，但更适合工程系统。


### 7.4 Reward Designer / Reward Critic

职责：

- 不直接自由写 Python。
- 只在一组预定义的 reward 模块里做组合和加权。

这是整个架构最关键的约束之一。

推荐的原则：

1. **先模块化，再交给 LLM 调度**
2. **让 LLM 调参数，不要一开始就调自由公式**
3. **把 reward 变成 DSL 或 JSON/YAML 结构**


### 7.5 Curriculum / Environment Adapter

职责：

- 根据当前策略水平，决定下一轮的环境难度和场景分布。

可调项示例：

- `v_prey`
- `use_eval`
- 初始站位随机程度
- 障碍物数量和布局
- 场景类型
- 目标行为模式

为什么重要：

- 你当前很多问题不是纯 reward 问题，而是“当前训练阶段的环境难度不匹配”。
- 会追但不会围时，不应该直接上更复杂场景。
- 会围简单场景但泛化差时，才应该增加随机性和障碍。


### 7.6 Role / Tactic Designer

职责：

- 给多机追捕任务引入“角色偏置”或“战术模板”。

这部分是本项目相比单智能体任务最值得做的地方。

可以从以下模板开始：

- `leader_intercept + two_flankers`
- `left_right_enclosure`
- `front_cut + rear_pressure`

这些模板不一定直接变成硬编码行为，而可以变成：

- reward 锚点
- 覆盖角度奖励
- 防扎堆奖励
- 对不同 agent 的轻微 reward 偏置


### 7.7 Safety Gate / Human Approval

职责：

- 限制外环自动化的改动范围。

建议默认自动的改动：

- reward 系数
- done 条件
- 日志补充
- resume 策略切换

建议必须人工确认的改动：

- 动作空间定义
- 网络结构
- TP 模型结构
- 物理参数大改
- 训练入口切换


## 8. Reward DSL 与模块库建议

### 8.1 为什么要做 reward DSL

如果没有 reward DSL，外环很快会退化成：

- 大模型直接改代码
- 每次改动不可控
- 很难对比历史 run
- 很难判断到底哪一项修改有效

因此更合理的路线是：

- 先把 reward 拆成模块
- 再让外环只操作这些模块


### 8.2 推荐的 reward 模块分类

#### A. 基础逼近类

- `dist_now_progress`
  - 当前已存在
  - 含义：朝当前目标更近

- `dist_pred_progress_k1`
  - 新增建议
  - 含义：朝短时未来预测目标更近

- `dist_pred_progress_k2`
  - 新增建议
  - 含义：朝更远一点的未来预测位置更近


#### B. 围堵几何类

- `encircle_reward`
  - 当前已经存在一个质心版本
  - 含义：团队整体位置更靠近包围态

- `coverage_reward`
  - 新增建议
  - 含义：无人机围绕目标或预测点的角度分布更开，不扎堆

- `anti_bunching_reward`
  - 新增建议
  - 含义：抑制多机从同一侧紧跟

- `front_intercept_reward`
  - 新增建议
  - 含义：鼓励抢占目标前方空间，而不是纯尾随


#### C. 捕获完成类

- `catch_reward`
  - 当前已有
  - 含义：完成捕获的稀疏成功奖励

- `close_range_control_reward`
  - 后续可选
  - 含义：进入近距离后鼓励稳定围堵而不是乱撞


#### D. 安全约束类

- `collision_penalty`
  - 当前已有

- `floor_penalty`
  - 当前已有

- `landed_penalty`
  - 当前已有

- `speed_penalty`
  - 当前已有

- `smoothness_reward`
  - 当前已有


#### E. 课程门控类

- `stage_gate_far`
- `stage_gate_mid`
- `stage_gate_near`

它们本身不是 reward，而是决定哪些 reward 在哪个阶段激活。


### 8.3 一个建议的 reward 配置格式

```yaml
reward_program:
  modules:
    dist_now_progress:
      enabled: true
      coef: 2.0

    dist_pred_progress_k1:
      enabled: true
      coef: 0.8
      gated_by: detection

    coverage_reward:
      enabled: false
      coef: 0.5
      gated_by: mid_range

    encircle_reward:
      enabled: true
      coef: 5.0

    catch_reward:
      enabled: true
      coef: 200.0

    collision_penalty:
      enabled: true
      coef: 20.0

    landed_penalty:
      enabled: true
      coef: 50.0

    speed_penalty:
      enabled: true
      coef: 2.0

    smoothness_reward:
      enabled: true
      coef: 0.05
      gated_by: airborne
```

LLM 外环应该输出这种结构，而不是直接输出一段 Python 公式。


## 9. 当前项目最值得做的 reward 拓展

结合当前代码和我们前面的诊断，最推荐先做的不是“学习式 reward model”，而是三件更扎实的事情。

### 9.1 让预测真正进入 reward

当前 TP 预测已经进入观测，但没有真正进入主 reward。

这是一个明显可利用的基础设施缺口。

建议新增：

- `distance_predicted_reward`
  - 对短时未来预测点做 progress reward
  - 不是绝对距离，而是“比上一时刻更接近预测点”


### 9.2 让团队几何进入 reward

当前 reward 虽然已经加入了 `encircle_reward`，但它还是偏质心导向，仍然不够表达：

- 分工
- 覆盖角度
- 防扎堆
- 前向拦截

建议下一步补：

- `coverage_reward`
- `anti_bunching_reward`
- 轻量级 `front_intercept_reward`


### 9.3 让 reward 随阶段切换

多机追捕天然是分阶段的：

1. 远距离：追近
2. 中距离：展开、包夹、截断
3. 近距离：完成捕获并避免乱撞

单一全局 reward 往往把这些阶段混在一起。

因此建议：

- 远距离阶段以 `dist_now_progress + dist_pred_progress` 为主
- 中距离阶段逐步提高 `coverage_reward / anti_bunching`
- 近距离阶段提高 `catch_reward` 和近距控制约束


## 10. 数据流设计

### 10.1 内环数据流

```text
obs/state
  -> actor
  -> action
  -> action transform / controller
  -> environment step
  -> reward + done + next_obs
  -> rollout buffer
  -> MAPPO update
  -> checkpoint / logs / tb scalars
```

这一层不需要大改。


### 10.2 外环数据流

```text
run finished / stalled / milestone
  -> locate artifacts
  -> evaluate selected checkpoints
  -> export key scalars and plots
  -> read logs + videos
  -> compare with historical runs
  -> extract failure mode
  -> choose minimal patch
  -> decide resume strategy
  -> apply patch or request review
  -> launch next run
  -> record experience into bank
```


### 10.3 外环接口建议

建议把外环中间产物都做成结构化文件：

- `latest_artifacts.json`
- `eval_targets.json`
- `run_summary.json`
- `postmortem_report.md`
- `outer_loop_decision.json`
- `experience_bank.jsonl`

这样可以减少每次都从大段非结构化文本重新解析。


## 11. 更新协议

这里给出一版适合本项目的**外环更新协议**。

### 11.1 触发条件

满足任一即可：

1. 训练进程退出
2. 长时间无新 checkpoint
3. 到达里程碑步数

推荐阈值：

- `on_exit`
- `on_stall = 60~90 min`
- `on_milestone = 20M~50M frames`


### 11.2 每轮分析输入

最少输入：

- 最新 run 目录
- 最新 checkpoint
- 1~2 个更早 checkpoint
- 评测视频
- TensorBoard 导出图或关键标量
- `logs/train.log`
- 当前任务配置
- 当前算法配置
- 历史训练分析文档
- 历史经验库


### 11.3 每轮分析顺序

建议固定如下顺序：

0. **先和历史 run 对比**
1. **先看 outcome**
   - `success`
   - `catch_reward`
   - `first_capture_step`
   - `return`
2. **再看 reward 分项**
   - `distance_reward`
   - `encircle_reward`
   - `collision_reward`
   - `speed_reward`
   - `smoothness_reward`
   - `landed_penalty`
3. **再看 optimizer 稳定性**
   - `ESS`
   - `policy loss`
   - `value loss`
   - `entropy`
   - `KL` 或近似指标
4. **再看控制与几何行为**
   - `action_norm`
   - `cmd_norm`
   - `collision_floor`
   - `pursuer_collisions_count`
5. **最后看视频**
   - 是否与指标一致
   - 是否出现新的局部最优


### 11.4 每轮只选一个主假设

这是非常重要的实验纪律。

一轮分析后，必须只选一个主失败模式，例如：

- `tail_chasing`
- `ground_collapse`
- `over_conservative_pursuit`
- `collision_heavy_swarming`
- `tracking_good_but_capture_absent`

然后只围绕这个主假设做一组最小改动。

否则很容易出现：

- 一轮改太多地方
- 结果变了，但根本不知道是哪个改动起作用


### 11.5 修改优先级

建议固定优先级如下：

1. 先修**明显代码/接线错误**
   - action 键污染
   - done 条件错误
   - stats 记录错误
   - reward 没真正接入

2. 再改 **reward 权重**

3. 再改 **reward 设计**
   - 新增模块
   - 新增 gating
   - 阶段切换

4. 再改 **观测设计**

5. 再改 **动作约束**

6. 最后才改 **算法主干**


### 11.6 每轮决策输出

每次外环必须输出：

1. 现象
2. 可能根因
3. 证据
4. 建议修改
5. 风险
6. 下一轮要验证什么
7. 继续策略
   - `resume_latest`
   - `resume_earlier`
   - `restart_scratch`


### 11.7 每轮结果回写

每轮结束后至少写回两处：

1. `training_history_analysis.md`
2. `experience_bank`

这样后续分析就不会只依赖上下文记忆。


## 12. 推荐的实验记忆结构

建议把经验库设计成 append-only 的 `jsonl` 或小型数据库，每条记录对应一次外环决策。

建议字段：

- `run_id`
- `timestamp`
- `parent_run_id`
- `checkpoint_strategy`
- `config_diff`
- `reward_program`
- `primary_failure_mode`
- `evidence_summary`
- `video_summary`
- `patch_summary`
- `expected_effect`
- `observed_effect`
- `regression_risk`
- `next_recommendation`

这份经验库未来可以有两个用途：

1. 给 Codex/LLM 提供项目内“记忆”
2. 给论文或技术报告提供系统化案例材料


## 13. 推荐的阶段化落地路线

### Phase 0：当前状态

你现在已经在做：

- 训练
- 导出图
- 看视频
- 找根因
- 改 reward / code
- 起下一轮

这其实已经是一个**人工外环**。


### Phase 1：半自动化外环

目标：

- workflow 自动找工件、自动评测、自动调用 skill
- 但 patch 和起新 run 仍然人工确认

这是最值得优先做的阶段，因为投入小，收益直接。


### Phase 2：reward DSL 化

目标：

- 把 reward 从硬编码大段公式，逐步整理成模块和配置
- 外环不直接改 Python 主体，而是改 reward program

这是整个系统从“prompt 工程”走向“研究框架”的关键一步。


### Phase 3：experience bank

目标：

- 不再每轮都从空白开始分析
- 强制做历史 run 对比
- 形成失败模式和改动效果的结构化沉淀


### Phase 4：课程学习与角色化

目标：

- 外环不只调 reward，还调场景难度和战术偏置
- 引入更明确的“角色模板”或“拦截几何模板”


### Phase 5：部分自动闭环

目标：

- 对低风险改动自动执行
- 自动起下一轮训练
- 人只在高风险改动时审批

这时才算真正形成一个“项目专属的双层训练系统”。


## 14. 这个架构为什么在本项目中可行

### 14.1 技术上已经具备的条件

当前项目已经具备几个非常关键的基础：

1. **训练主干是稳定可改的**
   - Hydra 配置明确
   - reward 主要集中在环境文件里

2. **预测能力已经存在**
   - TP 网络已接进 observation
   - 下一步只差把预测变成 reward 或阶段逻辑

3. **外环分析流程已经成形**
   - 你已经在用 Codex + antigravity 做诊断

4. **日志和工件比较完整**
   - checkpoints
   - TensorBoard
   - 视频
   - 文档

也就是说，这不是“想象一个新系统”，而是把现有链条正式化。


### 14.2 最大的不确定性

这条路线最主要的风险不在“能不能写出来”，而在“会不会变成新的噪声源”。

主要风险有：

1. **reward hacking 被放大**
   - 外环如果不受约束，可能不断制造新的漏洞。

2. **多改动混杂，难以归因**
   - 这是目前自动化系统最容易犯的错。

3. **LLM 幻觉**
   - 如果没有结构化输入和历史对照，LLM 很容易给出看似合理但不对症的建议。

4. **评测选择偏差**
   - 只测最新 checkpoint 会经常误判。

5. **工程复杂度膨胀**
   - 经验库、reward DSL、workflow、patch gate 全做起来后，系统本身就需要维护。


### 14.3 对应的控制策略

建议用下面几条规则控风险：

1. 每轮只改一个主假设
2. LLM 只在受限模块空间里操作
3. 必须和历史 run 对比
4. 必须同时看视频和指标
5. 默认评测多个 checkpoint，而不是只测最新
6. 高风险改动必须人工审批


## 15. 近期最值得做的最小可行版本

如果只做一个最小 MVP，我建议是：

### MVP-1：自动化外环，不碰算法主干

包括：

1. 固定 workflow
2. 固定 skill
3. 固定 artifact 收集
4. 固定 postmortem 输出
5. 固定 resume 策略

也就是：

- 先把“人工外环”标准化、自动化
- 暂时不做在线 reward model


### MVP-2：reward DSL 的第一版

只先支持 6~10 个模块：

- `dist_now_progress`
- `dist_pred_progress_k1`
- `encircle_reward`
- `coverage_reward`
- `catch_reward`
- `collision_penalty`
- `landed_penalty`
- `speed_penalty`
- `smoothness_reward`

这样已经足够支撑很多外环实验。


### MVP-3：经验库第一版

只记录最关键的结构化字段，不追求复杂数据库。

一个 `jsonl` 文件就够开始用了。


## 16. 建议优先验证的研究问题

如果后续你要做系统化研究，我建议优先验证下面这些问题：

1. **外环诊断是否能减少试错轮数**
   - 指标：达到首次稳定成功所需的 run 数量和总训练帧数

2. **reward DSL 是否优于自由改代码**
   - 指标：回归率、分析稳定性、修改可解释性

3. **加入预测增强 reward 后，是否能显著减少“尾随不围堵”**
   - 指标：success、catch_reward、围堵几何指标、视频行为

4. **历史经验库是否能降低重复犯错**
   - 指标：同类失败模式重复出现的频率

5. **外环能否正确判断 resume 策略**
   - 指标：`resume_earlier` 和 `restart_scratch` 的选择效果


## 17. 建议的近期实施顺序

如果按投入产出比排序，我建议是：

1. 完善 antigravity + Codex 的自动化外环
2. 建 reward DSL 第一版
3. 把预测真正接进 reward
4. 建 experience bank
5. 做角色化 / 围堵几何模块
6. 再考虑更强的 reward model 或课程自适应

这个顺序的核心原因是：

- 先把实验系统化
- 再把 reward 模块化
- 最后再追求更“智能”的自适应


## 18. 一句话总结

这条路线是可行的，而且与你当前的工作方式天然衔接。

最合适的方案不是“让大模型直接学飞”，而是做一个**任务特化的双层系统**：

- **内环 RL** 负责连续控制与策略学习
- **外环 LLM** 负责经验提炼、奖励模块调度、课程调整和实验决策

如果这套系统搭起来，它的价值不只是“帮你调一次 reward”，而是把你现在已经在做的高质量研究流程，变成一个能持续积累经验、可复现、可扩展的项目级训练框架。
