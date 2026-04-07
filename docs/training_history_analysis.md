# 多无人机追捕强化学习 - 训练历史与奖励演进记录

本文档汇总了多无人机追逃项目 (Multi-UAV Pursuit-Evasion) 的各阶段训练版本、奖励配置的变更动因、张量板 (TensorBoard) 表现及根因剖析，为后续的策略迭代和论文撰写提供核心材料。

---

## 阶段一：Version 1 (v1) - 原始配置基线测试

**基本情况**:
- **训练时长**: 约 4.45 亿步 (445M frames)
- **环境条件**: `use_eval=1` (固定开局初始位置)，无障碍物空旷环境
- **速度设定**: 猎物 `v_prey = 1.3 m/s` vs 追捕者 `v_drone = 1.0 m/s`
- 📈 **最终权重目录**: `/data/uavlab/multi-uav-pursuit2/checkpoints/HideAndSeek_20260324_033245/` (最高到 `checkpoint_511311872.pt`)
- 🎬 **测评视频路径**: `/data/uavlab/multi-uav-pursuit2/eval_videos/HideAndSeek_20260324_211942.mp4`

**核心奖励设定**:
| 参数 | 设定值 | 设定目的 / 倾向 |
|---|---|---|
| `catch_radius` | 0.3m | 判定捕获的极限距离（非常严苛） |
| `catch_reward_coef` | + 60.0 | 抓捕成功的分数 |
| `collision_coef` | - 100.0 | **重度制裁** 无人机互撞和撞墙 |
| `distance_reward_coef`| - 1.0 | 鼓励靠近目标 |

**训练结果与根因分析**:
- **成功率表现**: **横线 0%**，从未触发过成功捕获。
- **行为涌现**: 无人机学会了追踪猎物（距离惩罚减小），且极度遵守交通规则（碰撞发生率近乎为 0），控制平滑度指标（打满舵的频次）也表现良好。
- **失败根因阐析**：
  1. **“奖励冲突”导致的畏缩妥协**：无人机靠近目标必然会导致聚拢，聚拢极易引发互撞。由于互撞惩罚 (-100) 远大于成功抓捕 (+60)，AI 学到了一条最安全得高分的捷径——**“保持距离尾随，绝不冒险合围”**。
  2. **速度碾压与完美逃跑算法**：猎物由人工势场（APF）控制，会在无死角的情况下以 1.3m/s 轻松甩开龟速 (1.0m/s) 的追捕者，只有教科书般的完美三机包抄才能赢，但在高昂的碰撞成本下，这种高难度的探索被直接阻断。

---

## 阶段二：Version 2 (v2) - 突破奖励冲突瓶颈

**基本情况**:
- **训练载入**: 截取 v1 初期已学会基础飞行的 0.78 亿步权重 (`checkpoint_78774272.pt`) 热启动。
- 📈 **最终权重目录**: `/data/uavlab/multi-uav-pursuit2/checkpoints/auto_train_empty/` 及后续备份文件夹
- 🎬 **测评视频路径**: `/data/uavlab/multi-uav-pursuit2/eval_videos/HideAndSeek_20260325_001144.mp4`

**核心奖励设定变动**:
| 参数 | v1 旧值 | v2 新值 | 调整逻辑 |
|---|---|---|---|
| `catch_radius` | 0.3m | **0.4m** | 稍微放宽空间判定，降低合击抓捕的门槛。 |
| `catch_reward_coef` | 60.0 | **200.0** | “重赏之下必有勇夫”：将吃肉的收益提升为核心。 |
| `collision_coef` | 100.0 | **20.0** | 大幅削弱碰撞代价，鼓励智能体互相靠近摩擦，尝试极具进攻性的包揽动作。 |
| **新增指标追踪** | 无 | `pursuer_collisions_count` | 在张量板新增无人机真实坠毁/互撞次数的具体统计。 |

**训练结果与根因分析 (基于 ~80M步 测试截点)**:
- **训练健康度**: `actor_grad_norm` 与 `critic_grad_norm` 持续且稳健地下降，预测方差 `advantages_std` 降低，模型不仅没有崩溃，反而在坚实地汲取经验。
- **关键显性现象：动作范数 (`action_norm`) 暴涨突破 11**：
  - 由于阻碍（碰撞惩罚）被扫清且目标利润极大化，无人机产生了极其强烈的追杀欲望。
  - 由于系统里写死了物理护栏限制 (`max_linear_velocity = 1.0`) 和电机的输出极限拦截 `clamp(0,1)`，无人机在网络输出层上试图用越来越巨大（无上限增加的量级）的数值去猛踩油门和打死方向盘。
  - **结论**：网络进入了进攻性满负荷状态（控制饱和）。这标志着无人机从先前的“龟缩防守”转向了“激进试探”。

---

## 阶段三：Version 3 (v3) - 解决“状态坍缩”与丰富探索

**基本情况**:
- **优化方向**: 解决因单一固定开局导致的“死记硬背”（环境坍缩过拟合），拓宽泛化能力。
- 📈 **最新权重目录**: `/data/uavlab/multi-uav-pursuit2/checkpoints/HideAndSeek_20260325_*/` (系列时间戳最新目录)
- 🎬 **最新测评视频**: `/data/uavlab/multi-uav-pursuit2/eval_videos/HideAndSeek_20260325_144026.mp4`

**改动内容**:
1. **动态随机初始化 (Randomized Initial Positions)**：彻底废除 `use_eval=1`。无人机每次重生位置打乱，逼迫 AI 在任意杂乱站位下都能依靠观测场构建出临时的围捕阵型。
2. **底层 Bug 突围**：排查并修复了在开启多环境随机并发重置时，`_reset_idx` 逻辑里引发 Isaac Sim/PhysX 内存越界导致段错误（Exit Code 139 / Segfault）的隐患，保证了多维探索的引擎健壮性。
3. **动态动作噪声 (Dynamic Action Noise)**：人为注入额外的策略熵流（Policy Entropy），防止在 v2 激进参数下模型过早收敛次优解（例如两架撞在一起，另一架独自摸鱼），强制让无人机产生多样化的微操机动轨迹以填补非欧几里得轨迹解。

---

## 阶段四：Version 3.1 延续 (15:36 / 210M步 评估)

**基本情况**:
- **操作目的**: 回溯评估加入随机化与动态动作噪声后的长期影响（约 2.1 亿步状态）。
- 📈 **本次提取权重**: `/data/uavlab/multi-uav-pursuit2/checkpoints/HideAndSeek_20260325_153617/checkpoint_209846272.pt`
- 🎬 **刚生成的测评录像**: `/data/uavlab/multi-uav-pursuit2/eval_videos/HideAndSeek_20260325_164747.mp4`

**张量板 (TensorBoard) 核心现象分析 (210M 步横截面)**:
- **`TP_loss` 与梯度全面收敛**: 目标预测误差归零并且 `critic_grad_norm` 和 `actor_grad_norm` 均平稳贴地，说明模型并未遭受灾难性遗忘，基础学习通道是通畅的。
- **💥 `action_norm` (动作范数) 超量暴涨突破 200**: 对比之前 80M 步时的极值 11，现在的动作范数已经纵向垂直起飞。由于底层执行器模型中有强硬的 `clamp` 机制对油门进行拦截打折，网络通过暴力叠加上限来测试系统的极限输出响应（犹如油门踏板被踩穿）。
- **🌪️ `entropy` (策略熵) 反直觉暴增至 24**: 这是本次诊断中**最危险且最异常**的信号！在规范的 PPO 下，策略越收敛，其信息熵理应越小（确定性变强，向 0 靠拢）。但图表中的策略熵呈一条完美的爬坡直线飙升到了 24。
  - **根因推测**：这直接证明了 v3 阶段新加入的 **“动态动作噪声 (Dynamic Action Noise)”** 或者 `entropy_coef` 配置出现了严重坍塌。模型正被强行迫使输出无限宽广的高斯方差（几乎等同于是输出纯粹的随机动作），进而导致连带的动作范数剧增。网络被自己注入的探索噪声给“毒化”了。

**后续直接行动建议**:
强烈建议立刻打开上方的 `HideAndSeek_20260325_164747.mp4` 录像确认体态。若发现无人机不再进行有意识的追踪包抄，而是产生剧烈的乱颤、胡乱冲撞或是反物理的抽搐，则证实了动作噪声策略崩溃。必须立刻切断这个噪声增益选项或大幅度调低熵奖励系数。

---

## 📍 后续分析及推荐观测锚点

1. **视频肉眼排查 (Visual Rendering Check)**
   - 随后的评估阶段（Evaluation Rollout）必须优先验证采用 v2/v3 最新策略的轨迹平滑度。如果发现飞行体僵直、剧烈抽搐甚至导致反物理抖动，说明 `action_norm` 的膨胀已经负面影响了姿态控制。
2. **真机部署微调 (Two-Stage Refinement)**
   - 如果验证抓捕成功率终于破 0 并稳步爬升，务必进入代码内置的 `HideAndSeek_deploy.py`，逐步释放 `smoothness_reward_coef` 的拉力（即平滑惩罚）。这能把网络从极其野蛮的 Bang-Bang 控制拉回到符合飞行力学的柔性指令，从而确保代码能安全烧录转移到实车上。
3. **课程学习升维 (Curriculum Elevation)**
   - 随着目前随机站位（v3）的加入，不妨尝试在阶段性达到 80% 成功率后，唤醒内建的 `HideAndSeek_envgen.py`，让生成的环境随机刷出遮挡视线的圆柱体，考察预测网络 (`TP_net`) 是否有效。

---

## 2026-03-25 补充核对（供 antigravity 交叉验证）

这一节只保留基于当前仓库代码、TensorBoard event 文件和训练日志的可复核结论，并明确区分“已证实”和“待验证怀疑”。

### A. 当前有效训练配置核对

**已证实：当前活跃训练并不是随机初始化 v3。**

- `scripts/auto_train.sh` 当前实际启动参数仍然是：
  - `task=HideAndSeek`
  - `task.use_eval=1`
  - `task.use_random_cylinder=0`
  - `task.scenario_flag=empty`
  - `task.env.num_envs=2048`
- 证据见：
  - `scripts/auto_train.sh`
  - `formal_training.log` 开头的 Hydra 参数打印
- 这意味着本轮主训练仍然是：
  - 固定开局
  - 空场景
  - 2048 并行环境
  - 不是 `HideAndSeek_envgen`
  - 不是 `HideAndSeek_deploy`

**已证实：本轮奖励配置是 v2 风格。**

- `cfg/task/HideAndSeek.yaml` 中当前值为：
  - `catch_reward_coef: 200.0`
  - `collision_coef: 20.0`
  - `catch_radius: 0.4`
  - `dist_reward_coef: 2.0`
  - `speed_coef: 2.0`
  - `smoothness_coef: 0.0`

### B. 到底有没有成功

**已证实：有成功，但极少，不是稳定学会。**

从 TensorBoard event 原始标量提取结果：

- `runs/HideAndSeek_20260325_152109/`
  - `train/stats.success` 非零 4 次
  - 非零 step 分别为：`5,111,808`、`8,388,608`、`21,495,808`、`31,326,208`
- `runs/HideAndSeek_20260325_153617/`
  - `train/stats.success` 非零 1 次
  - 非零 step 为：`37,879,808`

对应日志中同时出现：

- `train/stats.catch_reward > 0`
- `train/stats.first_capture_step < 800`
- `train/stats.success > 0`

这说明这些不是 TensorBoard 幻觉，而是真实捕获事件。

**量级解释：**

- 非零 `success` 大约在 `4.88e-4` 左右，本质上接近 `1 / 2048`
- 这表示某一次聚合统计里，大约只有 1 个 episode 成功
- 因此当前阶段更准确的表述应为：
  - “出现过偶发捕获”
  - 不能表述成“已经学会围捕”

### C. 为什么成功率图像看起来像一个奇怪尖峰

**已证实：这是稀疏成功事件经过 TensorBoard 平滑后的显示效果。**

原因有两层：

1. `train/stats.success` 是 episode 级均值，不是逐步 success flag。
2. 当前成功极稀疏，原始值只有 `1e-4` 量级；一旦打开较大的 smoothing，就会显示成一个“尖峰后指数衰减”的形状。

因此图像上那条怪曲线并不表示“成功率真正先升后降”，而更像是：

- 某个时间点出现了单次极稀有成功
- 之后又恢复到 0
- TensorBoard 平滑把孤立点拉成了长尾

### D. 为什么又停了

这里要区分两次不同的“停”。

#### D1. `auto_train.sh` 那次停止

**已证实：父监控脚本先没了，但子训练还继续跑了一段时间。**

- `auto_train_monitor.log` 最后更新时间停在 `2026-03-25 15:02:48`
- 文件最后只记录到“第 5 次启动，训练 PID: 605955”
- 但 `formal_training.log` 仍继续写到约 `15:35`
- 并且 `HideAndSeek_20260325_152109/` 下继续落盘到了 `checkpoint_39452672.pt`

结论：

- 不是“训练正常结束后脚本退出”
- 而是 wrapper `auto_train.sh` 自己先没了
- 更像外部中断、shell/会话异常、或父进程被杀
- 不是脚本自身已有逻辑可以解释的正常退出

#### D2. `formal_training_scratch.log` 那次停止

**高概率：Isaac Sim 原生层崩溃，或被外部强制 kill。**

现象：

- 日志最后没有 `Final Eval`
- 没有 Python traceback
- 没有正常 `simulation_app.close()` 的收尾痕迹
- checkpoint 一直保存到 `checkpoint_183631872.pt`
- 日志还在继续推进到大约 `1.93e8` frames 后突然截断

结合此前 `auto_train.sh` 已多次记录 `exit code 139` 段错误，本次最像：

- 原生层 `SIGSEGV` / native crash
- 或外部 `SIGKILL`

当前环境下：

- `ulimit -c = 0`
- 系统未保存 core dump
- 普通用户无 `dmesg` 读取权限

所以目前无法把“native segfault”和“外部 kill”百分百区分，但可以排除“正常完成”。

### E. 为什么 `action_norm` 和 `entropy` 越来越大

这一部分分成“奖励层原因”和“实现层原因”。

#### E1. 奖励层原因

**已证实：当前奖励确实给了高熵策略继续漂移的空间。**

- 猎物速度仍高于追捕者：
  - `v_prey = 1.3`
  - `v_drone = 1.0`
- 稳定捕获极稀有，绝大多数时间 `catch_reward = 0`
- 因此主要优化信号仍来自：
  - `distance_reward`
  - `collision_reward`
  - `speed_reward`
- 同时：
  - `smoothness_coef = 0.0`
  - `entropy_coef = 0.01`

这会导致一个典型现象：

- 一旦策略学会“靠近一点、少撞一点”
- 但还抓不住
- PPO 就容易继续把分布做宽，以保留探索收益

#### E2. 实现层原因：`action_norm` 指标本身语义已经变了

**已证实：当前日志里的 `action_norm` 不是 raw policy action，而是 PID transform 后的控制器输出。**

证据链：

- 当前任务使用 `action_transform: PIDrate`
- `PIDRateController` transform 中会：
  - 先对原始策略动作 `torch.tanh(action)`
  - 再转成 `target_rate / target_thrust`
  - 再调用 controller 得到 `cmds`
  - 最后把 `("agents", "action")` 原地改写成 `cmds`
- 而 MAPPO 里的 `action_norm` 统计直接取 `tensordict[self.act_name]`

因此当前 TensorBoard 中的 `drone/action_norm` 实际是在统计：

- controller 输出命令范数
- 不是 actor 原始输出范数

这解释了为什么它可以从个位数一路飙到上百甚至五百以上。

#### E3. 实现层高风险疑点：PPO 很可能在错误的动作空间里计算 `log_prob`

**待验证，但这是目前最值得优先核对的代码级疑点。**

现象链如下：

- MAPPO 的 `act_name = ("agents", "action")`
- actor 的输入键也包含这个 `action`
- `update_actor()` 中用 batch 里的 `action` 来重新计算 `log_probs_new`
- 但 `PIDRateController` transform 已经把该键改写成 controller `cmds`

如果 rollout buffer 里存下来的 `("agents", "action")` 确实是 transform 后的 command，而不是 raw policy action，那么 PPO 更新时就会出现：

- 用 raw Gaussian policy
- 去给 controller 输出空间的大数值动作算 `log_prob`

这样会直接迫使策略：

- 不断增大 `std`
- 从而抬高 `entropy`
- 并连带使 `action_norm` 膨胀

这和当前观测高度一致：

- `entropy` 从约 `8` 一路涨到 `27+`
- `action_norm` 从个位数涨到 `500+`
- 同时 actor 梯度并没有爆炸，只是分布越来越宽

#### E4. 为什么动作很大却没有立刻数值炸掉

**已证实：物理执行端最后还会把电机命令夹住。**

在 rotor 模型里，最终会做：

- `torch.clamp((cmds + 1) / 2, 0, 1)`

这意味着：

- 策略端数值可以越变越大
- 但物理上很多都会被压成相同的饱和推力

结果就是：

- 日志层看，动作越来越大
- 物理层看，很多动作其实都已经“打满”
- 策略学到的是饱和控制，而不是细腻控制

### F. 当前最需要 antigravity 核对的点

优先级从高到低建议如下：

1. **核对 PPO 更新时使用的 `("agents", "action")` 到底是 raw action 还是 PID transform 后的 `cmds`**
   - 这是当前最关键的实现正确性问题
   - 若确认是后者，应视为训练目标错位
2. **核对 `drone/action_norm` 指标是否应改为记录 raw policy action**
   - 当前这条曲线很容易被误读
3. **核对训练停止时的系统级退出原因**
   - 若能访问 `dmesg` / journal / apport，优先确认是否 native segfault
4. **确认是否需要下调 `entropy_coef`**
   - 但这个应在 action pipeline 核对完成后再动手

### G. 当前最稳妥的结论

截至 2026-03-25 这批日志，最稳妥的表述是：

- 训练没有完全失败：策略已明显学会追近目标并减少碰撞
- 训练也没有真正成功：只出现过极少量偶发捕获，尚未形成稳定围捕能力
- `success` 曲线怪异主要是因为极稀疏非零事件被 TensorBoard 平滑放大
- `action_norm` 与 `entropy` 的同步膨胀，不能只归因于奖励设计
- 代码层面存在一个高风险疑点：`PIDrate` action transform 可能污染了 PPO 的动作概率计算空间

### H. 2026-03-25 修整结果

基于上面的代码核对，已经落实了以下修整：

1. **修正 PPO 动作空间对齐**
   - `PIDRateController` 仍会把环境执行动作写回 `("agents", "action")`
   - 但现在额外保留 `("agents", "action_raw")`
   - MAPPO 的 `train_in_keys` 已显式保留 `action_raw`
   - `update_actor()` 在重算 `log_prob` 时优先使用 `action_raw`
   - 这一步的目标是确保：
     - PPO 始终在 raw policy action 空间里训练
     - controller `cmds` 只负责环境执行，不再污染策略分布更新

2. **收紧策略方差**
   - 将 `entropy_coef` 从 `0.01` 下调到 `0.001`
   - 为 actor 高斯头新增：
     - `log_std_init`
     - `log_std_min`
     - `log_std_max`
   - 并在前向里对 `log_std` 做 clamp
   - 这一步的目标是阻断 `std -> entropy -> 大动作` 的正反馈链条

3. **补充诊断指标**
   - `action_norm` 现在用于记录 raw policy action 范数
   - 新增 `cmd_norm`，用于记录 controller 输出范数
   - 新增：
     - `actor_log_std_mean`
     - `actor_log_std_min`
     - `actor_log_std_max`
   - 以后判断“动作变大”时，要先区分：
     - 是策略原始输出变大
     - 还是 PID/controller 命令变大

4. **奖励侧做小幅稳定化，不重写主目标**
   - 保留当前：
     - `catch_reward_coef = 200`
     - `dist_reward_coef = 2`
     - `collision_coef = 20`
   - 仅做两处保守修整：
     - `smoothness_coef: 0.05`
     - `speed_reward` 由阈值二值惩罚改成连续超速惩罚
   - 修整意图是：
     - 不削弱稀疏成功信号
     - 但让“持续大动作 / 超速冲刺 / 饱和控制”开始付出连续代价

### I. 修整后的预期观测

如果上述修整生效，下一轮短程训练里更合理的现象应是：

- `actor_log_std_mean/max` 不再单调持续上升
- `action_norm` 维持在有限范围内，而不是持续发散
- `cmd_norm` 可能仍高于 `action_norm`，但不应出现指数式膨胀
- `entropy` 不再一路抬升到 20、30 以上
- `success` 不一定立刻显著提升
- 但行为应从“越来越随机、越来越猛”变成“更稳定地追、偶发成功后可继续复现”

若修整后仍然出现：

- `actor_log_std_max` 持续顶到上界
- `action_norm` 和 `cmd_norm` 同时继续上升
- `success` 仍长期接近 0

则下一步应优先考虑：

1. 再进一步下调 `entropy_coef`
2. 上调 `smoothness_coef`
3. 重新评估 `collision_coef` 是否仍过低
4. 引入更明确的“围堵完成”过程奖励，而不只依赖最终 `catch_reward`

### J. 2026-03-25 修整后长程训练复盘（`HideAndSeek_20260325_182624`）

这一节记录修整后的第一条长程 run，重点回答三个问题：

- 新修复是否压住了之前的 `entropy/action_norm` 失控
- 当前奖励与参数设置下，策略究竟学到了什么
- 为什么这次前期有成功，后期又回到 0

#### J1. 本次 run 的参数设置

**运行方式**

- run 名称：`HideAndSeek_20260325_182624`
- 启动方式：从头训练，`model_dir: null`
- 总帧数：event 文件最后一步为 `524,288,000`
- TensorBoard 路径：`runs/HideAndSeek_20260325_182624/`
- 环境：
  - `task=HideAndSeek`
  - `task.use_eval=1`
  - `task.use_random_cylinder=0`
  - `task.scenario_flag=empty`
  - `task.env.num_envs=2048`
  - `action_transform=PIDrate`

**本节结果图对应的关键 TensorBoard tags**

- 优化稳定性：
  - `drone/ESS`
  - `drone/actor_grad_norm`
  - `drone/advantages_mean`
  - `drone/advantages_std`
- 探索强度：
  - `drone/entropy`
  - `drone/actor_log_std_min`
  - `drone/actor_log_std_mean`
  - `drone/actor_log_std_max`
- 动作与平滑性：
  - `drone/action_norm`
  - `drone/cmd_norm`
  - `train/stats.action_error_order1_mean`
  - `train/stats.action_error_order1_max`
- 任务表现：
  - `train/stats.success`
  - `train/stats.catch_reward`
  - `train/stats.distance_reward`
  - `train/stats.collision`
  - `train/stats.collision_drone`
  - `train/stats.collision_wall`
  - `train/stats.collision_reward`

**当前奖励设置**

- `dist_reward_coef = 2.0`
- `catch_reward_coef = 200.0`
- `detect_reward_coef = 0.0`
- `collision_coef = 20.0`
- `speed_coef = 2.0`
- `smoothness_coef = 0.05`
- `catch_radius = 0.4`

**当前算法设置**

- `entropy_coef = 0.001`
- `log_std_init = 0.0`
- `log_std_min = -5.0`
- `log_std_max = 1.0`

**额外说明**

- 这次 reward 仍然是：
  - `distance_reward + detect_reward + catch_reward + collision_reward + speed_reward + smoothness_reward`
- 其中：
  - `distance_reward` 是距离进展奖励，不再是简单的负距离
  - `speed_reward` 已改为连续超速惩罚
  - `smoothness_reward` 已启用
- 但当前 `train/stats.smoothness_coef` 标量仍显示为 `0.0`
  - 这是统计项没有同步写入，不代表真实配置为 0

#### J2. 结果图的核心事实

**总体结论**

- 这次修整后的 run，确实压住了以前那种“熵一路向上爆炸、动作数值一路发散”的问题。
- 但它没有直接变成稳定成功，而是出现了另一种更典型的模式：
  - 前期仍有探索和少量成功
  - 中后期策略逐渐收缩成低碰撞、低熵、较平滑的“安全追逐策略”
  - 成功率在约 `100M` 之后重新归零

**按指标看：**

1. **熵不再爆炸，而是持续下降**
   - `drone/entropy`
   - 起点：`5.669`
   - 终点：`0.428`
   - 最低到：`-2.048`
- 这里的负值并不奇怪。
  - 对连续高斯分布，方差足够小的时候 differential entropy 可以是负数。
  - 这说明当前策略不是“越来越随机”，而是“越来越尖、越来越确定”。

2. **`actor_log_std` 明确收缩**
   - `actor_log_std_mean`
     - 起点约 `0`
     - 在 `347M` 左右降到最低 `-1.931`
     - 末尾回升到 `-1.310`
   - `actor_log_std_min`
     - 最低达到 `-3.149`
   - `actor_log_std_max`
     - 曾在 `152M` 左右短暂为正（最大 `0.176`）
     - 之后整体下行，末尾约 `-0.156`
- 这说明：
  - 修整后的探索并没有失控
  - 反而在中后期被明显压缩了

3. **`action_norm` 没有真正爆炸**
   - event 原始值显示：
     - 起点 `1.879`
     - 最低 `1.201`
     - 终点 `4.479`
- 因此截图中那个夸张的 `1.2e7` 纵轴，不是原始标量真实行为，更像 TensorBoard 显示异常或坐标轴缓存污染。

4. **`cmd_norm` 和 `action_norm` 完全重合**
   - 本次 event 中两条曲线逐点完全相同，最大绝对差为 `0.0`
- 这意味着：
  - 当前这版日志还没有真正把“raw policy action”和“controller command”观测分开
  - 因而本轮可以确认 PPO 不再出现明显熵爆炸
  - 但还不能通过 `cmd_norm` 单独判断执行端是否存在额外饱和

5. **碰撞显著下降**
   - `train/stats.collision`
     - `0.447 -> 0.0148`
     - 最低约 `0.00722`（约 `218M`）
   - `train/stats.collision_drone`
     - `0.214 -> 0.00919`
   - `train/stats.collision_reward`
     - `-5.562 -> -0.198`
   - `train/stats.collision_wall`
     - `0.0638 -> 0.00072`
- 说明策略显著学会了规避互撞和撞墙。

6. **距离进展奖励有效提升**
   - `train/stats.distance_reward`
   - `0.000529 -> 0.001094`
   - 峰值 `0.001396` 出现在约 `57.5M`
- 说明策略确实更会追近目标。

7. **动作平滑性明显提升**
   - `train/stats.action_error_order1_mean`
     - `1.688 -> 0.312`
     - 最低约 `0.211`
   - `train/stats.action_error_order1_max`
     - `2.599 -> 1.609`
     - 最低约 `1.317`
- 说明修整后的策略确实比旧 run 平滑很多。

8. **TP 网络很健康**
   - `TP_loss`
     - `0.154 -> 1.63e-5`
   - `TP_valid_windows`
     - 基本保持在 `120,832`
   - `TP_valid_windows_min`
     - 大部分时间是 `59`
     - 最低降到 `54`
- 这说明目标预测支路没有拖后腿，数据窗也基本稳定。

9. **中期存在一次明显的 PPO 大更新/分布收缩阶段**
   - `ESS` 最低跌到 `0.0426`（约 `259M`）
   - `actor_grad_norm` 最高达到 `99.6`（约 `292M`）
- 这说明在 `200M-320M` 附近出现了一次相当激烈的策略重排。
- 但它没有把训练炸掉，而是把策略推向了更低熵、更保守的区域。

#### J3. 成功到底有没有

**有，而且比旧 run 明显更多，但只集中在前期。**

- `train/stats.success` 非零一共出现 `17` 次
- 这些非零点全部集中在：
  - `21.5M` 到 `100.1M`
- 最高值出现在：
  - `57.54M`
  - 峰值 `0.01056`

对应 `train/stats.catch_reward`：

- 同样非零 `17` 次
- 峰值也出现在 `57.54M`
- 峰值 `0.01615`

这说明：

- 这次 run 前期确实抓到过，而且不止一次
- 成功密度明显高于此前那种 `1/2048` 级别的偶发命中
- 但这个能力没有被维持住
- 在约 `100M` 之后，成功和抓捕奖励都重新归零

#### J4. 这次 run 的阶段性分析

**阶段 A：0M - 100M**

- 探索尚在，熵仍然不低
- 碰撞快速下降
- 距离奖励快速提升
- 出现一串真实成功点

这是这次 run 的最好阶段。  
一句话概括：**“会追、会抓几次、但还不稳。”**

**阶段 B：100M - 200M**

- 成功开始消失
- `log_std` 继续下降
- 熵逐步收缩
- 策略开始从“有探索的追捕”往“更稳的追逐”迁移

一句话概括：**“探索在退，安全性在升，但抓捕能力没有继续巩固。”**

**阶段 C：200M - 350M**

- 出现明显的 ESS 下跌和 actor 梯度尖峰
- `log_std_mean/min/max` 同时大幅下探
- `advantages_std` 也一路压低到极小值

这一段最像：

- PPO 发生了一次比较强的策略压缩
- 将原本还能产生成功的策略，进一步推成了“低熵、低碰撞、低风险”的收缩版本

一句话概括：**“从有一定侵略性的追捕策略，收缩成过于保守的确定性策略。”**

**阶段 D：350M - 524M**

- 熵维持低位
- 碰撞维持低位
- 平滑性维持较好
- `action_norm` 末段有回升，但不是熵爆炸
- 成功始终为 0

一句话概括：**“训练后期很稳，但稳成了不会抓人的安全追逐器。”**

#### J5. 这次结果最重要的结论

与 `153617` 那种“熵越来越大、动作越来越乱”的失败模式相比，这次结果说明：

1. **修整后的代码已经基本修住了“探索无限膨胀”**
   - `entropy` 没有再爆炸
   - `log_std` 没有失控上扬
   - `action_norm` 原始值也没有发散

2. **当前的主要矛盾已经变了**
   - 不再是“策略越来越随机”
   - 而是“策略后期收得太死，变成低碰撞但低成功的确定性追逐”

3. **当前奖励与参数的整体倾向是**
   - 足以让策略学会追近、避撞、平滑
   - 但还不足以把“偶发成功”固化成“长期稳定成功”

4. **换句话说**
   - 旧问题是：探索失控
   - 新问题是：探索收缩过度，成功没能保持

#### J6. 针对这次 run 的稳妥建议

基于这次结果，下一步不建议再回到旧版那种大熵、无遮挡地放探索。  
更稳妥的方向是：

1. **维持 `log_std` 上下界，不要撤销**
   - 这次它确实拦住了熵爆炸

2. **`entropy_coef` 不宜继续下调**
   - 当前更像探索不够，而不是探索过强
   - 若再降，后期更容易过早塌成保守策略

3. **若要调，优先考虑小幅回调探索，而不是大改奖励**
   - 例如温和上调 `entropy_coef`
   - 或者改成随训练衰减而不是常数

4. **当前 reward 主体不必推翻**
   - `distance progress`
   - `catch reward`
   - `collision penalty`
   - `continuous speed penalty`
   - `smoothness reward`
   这套组合方向是对的

5. **更值得补的是“把早期成功保住”的机制**
   - 例如课程学习
   - 或在成功曾经出现的区间附近做 checkpoint 回溯
   - 观察 `57M-100M` 一带的策略录像与评估表现

#### J7. 本次 run 的一句话归纳

这次修整后的长程训练结果可以概括为：

- **修住了熵爆炸**
- **学会了更稳、更少撞、更平滑地追**
- **前期出现了比以前明显更多的真实成功**
- **但后期策略收缩过头，重新退化成“不会抓人的安全追逐器”**

---



## L. 2026-03-26 长程训练复盘与诊断（HideAndSeek_20260326_002011）

### L1. 现象 (Run Summary)
本轮 run 训练了近 4 亿步（最高到 `checkpoint_511311872.pt`），根据 TensorBoard 数据提取：
- `success` 与 `catch_reward` 始终为绝对 0，没有捕获目标。
- 控制变得极其平滑与保守，无人机之间学会了避免互相碰撞：`collision_drone` 从 0.17 锐减到 0.07。
- 策略分布急剧坍缩（完全确定性控制）：`drone/entropy` 呈直线大反转，从初始的 `5.07` 跌至极低的 `-3.24`。
- “贴地倒下”的频率变长：`landed_penalty` 的负向惩罚比初始恶化了许多。

### L2. 可能根因 (Root-cause hypothesis)
由以上现象推断，网络完全陷入了**“安全尾随、拒绝冒险（避免 -20 的互撞惩罚），乃至干脆就地降落摆烂”**的三维空间局部最优点。
之所以迟迟走不出局部最优，其最大的逻辑 Bug 是：**课程式学习（Curriculum Learning）被配置覆盖（旁路）了。**
在 `hideandseek.py` 中虽然有 `[v_prey += 0.05]` 的自动进阶逻辑，但 `cfg/task/HideAndSeek.yaml` 中的初始参数写死了 `v_prey = 1.3 (而无人机速度上限为 1.0)`。这导致猎物速度刚开始就达到了系统允许的绝对上限 `min(1.3, self.v_prey)`！由于猎物比追捕者快 1.3 倍，单向尾随无论如何都不可能带来抓捕奖励。由于成功信号完全断链，没有奖励信号告诉网络“需要从前方合围”，网络最终学会的只有：如何飞得平稳且不互相碰撞即可止损。

### L3. 证据 (Evidence)
- `train/stats.first_capture_step` 自始至终是 800（表示 0 次捕获）。
- 策略网络熵值 (-3.24) 已经证明策略完全收敛到了不抓捕的状态，而无任何探索增益。
- 代码 1113 行为 `self.v_prey = min(1.3, self.v_prey)`，而配置文件初始强加的最大值使 curriculum 失效。

### L4. 建议修改 (Minimal changes)
将 `HideAndSeek.yaml` 下的 `v_prey` 由 `1.3` 修改为 `1.0`。
通过速度对等，网络早期能从各种角度乃至从正后方体验到 `success`。当成功率达标后，代码内建的 curriculum 会将逃跑者难度稳步加至 1.3。

### L5. 风险
由于 `v_prey` 被削弱，模型早期的拟合可能会退化为极其纯粹的纯尾随游戏（即它能在猎物 1.0 速度时抓捕，却学不会高级夹击）；当课程进阶导致 `v_prey` 大于 `1.1` 时如果失败率暴增，届时可能依然需要根据启发式补充设计“多角度包围网 (Encirclement) 惩罚”。

### L6. 下一轮要验证什么 (Next-run success criteria & Resume Strategy)
**Resume Strategy**: `restart-scratch`
由于旧节点的神经网络极度坚信当前的确定性行动（Entropy: -3.24），旧权重具有强烈的“不作为毒性”。我们直接开启从零重训，评估初期能否在 10M-50M frame 期间首次看到 0 字突破及 `success` 曲线上翘。

---

## M. 2026-03-26 引入几何包围奖励（HideAndSeek_20260326_163903）

### M1. 现象 (Run Summary)
在修改 `v_prey = 1.0` 后重新进行时长约几个小时的测试运行（Run 163903）。即便速度已经抹平甚至持平，最终统计结果表明 `train/stats.success` 居然依旧是 0。
同时，`drone/entropy` 在该新版本依然从高位平滑暴跌至 -2.9，且 `first_capture_step` 坚若磐石地钉在 800。

### M2. 可能根因 (Root-cause hypothesis)
人工势场 (APF) 控制逃逸者的反向排斥力过于完美。即使多智能体与逃跑者速度相当（均为 1.0），当无人机单纯在后面展开线性尾随（直线迫近）时，APF 会促使红球精准向绝对反方向匀速逃窜。
物理学规律表明，速度齐平条件下的尾追是不可能通过时间缩短距离差的（必须依赖三台机体形成夹角网，切断退路才能捕获）。但当前系统中没有任何针对“立体包抄”的先期学习指导信号，因此网络无从顿悟合围策略，最终再次落回了安全尾随（避免 `collision_drone` 惩罚）的死循环局部最优点。

### M3. 证据 (Evidence)
- TensorBoard 中记录到 `first_capture_step` 依旧维持在最大判定值 800 步。
- 策略方差（Entropy）再次逼近 -3.0，意味着当前解已被视为唯一解，不再进行探索。
- 速度平衡并没有使得模型自然学会合围。

### M4. 建议修改 (Minimal changes)
必须主动引入**几何包抄奖励 (Encircle Reward)**，切断死脑筋跟踪的回路。
我们在 `_compute_reward_and_done` 增加对**追捕者质心 (Center of Mass)** 和目标质心之间距离 `com_dist` 的密集指引：
新奖励设为 `encircle_coef * torch.exp(-com_dist)`，其中 `encircle_coef: 5.0`。如果呈三足鼎立包抄猎物，质心将完美重合于猎物本体，获取最高额长时奖励。

### M5. 风险
通过质心距离进行奖励虽然极其优雅，但如果系数给得过强，可能会冲淡原有的 `catch_reward`，甚至导致机器人在早期为了维持一个虚假的质心重合状态（例如排成一个以猎物为圆心对称但半径远在 10m 之外的等边三角形）而拒不开火靠近。此种情况需在监控中注意。

### M6. 下一轮要验证什么 (Next-run success criteria & Resume Strategy)
**Resume Strategy**: `restart-scratch`
由于之前的权重未包含此项高权重的奖励反馈，直接清空 `auto_train_empty` 残留数据全新起跑。必须观测新版张量板中新增记录节点 `encircle_reward` 的爬升状态，借此预期策略真正涌现真正的多点多向突破。

---

## N. 2026-03-27 用预测增强的包围几何替换质心奖励（下一轮实验设计）

### N1. 现象 (Run Summary)
最新一轮长训虽把成功率抬到约 16%，但测评视频显示大量 episode 在中途以 `any_landed` 触发结束。行为层面看，策略已经不是完全不会追，而是只会“安全尾随”；一旦尝试压上做拦截，就容易掉高度或姿态失控。

### N2. 主导问题 (Primary failure mode)
`over-conservative pursuit` + `physics recovery failure`

### N3. 根因假设 (Root-cause hypothesis)
当前 `encircle_reward = exp(-com_dist)` 只要求团队质心靠近目标，不要求多机从不同方位展开，因此它和“扎堆追尾”是兼容的。与此同时，TP 预测只被奖励给单个固定编号无人机，这与共享 actor 的对称训练信号不一致，多机优势没有形成稳定梯度。

### N4. 最小改动 (Minimal changes)
- 增加 `distance_predicted_reward`，让所有 pursuer 都对 TP 预测终点获得 progress reward。
- 将 `encircle_reward` 改成“角度覆盖 + 包围面积收缩”：
  - 参考点使用 `当前目标 + 预测目标 blend`
  - 用最大角度 gap 和角度均匀性判断是否形成多侧包围
  - 用 pursuer 围成的多边形面积鼓励继续缩圈
- 将 `intercept_reward` 改成团队级前向截断奖励，鼓励至少一机抢到目标未来轨迹前方。
- 将 `smoothness_reward` 改为零基线惩罚，取消“只要动作平稳就能白拿正分”的 loophole。

### N5. Resume Strategy
**restart-scratch**

原因是 reward semantics 已从“质心贴近”改成“预测驱动的真实包围几何”，旧 critic 的价值基线会明显失配。

### N6. 下一轮成功标准 (Next-run success criteria)
- `train/stats.distance_predicted_reward` 在前 10M-30M frame 内稳定转正
- `train/stats.coverage_reward` 和 `train/stats.encircle_reward` 有同步抬升，而不是只有一个项虚高
- `train/stats.any_landed` 明显低于上一轮同训练阶段
- `train/stats.success` 不再在 0.16 左右长期横盘

---

## O. 2026-03-27 预测增强的包围几何测试结果 (HideAndSeek_20260327_123717)

### O1. 现象 (Run Summary)
对应用了 N 阶段设计的最新权重 `HideAndSeek_20260327_123717/checkpoint_511311872.pt` 进行测评：
- **评估数据**：100 环境并发的大规模测试中 `success` 精确定格在 16.0%，而在单环境强制渲染跟踪的完整 rollout 中，无人机在第 236 步直接以 `any_landed`（坠机撞地）硬性终止，该局 `success` 归 0。
- **环境指标表现**：`distance_predicted_reward` 为正 (0.052)，说明策略在积极抢占未来点；但 `smoothness_reward` 为负 (-0.007)，`speed_reward` 严重惩罚 (-0.362)，且 `landed_penalty` 遭到触发（-0.084）。

### O2. 可能根因 (Root-cause hypothesis)
N 配置引入的预测拦截 (`intercept_reward`) 与包围覆盖 (`encircle_reward`) 成功切断了单纯跟屁虫式的尾随逻辑（使得无人机试图飞向猎物移动前方的几何占位点）。
然而执行侧崩溃了：底层的飞行控制器和现行的平滑惩罚（`smoothness_coef=0.1`）无法拉住这种激进的控制输出。尤其是当猎物因被堵截而突然转向时，无人机试图执行超出物理极限（或由于没学会平滑转向导致的剧烈滚转）的急转弯，随之发生严重的掉高与姿态失稳，重重撞向地面触发 `any_landed` 死亡。这导致成功率“横盘”在 16%，多数集数由于飞控失稳还没摸到猎物就自我坠毁。

### O3. 证据 (Evidence)
- 与 N6 的期望直接证伪：`train/stats.success` **依然在 16% 横盘打转**，没有突破。
- 与 N6 期望相悖：`any_landed` 问题不但没有消失，反而在单机跟随评估中成为致命终止原因。
- 只有一项确实符合 N6 的设定，即 `dist_predicted_reward` 如预期转正，证实了奖励逻辑本身是被模型吃到了，确实在学抢位。

### O4. 建议修改 (Minimal changes)
为了化解“思维前卫，但四肢失调”的矛盾，重心必须立刻从宏观战术指标上挪回到微观姿态控制上：
1. **大幅提升稳定飞行的强制拉力**：上调 `smoothness_coef`（如升至 `0.3`~`0.5`）。
2. **削弱几何包围过于苛刻的空间压迫**：适当柔化 `distance_predicted_coef` 或放宽 `coverage_slack`，避免为了保持这丁点的占位便宜而牺牲飞行高度。

### O5. 风险
大幅提升平滑代价（`smoothness_coef`）极易重新扼杀好不容易萌芽的包围预测意识，使得模型发现“还是安安稳稳尾随最不扣分”，在战术上发生退化。

### O6. 下一轮要验证什么
**Resume Strategy**: `resume-latest`
继续基于当前的 5.11 亿步模型上微调训练，下一轮需要验证的是：增加平滑惩罚后，能否在保住 `distance_predicted_reward > 0` 的意识前提下，把 `any_landed` （撞地坠毁）的频率压制下去，从而兑现本来应该产生的合击成功率，将 `success` 成功带出 16% 的泥潭。

---

## P. 2026-03-27 对 O 阶段的复核：并非“什么都没学会”，而是学成了不稳定的抢前截击

### P1. 复核结果 (Evidence Update)
对 `runs/HideAndSeek_20260327_123717` 的完整 TensorBoard 重新解析，并复跑
`checkpoints/HideAndSeek_20260327_123717/checkpoint_511311872.pt` 的单环境评测后，得到以下更精确的结论：
- `success` 并非从头到尾为 0，而是在早期曾达到 `0.277`，随后退化并长期停在 `0.163` 左右。
- 单环境复核与视频一致：episode 在 **第 235 步** 以 `any_landed` 终止，属于稳定可复现的坠机终止，而不是偶发视频异常。
- `coverage_reward` 与 `encircle_reward` 全程几乎贴地（末值约 `0.0003 / 0.0006`），说明“真实多侧包围”基本没有学出来。
- 真正被策略持续吃到的是 `intercept_reward`（末值约 `0.108`），而 `distance_predicted_reward` 反而从早期正值翻成后期负值。
- 控制分布后期重新变激进：`drone/action_norm` 从 `3.10` 抬到 `11.23`，`drone/entropy` 末值回到 `7.71`，`actor_log_std_max` 重新顶到配置上限 `1.0`。

### P2. 修正后的根因 (Root-cause hypothesis)
O 阶段把问题主要归因于“平滑惩罚不够”，这个方向只说对了一半。

更准确的病理是：
1. **奖励侧**：当前真正激活的不是 `coverage/encircle`，而是孤立的 `intercept_reward`。策略学到的是“抢到目标前方”，不是“多机展开后再缩圈”。
2. **控制侧**：虽然一阶动作变化 (`action_error_order1_mean`) 比旧 run 更小，但绝对动作幅值却大幅升高，说明策略在“更平滑地打大舵”，而不是“更稳地飞”。
3. **优化侧**：后期 `entropy` 与 `actor_log_std_max` 一起回升，表明策略分布没有收住；在激进的前向截击奖励下，这会把机体推向大幅、持续的危险动作，最终掉高撞地。

### P3. 可执行的最小方案 (Actionable fixes)
下一轮不要只把 `smoothness_coef` 粗暴拉高。更可执行、且更符合证据的最小改动是：
1. **先削弱真正活跃的冒进信号**：将 `intercept_coef` 从 `1.0` 下调到 `0.3 ~ 0.5`。
2. **让截击必须依赖团队展开**：把 `intercept_reward` 乘上一个 `coverage_reward` 的软门控，避免“单机抢前、其余两机没到位”时也持续拿高分。
3. **加直接的防坠机约束，而不是只加 smoothness**：
   - 低空下沉惩罚：当 `z < 0.22 ~ 0.25` 且 `vz < 0` 时额外扣分。
   - 绝对动作幅值惩罚：对持续过大的 `action_norm/cmd_norm` 增加惩罚。
4. **收探索上限**：将 `entropy_coef` 从 `0.01` 降到 `0.003 ~ 0.005`，并把 `actor.log_std_max` 从 `1.0` 收到 `0.5 ~ 0.7`。

### P4. Resume Strategy
**resume-latest**

原因是这颗 5.11 亿步权重已经学到一部分“抢前”意识，不值得推倒重来；当前更像是给它补控制护栏，而不是重写任务定义。
