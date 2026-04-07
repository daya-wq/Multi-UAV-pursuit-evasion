# RL Warm-Start 失败分析交接（给 Claude）

## 1. 任务背景

我当前的主线不是从零开始训 RL，而是：

1. 先用专家策略做模仿学习 / DAgger。
2. 得到一个已经有不错成功率的 actor。
3. 再把这个 actor 当作 warm-start，接 PPO/MAPPO 做 RL fine-tune。

这条线里，当前最好的 warm-start actor 是：

- `checkpoints/dagger_best_preserved_20260405_gru52_goal23_landed17.pt`

它在独立评测里的结果大约是：

- `Capture 51.6%`
- `Goal 23.4%`
- `Landed 17.2%`
- `Timeout 7.8%`

对应记录见：

- `docs/expert_bc_handoff_20260404.md`
- `dagger_gru_prevact_frozen_probe.log`

也就是说，**warm-start actor 本身不是废的**，它已经有中等偏上的抓捕能力。


## 2. 遇到的核心问题

我把这个 actor 接到 RL 里以后，训练并没有把剩下那部分能力学出来，反而把已有能力冲掉了。

当前最关键的失败 run 是：

- checkpoint dir: `checkpoints/HideAndSeek_20260405_183521`
- run dir: `runs/HideAndSeek_20260405_183521`
- log: `outputs/2026-04-05/18-35-15/train.log`
- final ckpt: `checkpoints/HideAndSeek_20260405_183521/checkpoint_final.pt`

这轮是正常跑完的，不是崩溃中断。

它的启动参数是：

- `model_dir=checkpoints/dagger_best_preserved_20260405_gru52_goal23_landed17.pt`
- `total_frames=200000000`
- `task.env.num_envs=1024`
- `task.env.max_episode_length=1000`
- `task.curriculum.enabled=false`
- `task.use_eval=0`
- `task.v_drone=1.5`
- `task.v_prey=1.0`
- `algo.actor.lr=1e-4`
- `algo.ppo_epochs=2`
- `algo.max_grad_norm=5.0`
- `algo.entropy_schedule.enabled=false`
- `algo.entropy_coef=0.001`
- `algo.actor.log_std_max=0.2`

配置记录见：

- `outputs/2026-04-05/18-35-15/.hydra/overrides.yaml`
- `outputs/2026-04-05/18-35-15/.hydra/config.yaml`


## 3. 最终结果很差，具体差在哪里

这轮最终不是“数值爆炸”，而是**稳定地学成了一个更差的策略**。

最终指标大约是：

- `train/stats.success ≈ 0.0048`
- `train/stats.catch_reward ≈ 0.117`
- `train/stats.first_capture_step ≈ 995`
- `train/stats.return ≈ -44.47`
- `train/stats.target_predicted_error ≈ 0.032`
- `train/stats.any_landed ≈ 0.988`
- `eval/stats.success = 0.0`
- `eval/stats.catch_reward = 0.0`
- `eval/stats.first_capture_step = 1000`
- `eval/stats.return ≈ -300.78`
- `eval/stats.target_predicted_error ≈ 7.20`
- `eval/stats.any_landed ≈ 0.0039`

大白话解释：

- 训练时几乎局局落地。
- 评测时倒不是一直摔，但基本抓不到。
- 它有一定跟踪能力，但收不了口。
- 最后的 RL actor 比 warm-start 起点明显差。


## 4. 我是怎么分析的

我不是只看最后一个点，而是按下面顺序排查的：

### 4.1 先确认是不是正常结束

从日志尾部可以确认：

- 这轮到 `199950336 steps` 时做了 `Final Eval`
- 保存了 `checkpoint_final.pt`

也就是说这是一个**完整跑完但结果差**的 run，不是异常终止。

### 4.2 再看 outcome 曲线

重点看：

- `train/stats.success`
- `train/stats.catch_reward`
- `train/stats.first_capture_step`
- `train/stats.return`
- `eval/stats.success`
- `eval/stats.return`

结论：

- `train success` 全程都很低，最高也只有大约 `3.7%`
- `eval success` 最高只有约 `0.2%`
- `catch_reward` 长期很小
- `first_capture_step` 长期接近 episode 上限

这说明：

- 它不是偶尔差一点点
- 而是**系统性地不会完成抓捕**

### 4.3 再看 chase quality

看了：

- `train/stats.target_predicted_error`

结论：

- 从早期 `0.18` 很快降到 `0.03` 左右
- 说明 target prediction / tracking 子系统并不差

所以问题不是“根本看不见目标”。

### 4.4 再看 failure cost

看了：

- `train/stats.any_landed`
- `train/stats.collision_drone`
- `train/stats.pursuer_collisions_count`
- `train/stats.collision_floor`
- `eval/stats.any_landed`
- `eval/stats.collision_floor`

结论非常关键：

- 训练时：`any_landed ≈ 0.988`
- 评测时：`any_landed ≈ 0.0039`

这说明：

- deterministic policy 不是一上来就完全不会飞
- 但训练时带探索采样后，on-policy rollout 几乎局局进坏状态

另外：

- `eval/stats.collision_floor` 很大
- 说明评测里虽然不总是直接“landing done”，但经常贴地/擦地飞

### 4.5 再看 control behavior

看了：

- `drone/action_norm`
- `drone/cmd_norm`
- `train/stats.action_error_order1_mean`
- `train/stats.smoothness_mean`

这里发现一个很值得注意的现象：

- 早期 `action_norm` 大约只有 `2.38`
- 后期涨到 `31.36`
- `cmd_norm` 也显示成同样的量级

这看起来像“动作大得离谱”，于是我继续排查了代码里的定义。

### 4.6 最后查动作和熵的代码口径

我重点查了：

- `omni_drones/learning/mappo.py`
- `omni_drones/utils/torchrl/transforms.py`

确认了两个重要事实：

1. `action_norm` 记的是 **raw action**
   - 见 `mappo.py` 里 `train_info["action_norm"] = raw_action.norm(...)`

2. `PIDrate` 变换里会再做一次 `torch.tanh(action)`
   - 见 `transforms.py` 的 `_inv_call`

所以：

- `action_norm` 很大，不等于真实执行命令真的有 31 倍那么夸张
- 但这也明确说明：**raw policy 输出已经严重漂到饱和区**
- 当 raw action 到 30 左右时，`tanh(30)` 基本就是 1
- 也就是说，执行层面已经接近“长期满舵/满推力”


## 5. 熵是不是太大了

我的判断是：

**是，太大了，但更准确地说，是 warm-start 阶段的有效探索仍然过大。**

最终这轮里：

- `drone/entropy ≈ 5.76`
- `drone/actor_log_std_mean ≈ 0.0207`

而代码里当前 PPO 记录的这个 entropy，是根据 actor 的 Gaussian `log_std` 算出来的：

- 见 `omni_drones/learning/mappo.py` 里的 `_get_actor_entropy_bonus`

这意味着：

- 它描述的是 raw Gaussian 动作分布的熵
- 不是经过环境 `tanh` 后真实执行命令的“最终熵”

大白话说：

- raw-space 里探索还不小
- 同时 mean 又已经漂到了很大的值
- 于是训练时既有噪声，又有饱和
- 这两件事叠加，会把 warm-start 策略冲散


## 6. 当前最可信的根因判断

我现在认为主因不是单一一个，而是下面这组组合问题：

### 6.1 PPO 把已有的 imitation 行为冲散了

warm-start actor 起点本来有 `51.6%`。

但一进入 RL：

- 探索噪声让 rollout 分布偏离 expert/DAgger 成功轨迹
- PPO 用这些“偏掉后的轨迹”反过来更新 actor
- 结果把已有的抓捕行为冲散

### 6.2 动作参数化有 mismatch

当前 actor 优化的是 unbounded raw Gaussian。

但环境动作在 `PIDrate` transform 里又会做一次 `tanh`。

这会带来两个问题：

1. raw mean 可以越漂越大
2. 执行动作早就饱和，但 PPO 在 raw 空间里不一定“感觉得到”

结果就是：

- 策略会学成 raw output 非常大
- 但环境里其实只是长期贴边控制

### 6.3 训练时高噪声坏轨迹太多

证据是：

- `train any_landed` 极高
- `eval any_landed` 却很低

这说明现在 PPO 更新用到的大量数据，本身就是探索采样造成的坏轨迹。

这种数据会不断告诉 actor：

- 你当前这套 warm-start 行为也不稳定
- 于是 actor 被往坏方向改

### 6.4 这轮不是纯 optimizer 爆炸

这点也要说清楚。

前一轮失败过一次，确实是数值发散。
那次已经修过了：

- 熵项的实现
- 学习率
- PPO epochs
- grad clip
- log_std 上限

而这次 `HideAndSeek_20260405_183521`：

- 没再出现前一轮那种 `grad_norm=inf`
- 也没有熵爆到荒谬数量级

所以这轮更像是：

**稳定地收敛到坏局部解**

而不是“单纯又炸了”。


## 7. 我认为最值得 Claude 帮忙深入看的地方

我建议 Claude 不要泛泛地说“调奖励”或者“再降学习率”，而是优先围绕下面几个点做针对性分析。

### 7.1 动作参数化 / 动作变换的对齐问题

重点检查：

- `omni_drones/learning/mappo.py`
- `omni_drones/utils/torchrl/transforms.py`
- actor 分布定义在 `omni_drones/learning/modules/distributions.py`

我怀疑的点是：

- actor 现在输出的是 raw Gaussian
- 环境里又做一次 `tanh`
- PPO 的 log_prob / entropy / update 都是在 raw action 口径上
- 但环境执行的是 squash 后动作

想请 Claude 帮忙判断：

- 这是不是 warm-start PPO 退化的核心结构问题
- 是否应该改成更一致的 action parameterization
- 比如让 actor 自己就是 `tanh` policy，或者在 PPO 里显式处理 squash 后的分布

### 7.2 warm-start 阶段如何避免把 expert 能力冲掉

重点检查：

- `scripts/train.py`
- `omni_drones/learning/mappo.py`

想请 Claude 帮忙分析：

- 从高成功率 DAgger actor 接 PPO 时，是否应该：
  - 前期冻结部分层
  - 降低 actor update 强度
  - 限制 log_std 更低
  - 用更保守的 KL / trust region
  - 或引入 imitation anchor / behavior regularization

因为现在这轮最明显的问题是：

- actor 不是没起点
- 而是 PPO 把已有的好行为冲掉了

### 7.3 为什么训练 rollout 和 eval 表现差这么多

这个现象很关键：

- train 几乎局局落地
- eval 却不是全摔，而是抓不到

这说明要么：

- exploration 噪声太强
- 要么 rollout/采样时某些状态分布和 eval 差异太大

请 Claude 特别排查：

- stochastic action training vs deterministic eval 的差异
- recurrent hidden state 在 rollout / eval / PPO update 之间有没有不一致
- 是否存在 on-policy sampling 本身把 actor 拖进坏状态的机制

### 7.4 `action_norm` / `cmd_norm` 的统计口径

目前日志里：

- `action_norm`
- `cmd_norm`

几乎是同一个量级，而且都很大。

但从代码看，`cmd_norm` 本来应该更接近 post-transform/controller 的命令。

请 Claude 帮忙确认：

- 这两个指标当前是不是统计口径有问题
- 是否需要新增更直接的：
  - post-tanh saturation fraction
  - actual target_rate norm
  - thrust saturation fraction

否则现在看曲线时很容易误判。


## 8. 我认为优先要改的文件

请 Claude 优先看这些文件。

### 8.1 PPO / actor update 主逻辑

- `omni_drones/learning/mappo.py`

重点看：

- `update_actor`
- entropy bonus 的实现
- actor log_std 统计
- action_norm / cmd_norm 的记录
- warm-start 载入后的训练行为

### 8.2 动作变换 / 控制接口

- `omni_drones/utils/torchrl/transforms.py`

重点看：

- `PIDrate` 的 `_inv_call`
- 这里的 `torch.tanh(action)`
- raw action 到 target_rate / target_thrust 的映射

### 8.3 actor 分布实现

- `omni_drones/learning/modules/distributions.py`

重点看：

- 当前 actor 到底是不是 raw Gaussian
- tanh / bounded action 的分布是否和 PPO update 完全对齐

### 8.4 RL 训练入口

- `scripts/train.py`

重点看：

- warm-start checkpoint 载入逻辑
- rollout / eval 调用方式
- 是否有能在 warm-start 前几百万步里保护 actor 的入口

### 8.5 配置

- `cfg/algo/mappo.yaml`
- `outputs/2026-04-05/18-35-15/.hydra/config.yaml`
- `outputs/2026-04-05/18-35-15/.hydra/overrides.yaml`

重点看：

- entropy / log_std / lr / ppo_epochs
- 这些参数对 warm-start 是否仍然太激进


## 9. 我希望 Claude 回答的核心问题

我最想问 Claude 的不是“再给几个调参建议”，而是这几个更具体的问题：

1. 当前这套 PPO + raw Gaussian + env 内部再 `tanh` 的动作管线，是否本身就容易让 warm-start actor 学坏？
2. 为什么 `51.6%` 的 DAgger actor，一接 PPO 就会被冲到 `eval success = 0`？
3. 训练里几乎局局 landed，但 eval 却不是全摔，这种 train/eval gap 的主因更像什么？
4. `action_norm` 持续上升到 `31+`，这在当前代码口径下意味着什么？应该怎样更正确地监控“真实动作是否饱和”？
5. 如果只允许最小修改，应该优先改：
   - action parameterization
   - PPO 正则 / trust region
   - warm-start 前期冻结 / imitation regularization
   - 还是 rollout/eval 的 recurrent state 处理？


## 10. 我当前自己的结论

如果只用一句话总结我自己的判断：

**这轮 RL 差，不是因为专家 actor 没用，而是因为 PPO 在当前动作参数化和探索设置下，把一个还不错的 warm-start 策略稳定地推成了“raw 输出很大、训练时高噪声乱飞、评测时跟踪还行但抓不到”的坏局部解。**

所以我现在最怀疑的不是 reward 本身，而是：

- warm-start PPO 的更新方式
- raw action 与执行动作之间的参数化不一致
- 以及高噪声 on-policy 数据把 actor 往坏方向拖

