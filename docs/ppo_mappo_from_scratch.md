# PPO/MAPPO 从 0 到代码实现

本文档面向这样的读者:

- 已经知道强化学习里的基本词汇: `state`, `action`, `reward`, `actor`, `critic`
- 但还不知道 PPO 训练时到底在做什么
- 希望把"算法概念"和"这个项目的代码"对上

本文会先讲最基础的 PPO 直觉，再讲这个仓库里的实现。需要提前说明一件事:

- 这个项目真正训练时用的是 `MAPPO`
- `MAPPO = Multi-Agent PPO`
- 它的核心训练思想仍然是 PPO
- 所以你可以先把它当成"多智能体版 PPO"


## 1. 先说结论: PPO 到底在做什么

一句话概括:

**PPO 做的事情是: 先让当前策略去环境里跑一段，收集一批经验，然后用这批经验更新策略，但更新幅度不能太大。**

这里面有三个关键点:

1. 它不是一边走一步一边立刻更新，而是先收集一批数据
2. 它不是只更新 actor，也会更新 critic
3. 它不是让策略随便改，而是通过 `clip` 限制改动幅度

如果用更口语的话说:

- actor 负责"怎么行动"
- critic 负责"这一步到底值不值"
- PPO 通过 critic 给 actor 打分
- 但为了避免 actor 一次改太猛，把原来会的东西也改坏了，所以加一个"不要改太多"的约束


## 2. PPO 解决的是什么问题

如果你只做最原始的策略梯度，常见问题是:

- 更新不稳定
- 学着学着性能突然崩掉
- 一次更新步子太大，策略从"还行"直接跳成"很差"

PPO 的核心思想就是:

**我允许你朝更好的方向更新，但不允许你一次走太远。**

这就是 PPO 里 `Proximal` 的意思: 让新策略 stay close to old policy。


## 3. 训练一次 PPO，脑子里应该想什么

你可以把一次 PPO 训练分成下面 6 步:

1. 用旧策略和环境交互，收集一批轨迹
2. 用 critic 估计每一步的 value
3. 根据 reward 和 value 算 advantage
4. 用 advantage 告诉 actor: 哪些动作比预期好，哪些比预期差
5. 用 clipped objective 更新 actor
6. 用 returns 更新 critic

其中最重要的三个量是:

- `value`: critic 认为当前状态未来能拿多少回报
- `return`: 实际上这一步应该对应的训练目标
- `advantage`: 这一步动作比 critic 原本预期更好还是更差

最常见的一句总结是:

`advantage = return - value`

在这个项目里不是直接这么算，而是通过 `GAE` 先算 advantage，再由

`return = advantage + value`

得到训练 critic 的目标。


## 4. PPO 里的 actor 和 critic 分别干什么

### 4.1 actor 做什么

actor 输入状态，输出一个动作分布。

注意，不一定是直接输出某个确定动作，而通常是输出一个分布，比如:

- 连续动作: 高斯分布
- 离散动作: 分类分布

然后从这个分布里采样动作。

所以 actor 真正学的是:

**在某个状态下，什么动作更应该有更高概率。**


### 4.2 critic 做什么

critic 输入状态，输出一个标量 `V(s)`。

它表示:

**从这个状态开始，未来大概还能拿到多少累计回报。**

critic 不负责选动作，它负责给 actor 的行为提供"评价基准"。


## 5. rollout 是什么

PPO 训练时会反复出现 `rollout` 这个词。

你可以把 `rollout` 理解成:

**让当前策略在环境里实际跑一段时间，把这段经历完整记录下来。**

记录下来的内容一般包括:

- 当前状态 `s_t`
- 当前动作 `a_t`
- 当前动作在旧策略下的对数概率 `log pi_old(a_t | s_t)`
- 当前 critic 给出的 `V(s_t)`
- 奖励 `r_t`
- 下一时刻状态
- done 标志

在这个项目里，rollout 不是一整局 episode，而是"固定长度的一段"。

代码位置:

- `scripts/train.py`
- `omni_drones/utils/torchrl/collector.py`

训练入口里会构造:

```python
frames_per_batch = env.num_envs * int(cfg.algo.train_every)
collector = SyncDataCollector(
    env,
    policy=policy,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    device=cfg.sim.device,
    return_same_td=True,
)
```

见:

- `scripts/train.py`

这表示:

- 并行环境数是 `env.num_envs`
- 每个环境先跑 `train_every` 步
- 所有环境一起组成一批 rollout 数据

如果:

- `num_envs = 2048`
- `train_every = 64`

那么一次 rollout 的总采样量就是:

```text
2048 * 64 = 131072 environment steps
```


## 6. 这个项目里一次训练迭代到底发生了什么

下面按真实代码顺序讲。

### 6.1 训练脚本创建环境和策略

训练入口在:

- `scripts/train.py`

关键代码:

```python
base_env = env_class(cfg, headless=cfg.headless)
env = TransformedEnv(base_env, Compose(*transforms)).train()
policy = MAPPOPolicy(cfg.algo, agent_spec=agent_spec, device=cfg.sim.device, TP_net=base_env.TP)
```

这一步做了三件事:

1. 创建 Isaac Sim 环境
2. 给环境包上一些 TorchRL transform
3. 创建策略对象 `MAPPOPolicy`


### 6.2 collector 用当前策略采样 rollout

训练主循环是:

```python
for i, data in enumerate(pbar):
    ...
    info.update(policy.train_op(data.to_tensordict()))
```

也就是说:

- `collector` 每次先产出一批 `data`
- `data` 就是这次 rollout 收集到的轨迹
- 然后 `policy.train_op(...)` 用它做 PPO 更新


### 6.3 在 rollout 过程中，policy 会输出什么

`MAPPOPolicy.__call__()` 在:

- `omni_drones/learning/mappo.py`

它做的事情是:

1. 用 actor 根据 observation 产生动作
2. 记录该动作在旧策略下的 log probability
3. 用 critic 算当前状态 value

关键代码逻辑:

```python
tensordict.update(actor_output)
tensordict.update(self.value_op(tensordict))
```

也就是说 rollout 数据里，除了动作本身，还会保存:

- action
- action_logp
- state_value

这很关键，因为 PPO 更新时必须比较:

- 旧策略当时给这个动作的概率
- 新策略现在重新算这个动作的概率


## 7. 一个最核心的问题: advantage 到底是什么

先记一句最重要的话:

**advantage 说的不是"这一步奖励高不高"，而是"这个动作比 critic 原来预期得更好还是更差"。**

如果:

- `advantage > 0`，说明这个动作比预期好，应该提高它的概率
- `advantage < 0`，说明这个动作比预期差，应该降低它的概率

举一个极简例子:

- critic 觉得当前状态未来应该拿 `10`
- 实际上这一步往后看，你拿到了 `14`

那就说明:

- 这个动作比预期好
- advantage 是正的

反过来，如果实际只拿到 `6`，那 advantage 就是负的。


## 8. 为什么不直接用 reward，而要用 advantage

因为单步 reward 太短视。

PPO 关心的不是:

- "这一步是不是立刻得分"

而是:

- "从这一步开始，后面整体结果是不是比预期更好"

因此它要结合:

- 当前奖励
- 未来奖励
- critic 的 value baseline

最后得到 advantage。

这会比直接用 reward 稳定很多。


## 9. GAE 在这个项目里怎么计算

GAE 的代码在:

- `omni_drones/learning/utils/gae.py`

核心实现:

```python
delta = reward[:, step] + gamma * next_value * not_done[:, step] - value[:, step]
advantages[:, step] = gae = delta + (gamma * lmbda * not_done[:, step] * gae)
returns = advantages + value
```

### 9.1 逐项解释

- `reward[:, step]`: 当前这一步的奖励
- `next_value`: 下一时刻状态的 value
- `gamma`: 折扣因子，控制未来奖励看多远
- `done`: episode 是否结束
- `delta`: TD error，也可以理解成"当前这一步相对 critic 预期多出来多少"

其中:

```text
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

如果 episode 已经结束，那么 `not_done = 0`，未来项就不会继续往后传。


### 9.2 为什么 GAE 要从后往前算

因为当前这一步的 advantage，不只和当前 reward 有关，还和未来的误差累积有关。

所以代码里是:

```python
for step in reversed(range(num_steps)):
```

也就是从 rollout 的最后一步往前推。


### 9.3 GAE 解决了什么

如果你只用 Monte Carlo return:

- 无偏
- 但方差很大

如果你只用一步 TD:

- 方差低
- 但偏差更大

GAE 本质上是在 bias 和 variance 之间折中。

这个折中由 `lambda` 控制:

- `lambda` 大，更接近长时回报
- `lambda` 小，更接近一步 bootstrap

本项目默认:

- `gamma = 0.995`
- `gae_lambda = 0.95`

配置位置:

- `cfg/algo/mappo.yaml`


## 10. 为什么还要有 returns

在这个项目里:

```python
returns = advantages + value
```

这里的 `returns` 不是简单的 reward 累加，而是:

**给 critic 用的训练目标。**

所以分工是:

- actor 看 `advantages`
- critic 看 `returns`


## 11. PPO 最关键的地方: clip objective

这是 PPO 的灵魂。

actor update 在:

- `omni_drones/learning/mappo.py`

核心代码:

```python
ratio = torch.exp(log_probs_new - log_probs_old)
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
policy_loss = - torch.mean(torch.min(surr1, surr2) * self.act_dim)
```

### 11.1 `log_probs_old` 是什么

这是 rollout 时旧策略给当前动作的对数概率。

也就是:

```text
log pi_old(a_t | s_t)
```

它是在采样阶段记录下来的。


### 11.2 `log_probs_new` 是什么

这是训练时，把同样那批状态和动作重新喂给当前 actor 后，得到的新策略概率:

```text
log pi_new(a_t | s_t)
```


### 11.3 `ratio` 是什么

```text
ratio = pi_new / pi_old
```

如果:

- `ratio > 1`，说明新策略更倾向于这个动作
- `ratio < 1`，说明新策略更不倾向于这个动作


### 11.4 为什么 `ratio * advantage` 能更新策略

因为:

- 如果 `advantage > 0`，这个动作是好动作，就希望提高它概率
- 如果 `advantage < 0`，这个动作是坏动作，就希望降低它概率

这正好和 `ratio` 的方向对应。

例如:

#### 情况 A: 好动作

- `advantage = +2`
- `ratio = 1.1`

那 `ratio * advantage = 2.2`

这是好事，说明这个好动作被提高了概率。

#### 情况 B: 坏动作

- `advantage = -2`
- `ratio = 0.8`

那 `ratio * advantage = -1.6`

这也是对的，说明坏动作被降低了概率。


### 11.5 clip 到底在防什么

它防的是:

**新策略一下子改太狠。**

例如某个好动作:

- `advantage > 0`
- 但 `ratio` 直接冲到 `3.0`

那意味着新策略把这个动作概率抬得过高，可能造成训练不稳定。

于是 PPO 做:

```python
torch.clamp(ratio, 1 - clip, 1 + clip)
```

如果 `clip_param = 0.1`，那 ratio 就被限制在:

```text
[0.9, 1.1]
```

这不是硬性让参数不变，而是说:

**在 loss 里，不再奖励超过这个范围的进一步偏移。**


### 11.6 为什么取 `min(surr1, surr2)`

这是 PPO 非常关键的设计。

目标是保守更新。

对于好动作:

- 适度提高概率是好事
- 但提高太多就不额外鼓励了

对于坏动作:

- 适度降低概率是好事
- 但降低太多也不想一步走太远

所以 PPO 取保守版本。

你可以把它记成:

**PPO 总是倾向于选择那个更不激进的更新目标。**


## 12. critic 是怎么更新的

critic update 也在:

- `omni_drones/learning/mappo.py`

核心代码:

```python
values = self.value_op(critic_input)["state_value"]
b_values = batch["state_value"]
b_returns = batch["returns"]

value_pred_clipped = b_values + (values - b_values).clamp(-self.clip_param, self.clip_param)
value_loss_clipped = self.critic_loss_fn(b_returns, value_pred_clipped)
value_loss_original = self.critic_loss_fn(b_returns, values)
value_loss = torch.max(value_loss_original, value_loss_clipped)
```

这里和 actor 一样，也做了一个"保守更新"版本。

直觉上就是:

- critic 当然也想更接近 returns
- 但也不希望一次跳太猛

本项目里 critic loss 默认是:

- `HuberLoss`

配置位置:

- `cfg/algo/mappo.yaml`


## 13. entropy 在 PPO 里干什么

actor update 里还有一项:

```python
entropy_loss = - torch.mean(dist_entropy)
(policy_loss + entropy_loss * self.cfg.entropy_coef).backward()
```

熵的作用是:

**鼓励策略保持一定探索性，不要太快塌缩成特别死板的确定性策略。**

如果熵太低，策略会出现:

- 很早就只会一种动作
- 还没探索到更优解就过早收缩

如果熵太高:

- 动作太随机
- 学不稳定

所以 `entropy_coef` 是个平衡项。

本项目默认:

```text
entropy_coef: 0.0
```

这意味着当前配置基本不额外鼓励探索熵。


## 14. 为什么 PPO 能重复使用同一批 rollout 数据

这是 PPO 比普通 on-policy 方法更高效的一个地方。

在 `train_op()` 里:

```python
for ppo_epoch in range(self.ppo_epoch):
    dataset = make_dataset_naive(tensordict, int(self.cfg.num_minibatches), ...)
    for minibatch in dataset:
        update_actor(...)
        update_critic(...)
```

也就是说:

- 先采一批 rollout
- 这批 rollout 不只用一次
- 会被拆成多个 mini-batch
- 然后重复训练多个 epoch

本项目默认:

- `ppo_epochs = 4`
- `num_minibatches = 16`

意思是:

- 同一批 rollout
- 会被随机打散成 16 个小批次
- 一共循环 4 遍

注意:

这不等于它变成了离策略算法，因为它仍然只用"刚刚那一批旧策略采样的数据"。


## 15. mini-batch 在这个项目里怎么切

切分函数是:

- `make_dataset_naive()`

当 `seq_len == 1` 时，逻辑是:

```python
tensordict = tensordict.reshape(-1)
perm = torch.randperm(...).reshape(num_minibatches, -1)
for indices in perm:
    yield tensordict[indices]
```

这表示:

- 先把 `[N, T, ...]` 展平成一大堆样本
- 再随机打乱
- 切成若干 mini-batch

如果用了 RNN，才会按序列长度切。


## 16. 这个项目里一次完整训练更新的顺序

下面把 `MAPPOPolicy.train_op()` 拆成最基础的动作序列。

### 步骤 1: 切出训练需要的字段

```python
tensordict = tensordict.select(*self.train_in_keys, strict=False)
```

只保留 PPO 训练需要的内容，比如:

- observation
- action
- action_logp
- reward
- state_value
- next


### 步骤 2: 算最后一个时刻的 bootstrap value

```python
next_tensordict = tensordict["next"][:, -1]
with torch.no_grad():
    value_output = self.value_op(next_tensordict)
```

原因:

- GAE 需要最后一个 `next_value`


### 步骤 3: 取 reward、value、done

```python
rewards = tensordict.get(("next", *self.reward_name))
values = tensordict["state_value"]
dones = self._get_dones(tensordict)
```


### 步骤 4: 算 GAE advantages 和 returns

```python
tensordict["advantages"], tensordict["returns"] = compute_gae(...)
```


### 步骤 5: advantage 标准化

```python
tensordict["advantages"] = (advantages - mean) / (std + 1e-8)
```

为什么做这个:

- advantage 数值尺度变化太大时，actor 更新会不稳定
- 标准化后更容易训练


### 步骤 6: 更新 value normalizer

如果启用了 value normalization:

```python
self.value_normalizer.update(tensordict["returns"])
tensordict["returns"] = self.value_normalizer.normalize(tensordict["returns"])
```

作用是:

- 让 critic 回归目标尺度更稳定


### 步骤 7: 多个 PPO epoch 训练 actor/critic

```python
for ppo_epoch in range(self.ppo_epoch):
    dataset = make_dataset_naive(...)
    for minibatch in dataset:
        update_actor(minibatch)
        update_critic(minibatch)
```


### 步骤 8: 汇总日志并返回

最后返回诸如:

- `policy_loss`
- `value_loss`
- `entropy`
- `ESS`
- `advantages_mean`
- `advantages_std`

训练脚本再把这些写到 TensorBoard。


## 17. 这个仓库为什么叫 MAPPO，不直接叫 PPO

因为这里是多智能体场景。

多智能体版本和单智能体版本最大的差别在于:

### 17.1 多了 agent 维度

单智能体常见 shape 是:

```text
[batch, ...]
```

这里经常会变成:

```text
[num_envs, time, num_agents, ...]
```


### 17.2 actor 可以共享

代码:

```python
if self.cfg.share_actor:
    self.actor = create_actor_fn()
```

这表示所有追捕无人机共享一套 actor 参数。

直觉上就是:

- 三个无人机不是各学各的网络
- 而是共用一个决策模型


### 17.3 critic 可以看更全局的信息

配置里有:

```text
critic_input: obs  # or state
```

如果设成 `state`，critic 可以看更完整的全局状态，这是典型的 MAPPO 思路:

- actor 用局部可执行观测
- critic 用更全局信息来做训练辅助

本项目当前默认是:

```text
critic_input: obs
```

也就是 critic 默认仍然基于观测输入。


## 18. 这个仓库里 actor 和 critic 前面是什么网络

构建入口在:

- `make_ppo_actor()`
- `make_critic()`
- `make_encoder()`

编码器的逻辑在:

- `omni_drones/learning/common.py`

如果输入是普通连续向量:

- 先 `LayerNorm`
- 再过多层 MLP

如果输入是 `CompositeSpec`:

- 会按状态部分和视觉部分拆开
- 对状态部分用 attention encoder 或别的 encoder

也就是说，PPO 的核心训练逻辑和网络结构是两层东西:

- PPO 决定"怎么更新"
- encoder 决定"拿什么特征做决策"


## 19. 这个项目相对标准 PPO 的额外部分: TP_net

你会在 `mappo.py` 里看到 `TP_net`。

这是本项目为了目标未来轨迹预测加上的额外模块，不是标准 PPO 的必要部分。

训练逻辑是:

- 先单独训练 `TP_net`
- 再做 PPO actor/critic 更新

代码位置:

- `update_TP()`
- `train_op()` 里的 `if self.use_TP_net`

如果你当前的目标只是掌握 PPO，本部分可以先当成:

**项目额外加的一条辅助监督分支，不是 PPO 本体。**


## 20. 把一次训练想成一条流水线

你可以把整个训练过程记成下面这条流水线:

```text
旧策略
  -> 在环境里 rollout
  -> 得到 obs/action/logp/value/reward/done
  -> 用最后一步 value 做 bootstrap
  -> 算 GAE advantages
  -> 算 returns
  -> 标准化 advantages
  -> 切 mini-batch
  -> 多个 epoch 更新 actor
  -> 多个 epoch 更新 critic
  -> 得到新策略
  -> 再去 rollout
```


## 21. 最容易混淆的概念

### 21.1 rollout 不是完整 episode

在这个项目里，rollout 通常只是固定 `train_every` 步。


### 21.2 frames 不是 episode 数

`frames` 是所有并行环境累积起来的总 step 数。


### 21.3 actor 学的不是"正确标签"

PPO 不是监督学习，不存在一个标准答案动作。

actor 学的是:

- 好动作概率升高
- 坏动作概率降低


### 21.4 clip 不是在裁剪梯度

PPO 的 `clip` 指的是:

- 裁剪 policy ratio

不是 gradient clipping。

这个项目里两者都有:

- PPO clip: `clip_param`
- 梯度裁剪: `clip_grad_norm_()`


### 21.5 critic 不是算 reward

critic 输出的是长期价值 `V(s)`，不是当前一步奖励。


## 22. 结合你这个项目配置，怎么理解一次更新

如果你现在配置是:

- `num_envs = 2048`
- `train_every = 64`
- `ppo_epochs = 4`
- `num_minibatches = 16`

那么一次训练大概是:

1. 2048 个环境并行各走 64 步
2. 一共得到 `2048 * 64 = 131072` 个 step
3. 用这批 step 算 advantage 和 return
4. 把这 `131072` 个样本打散成 16 个 mini-batch
5. 共训练 4 个 epoch
6. 所以 actor/critic 大约各更新 `16 * 4 = 64` 次小步
7. 然后再去采下一次 rollout

这也是为什么你在 TensorBoard 里会看到:

- `rollout_fps` 表示采样速度
- `frames` 每次跳一大截


## 23. 建议你读代码的顺序

如果你准备真正掌握这套实现，建议按这个顺序读:

1. `scripts/train.py`
   - 看训练循环骨架
2. `omni_drones/utils/torchrl/collector.py`
   - 看 rollout 和 frames/fps
3. `omni_drones/learning/mappo.py` 的 `train_op`
   - 看 PPO 主过程
4. `omni_drones/learning/utils/gae.py`
   - 看 advantage/return 怎么算
5. `mappo.py` 里的 `update_actor` 和 `update_critic`
   - 看两个 loss
6. `mappo.py` 里的 `Actor`, `Critic`, `make_encoder`
   - 看网络结构


## 24. 学 PPO 时最应该真正理解的 5 句话

如果你读完整篇只能记住 5 句话，我建议记这 5 句:

1. PPO 先采样，再更新，不是一边走一步一边更新。
2. actor 靠 advantage 学，critic 靠 returns 学。
3. advantage 表示"这个动作比原本预期更好还是更差"。
4. PPO 的 clip 用来防止新策略一次改太多。
5. 一个 rollout 会被重复切 mini-batch 训练多个 epoch。


## 25. 下一步你应该怎么继续学

建议按下面顺序继续:

1. 先把本文里第 9 节到第 16 节彻底吃透
2. 自己在纸上手算一个 3 步轨迹的 `delta`, `advantage`, `return`
3. 再回去看 `update_actor()` 里的 `ratio`, `surr1`, `surr2`
4. 最后再看这个项目里的多智能体细节

如果你做到了这一步，你对 PPO 的核心训练流程就已经算真正入门了。


## 26. 本文档对应的核心代码位置

- 训练入口: `scripts/train.py`
- rollout collector: `omni_drones/utils/torchrl/collector.py`
- PPO/MAPPO 主逻辑: `omni_drones/learning/mappo.py`
- GAE: `omni_drones/learning/utils/gae.py`
- encoder 构建: `omni_drones/learning/common.py`
- 算法超参数: `cfg/algo/mappo.yaml`

