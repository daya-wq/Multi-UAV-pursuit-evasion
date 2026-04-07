# 多无人机协同追捕：完整设计方案（最终整合版）

> **适用场景**：3D 空间，M 架追捕 vs 1 个目标，**追捕方数量可变**，目标持续向守区逃跑并有回避机动，使用 MAPPO + TP_net 框架。
>
> **文档版本**：v3，整合了讨论与两份 GPT 补充文档的最终结论，其中 Φ_team 和奖励骨架取自守区防守专项设计，观测特征和网络架构取自协同讨论版本。

---

## 一、根本问题诊断

### 1.1 四个根因

| 根因 | 具体表现 | 严重程度 |
|------|---------|---------|
| **角色坍缩 (Role Collapse)** | 共享 Actor + 相同输入 → 所有 agent 输出相同动作 | ⭐⭐⭐⭐⭐ 最核心 |
| **观测信息不足** | `state_others` 只有 3D 位置，看不到队友战术态势 | ⭐⭐⭐⭐⭐ |
| **归因问题 (Credit Assignment)** | 每架收到相同团队奖励，无法区分"在补位"还是"在挤堆" | ⭐⭐⭐⭐⭐ |
| **Critic 不支持 CTDE** | `critic_input: obs`，Critic 看不到全局状态，方差大 | ⭐⭐⭐ |

### 1.2 核心判断

> 只解决**表示问题**（观测 + 网络），策略会看懂局面但**学不稳**。
> 只解决**归因问题**，有梯度信号但**表示能力不够**，仍然学成同质直追。
> 两者必须同时补上。

### 1.3 训练数据的直接证据

| Run | success | encircle_reward | return |
|-----|---------|----------------|--------|
| 20260327 | 0.163 | 0.001 | -79.3 |
| 20260326 | 0.162 | 2.656 | -15.3 |

`encircle_reward` 量级相差 2000 倍，success 几乎完全相同。说明手工调奖励系数无法触达协同策略本体，问题根源在观测和归因。

---

## 二、观测空间设计

### 2.1 三组观测的分工

```
每架无人机的完整观测：

  state_self   (固定维度)
  │  "我自己的状态"——自身动力学 + 目标感知
  │  与追捕方数量 N 无关，不含任何队友信息
  │
  state_others (变长序列，每个队友一行，走 Attention)
  │  "每个队友的战术态势"——含战术投影，N 可变
  │
  cooperation  (固定维度，per-agent)
     "我在这个团队中的战术位置"
     每架 agent 各自的值不同，驱动角色分化的关键
```

### 2.2 state_others 扩展（每个队友 8D）

> **核心**：必须包含队友的战术态势特征，否则 Attention 推断不出"某位置是否已被覆盖"。

```python
state_others = torch.cat([
    drone_rpos,                          # 队友相对我的位置          (3D)  已有
    drone_rvel,                          # 队友相对我的速度          (3D)  新增
    j_ahead_proj_goal.unsqueeze(-1),     # 队友在守区威胁轴的前后投影 (1D)  新增★
    j_dist_to_target.unsqueeze(-1),      # 队友离目标的距离          (1D)  新增★
], dim=-1)   # [E, N, N-1, 8]
```

**为什么 `j_ahead_proj_goal` 是关键**：Agent A 看到队友 B 的 `j_ahead_proj_goal > 0`，说明 B 已经在守区威胁轴前方堵截了 → Attention 驱动 A 去侧翼补缺口，而不是跟着 B 挤堆。

### 2.3 cooperation 特征（10D，per-agent）

| 特征 | 维度 | 物理含义 | 说明 |
|------|------|---------|------|
| `coverage_quality` | 1 | 全队包围均匀度 | 0→1，全队同一值 |
| `goal_threat_dir` | 3 | 战略方向：目标→守区 | 稳定，不随局部机动跳变 |
| `tp_escape_dir` | 3 | 战术方向：TP_net 预测下一步 | 实时，反映当前逃跑意图 |
| `my_ahead_proj_goal` | 1 | 我在守区威胁轴的前后位置 | 正值=在目标和守区之间 |
| `my_ahead_proj_tp` | 1 | 我在战术追截轴上的前后 | 正值=在目标当前运动前方 |
| `my_lateral_dist_goal` | 1 | 我偏离守区威胁轴多远 | 越小越在堵截位 |

**为什么用两个方向而不做线性混合**：

目标做局部规避时，`tp_escape_dir` 可能临时与 `goal_threat_dir` 相反。固定 `blend(β=0.25)` 无法适应，而分开编码为独立特征后，Transformer 的 Attention 可以根据两者的一致程度自适应加权——当两者高度一致时（目标在直奔守区），两个特征协同增强；当目标在做局部规避时，`goal_threat_dir` 提供稳定信号不受干扰。

**计算方法（3D）**：

```python
# 战略方向（稳定）
goal_threat_dir = F.normalize(
    goal_center.view(1, 3) - target_pos.squeeze(1), dim=-1   # [E, 3]
)

# 战术方向（TP_net，实时）
escape_vec    = target_pos_predicted[:, 0, :] - target_pos.squeeze(1)
tp_escape_dir = F.normalize(escape_vec, dim=-1)              # [E, 3]

# 计算每架 agent 的投影
agent_from_target  = drone_pos - target_pos                  # [E, N, 3]
g_exp = goal_threat_dir.unsqueeze(1).expand(-1, N, -1)
t_exp = tp_escape_dir.unsqueeze(1).expand(-1, N, -1)

my_ahead_proj_goal    = (agent_from_target * g_exp).sum(-1)  # [E, N]
my_ahead_proj_tp      = (agent_from_target * t_exp).sum(-1)  # [E, N]
lateral_vec           = agent_from_target - my_ahead_proj_goal.unsqueeze(-1) * g_exp
my_lateral_dist_goal  = lateral_vec.norm(dim=-1)             # [E, N]

# coverage_quality（方法A，适用任意 N）
norm_dir         = F.normalize(agent_from_target, dim=-1)    # [E, N, 3]
mean_dir         = norm_dir.mean(dim=1)                      # [E, 3]
coverage_quality = 1.0 - mean_dir.norm(dim=-1)               # [E]
coverage_quality = coverage_quality.unsqueeze(1).expand(-1, N)

# 拼接 cooperation
cooperation = torch.stack([
    coverage_quality,
    *goal_threat_dir.unsqueeze(1).expand(-1, N, -1).unbind(-1),
    *tp_escape_dir.unsqueeze(1).expand(-1, N, -1).unbind(-1),
    my_ahead_proj_goal,
    my_ahead_proj_tp,
    my_lateral_dist_goal,
], dim=-1).unsqueeze(2)   # [E, N, 1, 10]
```

### 2.4 变量 N 兼容性

| 观测组 | 维度 | 支持可变 N？ | 原因 |
|--------|------|-----------|------|
| `state_self` | 固定 | ✅ | 不依赖 N |
| `cooperation` | 固定 10D（per-agent） | ✅ | 基于全队统计量 |
| `state_others` | `(N-1) × 8` | ✅ | Attention 天然处理变长序列 |
| ~~Agent One-Hot ID~~ | ~~固定 N 维~~ | ❌ | **不使用** |

---

## 三、网络架构

### 3.1 以自我为中心的实体注意力（Self-centric Entity Attention）

```
token_0: [自我]     = Linear(state_self + cooperation)  → d_model
token_1: [队友 1]   = Linear(state_others[0])           → d_model
token_2: [队友 2]   = Linear(state_others[1])           → d_model
  ···
token_k: [队友 N-1] = Linear(state_others[N-2])         → d_model

→ Multi-Head Self-Attention
→ 取 token_0 输出
→ Actor MLP → 动作
```

每个 token 都能与所有其他 token 交互，避免分组处理时丢失交叉信息（例如"我的 ahead_proj 与队友的差值"才是关键信号）。

### 3.2 实现代码

```python
class CoopEntityAttentionEncoder(nn.Module):
    def __init__(self, self_dim, other_dim, coop_dim,
                 d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.self_embed  = nn.Linear(self_dim + coop_dim, d_model)
        self.other_embed = nn.Linear(other_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=256, dropout=0.0, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_dim  = d_model

    def forward(self, state_self, cooperation, state_others):
        self_token   = self.self_embed(
            torch.cat([state_self, cooperation], dim=-1)    # [B, 1, self+coop]
        )                                                    # [B, 1, d_model]
        other_tokens = self.other_embed(state_others)        # [B, N-1, d_model]
        tokens       = torch.cat([self_token, other_tokens], dim=1)
        out          = self.transformer(tokens)
        return out[:, 0, :]   # 只取自我 token 的输出
```

---

## 四、协同势函数 Φ_team（守区防守专版）

> 这是归因层的核心。纯角度覆盖质量不够，需要感知守区威胁。

### 4.1 基本记号

```
A         : 当前 active pursuer 集合
n = |A|
u_goal    = normalize(goal_center - target_pos)   # 战略威胁方向
u_tp      = tp_escape_dir                          # 战术方向

v_i = drone_pos_i - target_pos   # 每架无人机从目标指出的向量
d_i = ||v_i||                    # 到目标的距离
d_goal_t  = distance(target_pos, goal_boundary)   # target 到守区边界的距离
```

### 4.2 守区紧急度门控

```python
g_urg = torch.sigmoid((d_crit - d_goal_t) / tau_urg)
# d_crit = 0.8m（临近守区时开始门控）
# tau_urg = 0.15m
# target 越近守区 → g_urg → 1 → 团队更应该"堵门"而非展开
```

### 4.3 单机堵截分数 block_i

```python
# a_i: 在守区威胁轴 u_goal 上的前向投影
a_i = (v_i * u_goal).sum(-1)                    # [E, N]
# l_i: 侧向偏差
l_i = (v_i - a_i.unsqueeze(-1) * u_goal).norm(-1)  # [E, N]
# goal_dist_from_target: 目标到守区中心的距离
goal_to_target_dist = (goal_center - target_pos).norm(-1)  # [E]

block_i = (
    torch.sigmoid((a_i - 0.15) / 0.10)              # 在目标前方
    * torch.sigmoid(
        (goal_to_target_dist.unsqueeze(1) - a_i) / 0.15
      )                                              # 在目标和守区之间
    * torch.exp(-l_i / 0.35)                         # 贴近威胁轴
)   # [E, N]，范围约 (0, 1)
```

### 4.4 三项团队势函数

```python
# A. 堵截项（主项）
top2_block = block_i.topk(min(n, 2), dim=1).values   # [E, 2]
phi_block  = 0.7 * block_i.max(dim=1).values \
           + 0.3 * top2_block.mean(dim=1)

# B. 压力项（最近2架的距离）
top2_dist  = d_i.topk(min(n, 2), dim=1, largest=False).values  # [E, 2]
phi_pressure = torch.exp(-top2_dist.mean(dim=1) / 0.8)

# C. 展开项（随 N 自适应语义）
if n == 1:
    phi_spread = torch.zeros(E, device=device)
elif n == 2:
    cos12      = (norm_dir[:, 0, :] * norm_dir[:, 1, :]).sum(-1)
    phi_spread = 0.5 * (1 - cos12)
else:  # n >= 3
    phi_spread = 1.0 - norm_dir.mean(dim=1).norm(dim=-1)  # 方法A

# D. 加权组合（守区紧急时 block 权重提高，spread 权重降低）
Phi_team = (
    0.50 * (0.4 + 0.6 * g_urg) * phi_block
    + 0.30 * phi_pressure
    + 0.20 * (1.0 - g_urg) * phi_spread
)   # [E]
```

---

## 五、差分边际贡献 D_i

### 5.1 数学定义

$$D_i = \Phi_{team}(A) - \Phi_{team}(A \setminus \{i\})$$

- $D_i > 0$：移除 agent $i$ 后防守质量下降 → 它在补缺口，有贡献
- $D_i < 0$：移除 $i$ 后防守质量提升 → 它在挤堆，冗余

### 5.2 代码实现

```python
def compute_Di(drone_pos, target_pos, block_i, d_i, norm_dir, g_urg,
               goal_to_target_dist, n):
    """
    n = 1 时直接返回 0，避免引入噪声。
    """
    E = drone_pos.shape[0]
    D = torch.zeros(E, n, device=drone_pos.device)
    if n == 1:
        return D

    Phi_full = compute_Phi_team(block_i, d_i, norm_dir, g_urg, goal_to_target_dist, n)

    for i in range(n):
        idx = [j for j in range(n) if j != i]
        Phi_without_i = compute_Phi_team(
            block_i[:, idx], d_i[:, idx], norm_dir[:, idx, :],
            g_urg, goal_to_target_dist, n - 1
        )
        D[:, i] = Phi_full - Phi_without_i

    # 裁剪防止极端值
    D_clip = torch.tanh(D / 0.15)
    return D_clip   # [E, N]
```

---

## 六、奖励函数

### 6.1 设计原则

1. 主目标少而强，围绕任务成败
2. 协同项做辅助信号，不压过主目标，需退火
3. 安全项用拉格朗日约束，不是无限堆负奖励
4. 奖励聚合用 mean/max，不用 sum，保证 N 可变时量级稳定

### 6.2 完整奖励表达式

```python
r_i = (
    r_terminal             # 稀疏主信号
    + r_goal_progress      # leader dense 信号
    + r_capture_progress   # 追击 dense 信号
    + w_coop(t) * r_coop  # 协同辅助，退火
    - lambda_col  * c_collision_i   # 自适应碰撞约束
    - lambda_land * c_landed_env    # 自适应坠地约束
    - 0.05 * F.relu(speed_i / v_max - 1)  # 超速惩罚
    - 0.002                # 固定时间惩罚（逼迫积极行动）
)
```

### 6.3 各项详解

**终局项（最强信号）**

```python
r_terminal = (
    +10.0 * capture_before_goal.float()   # 在目标进守区前抓住 ✅
    - 10.0 * goal_reached.float()         # 目标进入守区 ❌
    -  2.0 * timeout.float()              # 超时未捕获 ⚠️
)
```

**守区进度项（每步稠密，核心 dense 信号）**

```python
# target 被逼离守区 → 正奖励；target 逼近守区 → 负奖励
d_goal_next = distance(target_pos_next, goal_boundary)
d_goal_curr = distance(target_pos_curr, goal_boundary)
r_goal_progress = 0.40 * torch.tanh((d_goal_next - d_goal_curr) / 0.05)
```

**抓捕进度项（每步稠密）**

```python
# 只看最近 1-2 架，N 增大时量级稳定
k = min(n, 2)
d_team_next = d_i_next.topk(k, largest=False).values.mean(-1)
d_team_curr = d_i_curr.topk(k, largest=False).values.mean(-1)
r_capture_progress = 0.20 * torch.tanh((d_team_curr - d_team_next) / 0.05)
```

**协同辅助项（退火）**

```python
r_coop = 0.35 * D_clip + 0.15 * Phi_team.unsqueeze(1).expand(-1, N)
# D_clip：个体边际贡献（归因信号）
# Phi_team：团队结构参考（稳定信号）
```

**协同退火调度**

```python
# 前50%训练进度：w_coop 从 0.60 退火到 0.15
progress = current_frame / total_frames   # [0, 1]
w_coop = max(0.15, 0.60 * (1.0 - 2.0 * min(progress, 0.5)))
```

### 6.4 自适应拉格朗日约束（先用固定值，稳定后切换）

```python
# 固定版（快速落地）
lambda_col  = 2.0   # 碰撞惩罚系数
lambda_land = 6.0   # 坠地惩罚系数

# 自适应版（稳定后升级）
lambda_col = torch.clamp(
    lambda_col + 0.01 * (collision_rate - 0.03), min=0
)
lambda_land = torch.clamp(
    lambda_land + 0.01 * (landed_rate - 0.005), min=0
)
```

### 6.5 各组奖励量级参考

| 项目 | 量级 | 类型 |
|------|------|------|
| r_terminal | ±10 | 稀疏 |
| r_goal_progress | ±0.4 per step | 稠密 |
| r_capture_progress | ±0.2 per step | 稠密 |
| r_coop | 0~0.5 × w_coop | 稠密，退火 |
| 安全约束 | -λ×cost，动态 | 约束 |
| 时间惩罚 | -0.002 per step | 固定 |

---

## 七、关键训练技巧

### 7.1 训练时随机打乱 agent 索引（必须）

防止网络只背编号而不学几何协同：

```python
if self.training:
    perm = torch.randperm(self.num_agents, device=self.device)
    obs["state_self"]   = obs["state_self"][:, perm]
    obs["cooperation"]  = obs["cooperation"][:, perm]
    # state_others 的相对关系需要对应重新排列
```

### 7.2 课程学习（门控条件，不是固定 frame 数）

```
阶段 1：
  参数：初始距离 ∈ [0.3, 0.8]m，目标速度 0.5m/s
  初始位置：在该距离范围内球面均匀随机采样（角度完全随机）
  进入下一阶段条件：rolling success > 60% 且 first_capture_step < 400 步

阶段 2：
  参数：初始距离 ∈ [0.5, 1.5]m，目标速度 0.8m/s
  进入下一阶段条件：rolling success > 50%

阶段 3（正常难度）：
  参数：初始距离 ∈ [1.0, 2.5]m，目标速度 1.0m/s
```

**关键**：课程控制的是距离范围，**初始角度始终随机**（均匀球面采样），避免对称性导致角色坍缩。

### 7.3 对称性破缺

```python
# 初始位置加随机扰动，防止所有 agent 陷入对称不动点
init_pos += torch.randn_like(init_pos) * 0.3
```

### 7.4 探索调度（保持 entropy，让角色涌现）

```yaml
entropy_coef: 0.05   # 初期高，保持探索
# CosineAnnealing 衰减到 0.005
# T_0: 1000 iteration，探索期足够长
```

### 7.5 CTDE Critic

```yaml
critic_input: state   # 训练时 Critic 看全局状态
```

---

## 八、验证方案

### 8.1 五类验证指标，**按 N 分开统计**

```python
for n_pursuers in [1, 2, 3, 4]:  # 你训练过的所有数量
    log(f"success_rate_N{n_pursuers}",   ...)
    log(f"goal_reached_rate_N{n_pursuers}", ...)
    log(f"capture_step_N{n_pursuers}",   ...)
```

**为什么按 N 分**：不分开就无法判断"策略是真的学会了可变数量泛化，还是只在特定 N 下好用"。

| 类别 | 关键指标 |
|------|---------|
| **结果类** | `success(N)`, `goal_reached(N)`, `capture_step(N)` |
| **防守类** | `urgent_block_rate`（g_urg>0.5 时 phi_block>0.6 的步占比），`n_agents_ahead_mean` |
| **协同类** | `Phi_team_mean`, `Di_mean`, `Di_min`, `frac(Di < 0)` |
| **稳定性类** | `collision_rate(N)`, `landed_rate(N)` |
| **泛化类** | 打乱 agent 索引后的 success 差值，`drop-one-agent` success 下降幅度 |

### 8.2 最重要的验收标准

> **移除任意一架 pursuer，成功率是否显著下降？**
>
> - 下降不大 → 当前是**冗余追击**，不是真协同
> - 大幅下降，且剩余无人机也失去堵截能力 → **真协同初步成立**

### 8.3 Di 相关诊断

```
Di_min 长期为负 → 有 agent 在挤堆，协同未涌现
frac(Di < 0) 下降 → 分工在改善
Di_mean 随 success 同步上升 → 协同与结果正相关，设计有效
```

---

## 九、不建议优先做的事情

- ❌ 继续手工调 `encircle_coef`、`intercept_coef` 等十几个系数
- ❌ 看视频就补一个角色专属奖励（blocker 奖励、chaser 奖励）
- ❌ 使用固定 One-Hot Agent ID（可变 N 下根本失效）
- ❌ 观测和归因没整理好前就重写更大的 Encoder
- ❌ 先把在线 LLM 角色分配塞进训练回路（基础不稳时叠加不稳定）

---

## 十、实施路线图

### P0（2-5天）：基础实现，打破 Role Collapse

- [ ] 扩展 `state_others`：3D → 8D（加 `j_ahead_proj_goal`, `j_dist_to_target`）
- [ ] 新增 `cooperation` 观测组（10D）：双方向 + 各投影特征
- [ ] 在 `_compute_state_and_obs` 中实现 `g_urg`, `block_i`, `Phi_team`, `D_i`
- [ ] 实现 `CoopEntityAttentionEncoder`，替换 `PartialAttentionEncoder`
- [ ] `mappo.yaml`：`critic_input: state`
- [ ] 添加训练时随机打乱 agent 索引
- [ ] 奖励替换为：`r_terminal + r_goal_progress + r_capture_progress + w_coop*r_coop`
- [ ] 安全约束先用固定版（`lambda_col=2.0, lambda_land=6.0`）
- [ ] 添加完整验证指标（按 N 分组）

**期望结果**：success 突破 16%，`Di_min` 从负值逐渐变正，`urgent_block_rate` 稳定上升。

### P1（中期，1-2周）：提升稳定性，减少人工干预

- [ ] 安全约束升级为自适应拉格朗日
- [ ] 协同 shaping 退火调度精细化
- [ ] 自动课程（ALP-GMM 或 ADR）
- [ ] PBT 自动调度少量超参
- [ ] LLM 离线分析：读训练日志，建议奖励权重

### P2（研究性）

- [ ] 在线 LLM 低频角色分配（注入 cooperation 中）
- [ ] 多目标场景扩展
- [ ] HAPPO / HARL（如果参数共享仍然受限）

---

## 十一、完整设计总览

```
观测空间
  state_self (固定，35D)
    自身动力学(13D) + 目标感知(3D) + TP预测(15D) + 时间(4D)

  state_others (变长，每队友 8D)
    rel_pos(3) + rel_vel(3) + j_ahead_proj_goal(1) + j_dist_to_target(1)

  cooperation (固定，10D，per-agent)
    coverage_quality(1) + goal_threat_dir(3) + tp_escape_dir(3)
    + my_ahead_proj_goal(1) + my_ahead_proj_tp(1) + my_lateral_dist_goal(1)

网络
  [self+coop token, other_token×(N-1)] → TransformerEncoder(4 heads,2 layers)
  → token_0 输出 → Actor MLP → 动作

协同势函数 Φ_team（守区防守专版）
  = 0.50×(0.4+0.6×g_urg)×phi_block
  + 0.30×phi_pressure
  + 0.20×(1-g_urg)×phi_spread（n自适应）

D_i = Φ(全队) - Φ(移除i后)，n=1时为0，裁剪为 tanh(D/0.15)

奖励骨架
  r_terminal(±10) + r_goal_progress(0.4×tanh) + r_capture_progress(0.2×tanh)
  + w_coop(退火)×(0.35×Di + 0.15×Phi) + 自适应安全约束 - 0.002

训练设置
  CTDE critic, 随机打乱 agent 索引, 门控课程学习, entropy 退火
```

---

*文档版本：v3（2026-04-02）| 整合协同追捕设计讨论 + 守区防守专项补充*
