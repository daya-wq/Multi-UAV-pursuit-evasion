# 代码修改报告 (Code Changes Report)

持续维护文档，记录每次代码修改的详细内容。

---

## 2026-03-23: 第一阶段训练 Bug 修复包

**修改目标**: 修复 5 个影响第一阶段训练正确性和可复现性的问题。不涉及第二阶段部署训练。

### 1. 统一随机种子（P0）

**问题**: `train.py`、`train_generator.py`、`train_deploy.py` 硬编码 `seed = 42`，与 `cfg.seed`（默认 `0`）冲突，导致复现性不可靠。

**修改**:

| 文件 | 修改内容 |
|---|---|
| `scripts/train.py` | `seed = 42; set_seed(seed)` → `set_seed(cfg.seed)` |
| `scripts/train_generator.py` | 同上 |
| `scripts/train_deploy.py` | 同上 |

**影响**: 现在所有随机源统一由 `cfg.seed` 控制（默认 `0`，可通过命令行 `seed=42` 覆盖）。

### 2. 去掉设备硬编码（P0）

**问题**: 5 处写死 `device="cuda"` 或 `device="cuda:0"`，阻碍多 GPU 和非 0 号卡使用。

**修改**:

| 文件 | 行 | 修改前 | 修改后 |
|---|---|---|---|
| `scripts/train.py` | 策略创建 | `device="cuda"` | `device=cfg.sim.device` |
| `scripts/train_generator.py` | 策略创建 | `device="cuda"` | `device=cfg.sim.device` |
| `scripts/train_deploy.py` | 策略创建 | `device="cuda"` | `device=cfg.sim.device` |
| `scripts/eval.py` | 策略创建 | `device="cuda"` | `device=cfg.sim.device` |
| `omni_drones/envs/isaac_env.py` | SimulationContext | `device="cuda:0"` | `device=str(self.cfg.sim.device)` |

**影响**: 设备完全跟随 `cfg/base/sim_base.yaml` 中的 `sim.device`（默认 `cuda:0`）。多 GPU 运行时修改该配置即可。

### 3. 修复 `IsaacEnv.close()` 短路（P0）

**问题**: `close()` 方法第一行 `return`，后续的仿真停止、回调清理、stage 清理全部跳过。

**修改**: `omni_drones/envs/isaac_env.py`
- 删除 `return # TODO: fix this`
- 在清理逻辑外层包裹 `try/except/finally`，确保即使 Isaac Sim 内部抛异常也能标记 `_is_closed = True`

**影响**: 环境正常关闭时释放 GPU 资源，避免批量实验资源泄漏。

### 4. 抓到目标后 episode 立即结束（P0）

**问题**: `_compute_reward_and_done()` 中 `done` 仅判断超时（`progress_buf >= max_episode_length`），capture 不触发 episode 结束。

**修改**: `omni_drones/envs/hide_and_seek/hideandseek.py`

```python
# 修改前
done = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

# 修改后
capture_done = torch.any(masked_capture, dim=-1).unsqueeze(-1)
done = (
    (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
    | capture_done
)
```

**影响**: Episode 在任一未被遮挡的无人机抓到目标后立即结束。⚠️ 这会改变训练数据分布，旧 checkpoint 继续训练需注意。

### 5. 学习率衰减代码路径补全（P0）

**问题**: `mappo.py` 中 `make_critic()` 创建了 `critic_opt_scheduler` 但 `train_op()` 从未调用 `.step()`；Actor 侧完全没有 scheduler 逻辑。

**修改**: `omni_drones/learning/mappo.py`
1. `make_actor()` 末尾添加 actor LR scheduler 创建逻辑（与 critic 对称）
2. `train_op()` 末尾（`self.n_updates += 1` 之后）添加 `critic_opt_scheduler.step()` 和 `actor_opt_scheduler.step()`

**影响**: 当前 `mappo.yaml` 中 `lr_scheduler` 为空，默认不衰减（行为不变）。需要 LR 衰减时只需在 yaml 中配置即可。

---

## 待修复项（后续计划）

以下问题暂未处理，等第一阶段训练成功后再逐步推进：

- 第二阶段部署训练默认配置对齐（`HideAndSeek_deploy` 默认 `use_deployment: 0`）
- `eval.py` 缺少 checkpoint 强制检查
- `scripts/` 训练入口代码去重
- 三个追逃环境文件的公共基类重构
- 文档对齐和默认配置清理

---

## 2026-03-26: 重启 Curriculum Learning 与 v_prey 配置修正

**问题**: `cfg/task/HideAndSeek.yaml` 默认配置 `v_prey: 1.3`，导致在 `_compute_reward_and_done()` (第1113行) `min(1.3, self.v_prey)` 的自动难度递增代码被完全旁路 (bypassed)。初始速度差 (1.3 vs 1.0) 过大使得策略陷入“追而不抓”（尾随以防互撞）极其顽固的局部最优，导致抓捕成功率降为绝对 0。

**修改**: `cfg/task/HideAndSeek.yaml`
- `v_prey: 1.3` → `v_prey: 1.0`

**影响**: 重启了被卡住的自动难度递增课程式学习（Curriculum Learning）。训练初期将享有平等的追击速度，利于网络尽早建立抓捕正反馈，打破目前 entropy=-3.2 的过拟合保守尾随行为。改动极小但物理意义重大。

---

## 2026-03-26 (阶段二): 引入包围几何奖励 (Encircle Reward)

**问题**: 在前次将 `v_prey` 降至 1.0 后，短时训练的最新 run（163903）显示胜率依然为绝对 0，且熵再次迅速滑落至 -2.9。原因是逃跑者使用的 APF（人工势场）算法完美排斥靠近的无人机，虽然最大速度对等，但“尾随策略”在物理上永远无法抓捕到逃跑者。无人机未能在漫无目的的试错中撞大运实现包夹，于是再次收敛进“避免互撞的保守追逐”这一局部最优解。

**修改**: `omni_drones/envs/hide_and_seek/hideandseek.py` & `HideAndSeek.yaml`
- 此轮修改主打“重赏围堵矩阵”：新增 `encircle_reward`，其计算方式为 `encircle_coef * torch.exp(-com_dist)`。
- `com_dist` 为所有追捕者质心 (Center of Mass) 到逃跑者的距离。
- `HideAndSeek.yaml` 中增加超参数 `encircle_coef: 5.0`。

**影响**: 建立了一条高密度的几何奖励梯度。当三架无人机单纯尾随逃跑者时，其质心落在目标后方，奖励低；当三架无人机彻底从多角度“包围”猎物时，质心完美重合于目标点 (`com_dist` 趋近 0)，直接获取最高 `5.0` 的每步密集奖励（同时还受避障惩罚避免相撞）。这将利用纯数学约束，强制网络学到“多向合围 (Pincer Movement)” 的战术，彻底终结单纯跟踪行为。

---

## 2026-03-27: 将质心包围改为预测增强的包围几何奖励

**问题**: 最新一轮训练与评测视频表明，`encircle_reward = exp(-com_dist)` 并没有真正约束“多机从多侧缩圈”，反而允许三机继续抱团追尾；同时 TP 预测只被固定给 0 号机使用，这和共享策略的对称性不一致，也没有把多机优势转成稳定梯度。

**修改**: `omni_drones/envs/hide_and_seek/hideandseek.py` & `cfg/task/HideAndSeek.yaml`
- 新增 `distance_predicted_reward`，让所有 pursuer 都对 TP 末端预测点获得 progress reward。
- 将 `encircle_reward` 重写为“预测参考点上的角度覆盖 + 包围面积收缩”：
  - 用当前目标点和 TP 预测点做 blend，得到参考包围中心。
  - 用角度 gap 判断团队是否真正形成了多侧包围。
  - 用 pursuer 包围多边形面积的指数衰减项鼓励继续缩圈，更接近论文里 Voronoi 区域收缩的思想。
- 将 `intercept_reward` 改成团队级前向截断奖励，鼓励至少一架机体占据目标预测逃逸方向前方的通道。
- 将 `smoothness_reward` 改成零基线惩罚，平稳动作不再白拿正分，只在动作过猛时扣分。

**影响**: 新 reward 不再奖励“整体扎堆但质心凑巧靠近目标”的伪包围，而是明确要求：
1. 预测轨迹前方要有人抢位。
2. 多机围绕目标要真正展开角度覆盖。
3. 形成包围后要继续缩圈完成捕获。

由于价值地貌已再次发生显著变化，下一轮仍应采用 `restart-scratch`。
