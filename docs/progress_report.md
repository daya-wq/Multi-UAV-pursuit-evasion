# 项目进度报告 (Progress Report)

---

## 当前状态

| 项目 | 状态 |
|---|---|
| 第一阶段 Bug 修复 (5项) | ✅ 已完成 |
| TensorBoard 集成 | ✅ 已完成 |
| 离屏渲染视频导出 | ✅ 已完成 (`use_local_usd: false`) |
| GPU 选择配置 | ✅ 已解决 |
| Isaac Sim 崩溃自动重启 | ✅ `scripts/auto_train.sh` |
| 正式训练 (v1 原始奖励) | ✅ 已完成 ~445M 帧，成功率 0% |
| 奖励函数调优 (v2) | ⏳ 进行中 - 从 78M 帧权重继续训练 |
| 0% 成功率根因分析 | ✅ 已完成 - 速度劣势 + 碰撞惩罚过高 |

---

## 2026-03-24 (晚): 0% 成功率根因分析 & 奖励调优

### TensorBoard 曲线诊断（~445M 帧）

经过约 4.45 亿步训练，AI 已学会：
- ✅ **追踪猎物**：`distance_reward` 从 -1.4 上升到 -1.1
- ✅ **避免碰撞**：`collision_wall`, `collision_drone` 均降至接近 0
- ✅ **平滑控制**：`action_error_order1` 持续稳定下降
- ❌ **未学会围捕**：`catch_reward` 和 `success` 持续为 0

### 0% 成功率根因（非代码Bug）

| 原因 | 详情 |
|---|---|
| ① 速度绝对劣势 | 猎物 1.3 m/s vs 无人机 1.0 m/s |
| ② 碰撞惩罚过高 | `collision_coef=100` >> `catch_reward_coef=60` |
| ③ 固定初始位置 | `use_eval=1` 每局位置完全相同，零探索多样性 |
| ④ APF 完美逃跑 | 猎物用人工势场法逃跑，无障碍物环境中只有完美包抄才能抓到 |

### 奖励参数调优（v2）

修改文件：`cfg/task/HideAndSeek.yaml`

| 参数 | v1 (旧值) | v2 (新值) | 调整理由 |
|---|---|---|---|
| `catch_radius` | 0.3m | **0.4m** | 降低抓捕空间难度 |
| `catch_reward_coef` | 60.0 | **200.0** | 重赏之下必有勇夫 |
| `collision_coef` | 100.0 | **20.0** | 允许适度莽撞，鼓励近距离协作 |

### 新增 TensorBoard 指标

修改文件：`omni_drones/envs/hide_and_seek/hideandseek.py`
- 新增 `train/stats.pursuer_collisions_count`：追捕方每局实际互撞总次数

### 评估视频

使用 `checkpoint_445775872.pt`（v1 奖励，445M 帧）生成的策略视频：
- 路径：`eval_videos/HideAndSeek_20260324_211942.mp4`
- 观察：无人机能追踪猎物，但在接近时会因害怕互撞惩罚而保持距离、不敢合围

### 当前训练状态

从 `checkpoint_78774272.pt`（78M 帧）开始，使用 v2 奖励参数继续训练：
```bash
nohup bash scripts/auto_train.sh > auto_train_monitor.log 2>&1 &
```
- 并行环境数：2048
- GPU：RTX 4090 (cuda:1)，显存 17.8GB / 24.5GB
- 吞吐量：rollout_fps ≈ 125K-148K
- Isaac Sim 约每 2-2.5 小时崩溃一次，脚本自动重启续训

---

## 2026-03-24: 环境配置 & 测试

### 修改的文件

| 文件 | 修改内容 |
|---|---|
| `scripts/train.py` | TensorBoard 日志 + 本地 checkpoint 保存 |
| `scripts/eval.py` | imageio 本地视频导出到 `eval_videos/` |
| `cfg/train.yaml` | `use_local_usd: false`（使用内置场景） |
| `cfg/base/sim_base.yaml` | 默认 `cuda:0` |
| `omni_drones/__init__.py` | 原始状态（GPU 配置回滚） |
| `omni_drones/envs/isaac_env.py` | `close()` 修复 |
| `omni_drones/envs/hide_and_seek/hideandseek.py` | 抓捕终止逻辑 |
| `omni_drones/learning/mappo.py` | LR scheduler 修复 |

### 关键发现

- **白屏视频解决**：`use_local_usd: false` 使用 Isaac Sim 内置默认场景（自带光源和地板）
- **多 GPU 与指定 GPU 运行**：通过 `sim.device=cuda:1` 可以成功指定特定 GPU 运行
- **并行环境数**：2048 个（正式训练配置）

---

## 2026-03-23: 第一阶段 Bug 修复

| # | 问题 | 修改文件 | 状态 |
|---|---|---|---|
| 1 | 随机种子硬编码 | `train.py` 等 | ✅ |
| 2 | 设备硬编码 `cuda:0` | 多文件 | ✅ |
| 3 | `close()` 短路 | `isaac_env.py` | ✅ |
| 4 | 抓到目标不结束 episode | `hideandseek.py` | ✅ |
| 5 | LR scheduler 不生效 | `mappo.py` | ✅ |

---

## 🚀 如何运行 (How to Run)

### 1. 激活环境
```bash
conda activate sim
cd /data/uavlab/multi-uav-pursuit2
```

### 2. 启动训练 (Training)

**推荐方式：使用自动重启脚本**（处理 Isaac Sim 周期性崩溃）：
```bash
nohup bash scripts/auto_train.sh > auto_train_monitor.log 2>&1 &
```

手动单次训练：
```bash
python3 scripts/train.py headless=true wandb.mode=disabled \
  task=HideAndSeek task.use_eval=1 task.use_random_cylinder=0 \
  task.scenario_flag=empty task.env.num_envs=2048 \
  total_frames=2000000000 eval_interval=4000
```
* **参数说明**：
  * `headless=true`: 无头模式
  * `task.scenario_flag=empty/wall/passage`: 场景选择
  * `task.env.num_envs=2048`: 并行环境数
  * `model_dir=<checkpoint路径>`: 加载预训练权重继续训练
* **输出**：
  * 模型权重: `checkpoints/HideAndSeek_<时间戳>/checkpoint_*.pt`
  * 日志: `runs/HideAndSeek_<时间戳>/`

### 3. 查看训练曲线 (TensorBoard)
```bash
tensorboard --logdir=runs/ --bind_all
```
浏览器打开：`http://<服务器IP>:6006`

### 4. 评估并导出视频 (Evaluation)
```bash
python3 scripts/eval.py headless=true wandb.mode=disabled \
  model_dir=checkpoints/HideAndSeek_<时间戳>/checkpoint_*.pt \
  task=HideAndSeek task.scenario_flag=empty task.use_eval=1 task.env.num_envs=1
```
视频自动保存在 `eval_videos/HideAndSeek_<时间戳>.mp4`

---

## 待办事项

- [x] 启动正式长时间训练
- [x] 0% 成功率根因分析
- [x] 奖励参数调优 (v2)
- [ ] 观察 v2 奖励训练是否突破 0% 成功率
- [ ] 若成功，逐步恢复猎物速度至 1.3 m/s
- [ ] 尝试 `wall` / `passage` 等有障碍物场景
- [ ] 解决 `use_eval=0`（随机初始化）Isaac Sim 段错误
- [ ] 第二阶段代码优化
