# Multi-UAV Pursuit-Evasion — 完整运行交接指南

> **目标读者**：接手本项目的工程师或 AI 助手，需要在一台新机器（或已有机器）上首次或重新运行训练/评估。  
> **版本**：2026-04-02，适用于本仓库 master 分支。

---

## 目录

1. [项目概述](#1-项目概述)
2. [硬件要求](#2-硬件要求)
3. [软件依赖全景图](#3-软件依赖全景图)
4. [Step-by-Step 安装指南](#4-step-by-step-安装指南)
5. [关键环境变量 & 路径约定](#5-关键环境变量--路径约定)
6. [快速验证（运行前必做）](#6-快速验证运行前必做)
7. [训练](#7-训练)
8. [评估 & 生成视频](#8-评估--生成视频)
9. [目录结构说明](#9-目录结构说明)
10. [常见问题 & 已知坑](#10-常见问题--已知坑)
11. [GPU 显存清理](#11-gpu-显存清理)
12. [本机实际配置速查](#12-本机实际配置速查)

---

## 1. 项目概述

本项目基于 [OmniDrones](https://github.com/btx0424/OmniDrones) 框架，使用 **MAPPO（多智能体 PPO）** 算法在 NVIDIA Isaac Sim 内训练多架 Crazyflie 无人机协同追捕一个逃跑目标（HideAndSeek 任务）。

**核心文件**：
- `scripts/train.py` — 训练入口
- `scripts/eval.py` — 评估 + 录像入口
- `scripts/auto_train.sh` — 自动重启训练循环（推荐用于长期无人值守训练）
- `cfg/task/HideAndSeek.yaml` — 环境 & 奖励参数（最常改）
- `cfg/algo/mappo.yaml` — MAPPO 算法超参
- `omni_drones/envs/hide_and_seek/hideandseek.py` — 环境主逻辑（观测/奖励/重置）

---

## 2. 硬件要求

| 项目 | 最低 | 本机实测 |
|------|------|---------|
| GPU | NVIDIA GPU，8 GB 显存（100 envs） | **2× RTX 4090（各 24 GB）** |
| CUDA 驱动 | ≥ 11.7（Isaac Sim 2022.2.0 自带 CUDA 11.7） | Driver 580.126.09，CUDA 12.8 |
| 系统内存 | ≥ 32 GB | 推荐 64 GB + |
| 磁盘 | Isaac Sim ≈ 50 GB，模型/日志 ≈ 10 GB+ | — |
| 操作系统 | Ubuntu 20.04 / 22.04 | Ubuntu（SSH 接入） |

> **多 GPU 说明**：默认使用 GPU 0（`cfg.sim.device = "cuda:0"`）。若要切换到 GPU 1，  
> 在启动命令加 `sim.device=cuda:1` 或在 `cfg/base/sim_base.yaml` 中修改。

---

## 3. 软件依赖全景图

```
操作系统
└── NVIDIA 驱动 (≥ 520，支持 CUDA 11.7+)
    └── Isaac Sim 2022.2.0  (自带 Python 3.7 + PyTorch 1.13.0+cu117)
        └── Conda 环境 "sim"  (Python 3.7，复用 Isaac Sim 的 Python 路径)
            ├── PyTorch 1.13.0+cu117      ← Isaac Sim 内置，conda env 自动挂载
            ├── tensordict 0.1.2+5e6205c  ← git submodule (third_party/tensordict)
            ├── torchrl 0.1.1+e39e701     ← git submodule (third_party/torchrl)
            ├── dgl 1.1.3                 ← pip，用于环境生成
            ├── hydra-core 1.3.2          ← pip，配置系统
            ├── omegaconf 2.3.0           ← pip
            ├── wandb 0.18.7              ← pip，可选（默认 disabled）
            ├── setproctitle 1.3.3        ← pip
            └── imageio 2.31.2            ← pip，录像保存
```

> ⚠️ **最关键的约束**：必须使用 **Isaac Sim 2022.2.0** + **Python 3.7**。  
> 其他版本的 Isaac Sim 未经验证，极大概率出现 API 不兼容。

---

## 4. Step-by-Step 安装指南

### 4.1 安装 Isaac Sim 2022.2.0

Isaac Sim **体积约 50 GB**，本机已安装在 `/data/uavlab/isaac_sim_2022.2.0`。  
若换新机器，按以下任一方式安装：

**方式 A：官方下载（需 NVIDIA 开发者账号）**
```bash
# 下载 Omniverse Launcher，然后在 GUI 中安装 Isaac Sim 2022.2.0
# 安装后默认路径：~/.local/share/ov/pkg/isaac_sim-2022.2.0
```

**方式 B：Google Drive 预打包 zip（官方提供，约 50 GB）**
```
https://drive.google.com/file/d/1ZrfhIkQVdRynthJ2FqGBC5jA93J6yEiZ/view?usp=sharing
```
下载后解压即可，**不需要 root**。

**设置环境变量**（加入 `~/.bashrc`）：
```bash
# ！！按实际安装路径修改！！
export ISAACSIM_PATH="/data/uavlab/isaac_sim_2022.2.0"
# 若用官方 Launcher 安装，路径通常是：
# export ISAACSIM_PATH="${HOME}/.local/share/ov/pkg/isaac_sim-2022.2.0"
source ~/.bashrc
```

---

### 4.2 创建并配置 Conda 环境

```bash
# 1. 创建 Python 3.7 环境（Isaac Sim 要求）
conda create -n sim python=3.7 -y
conda activate sim

# 2. 复制 conda 激活钩子（这是关键！）
#    钩子会在激活时自动 source Isaac Sim 的 setup_conda_env.sh，挂载其 Python 路径
cd /data/uavlab/multi-uav-pursuit2  # 项目根目录
cp -r conda_setup/etc $CONDA_PREFIX

# 3. 重新激活，让钩子生效
conda activate sim
#    此时应看到：
#    "Setup Isaac Sim Conda environment."
#    "Isaac Sim path: /data/uavlab/isaac_sim_2022.2.0"

# 4. 安装本项目包（omni_drones）
pip install -e .
```

---

### 4.3 安装 Git Submodules（tensordict & torchrl）

这两个是**版本锁定**的库，必须从 submodule 安装，不能直接 pip install 最新版：

```bash
cd /data/uavlab/multi-uav-pursuit2

# 拉取 submodule（如果还没 clone 好）
git submodule update --init --recursive

# 安装 tensordict（版本 0.1.2+5e6205c）
cd third_party/tensordict
pip install -e . --no-build-isolation
cd ../..

# 安装 torchrl（版本 0.1.1+e39e701）
cd third_party/torchrl
pip install -e . --no-build-isolation
cd ../..
```

---

### 4.4 安装 DGL（用于 hideandseek_envgen）

```bash
# CUDA 11.7 对应的 dgl wheel
pip install dgl -f https://data.dgl.ai/wheels/torch-1.13/cu117/repo.html
# 注意：本机 conda sim 环境内 PyTorch = 1.13.0+cu117，与上面 url 匹配
```

---

### 4.5 安装其他 pip 依赖

```bash
pip install hydra-core omegaconf wandb setproctitle imageio moviepy plotly
```

---

### 4.6 验证安装

```bash
conda activate sim
# 验证 Isaac Sim Python 路径已挂载
python -c "from omni.isaac.kit import SimulationApp; print('Isaac Sim OK')"

# 验证 PyTorch（应为 1.13.x + cu117）
python -c "import torch; print(torch.__version__, torch.version.cuda)"

# 验证 tensordict & torchrl
python -c "import tensordict; import torchrl; print('tensordict:', tensordict.__version__)"
```

---

## 5. 关键环境变量 & 路径约定

| 变量 | 值（本机） | 说明 |
|------|-----------|------|
| `ISAACSIM_PATH` | `/data/uavlab/isaac_sim_2022.2.0` | Isaac Sim 安装根目录，**必须设置** |
| `CONDA_ENV` | `sim` | 训练/评估均在此环境运行 |
| `DISPLAY` | `:10.0`（SSH 时自动设置） | X11 显示，headless 模式不需要 |
| 项目根 | `/data/uavlab/multi-uav-pursuit2` | `auto_train.sh` 中硬编码 |
| checkpoint 目录 | `<项目根>/checkpoints/` | 按 `<任务>_<时间戳>/` 子目录组织 |
| TensorBoard 日志 | `<项目根>/runs/` | `tb_logdir` |
| 评估视频 | `<项目根>/eval_videos/` | `.mp4` 文件 |
| 训练日志 | `<项目根>/formal_training.log` | `auto_train.sh` 的主日志 |

---

## 6. 快速验证（运行前必做）

```bash
conda activate sim
cd /data/uavlab/multi-uav-pursuit2/scripts

# 50000 步冒烟测试，无 GUI，不写 wandb，约 5~10 分钟
python train.py headless=true wandb.mode=disabled total_frames=50000 task=HideAndSeek
```

**正常输出特征**：
- 看到 `Setup Isaac Sim Conda environment.` 表示 Isaac Sim 路径加载正确
- 看到 `[Warning] [omni.physx...]` 等 PhysX 日志是**正常的**，不是错误
- 看到 tqdm 进度条在滚动，`rollout_fps > 0` 即为成功
- 有 `checkpoint_XXXXX.pt` 文件生成在 `checkpoints/` 下

**常见错误及解决**：见 [第 10 节](#10-常见问题--已知坑)

---

## 7. 训练

### 7.1 单次手动训练（调试用）

```bash
conda activate sim
cd /data/uavlab/multi-uav-pursuit2/scripts

python train.py \
    headless=true \
    wandb.mode=disabled \
    task=HideAndSeek \
    task.env.num_envs=2048 \
    task.use_eval=1 \
    task.use_random_cylinder=0 \
    task.scenario_flag=goal_defense \
    total_frames=2000000000 \
    eval_interval=4000 \
    save_interval=100
```

**常用参数覆盖**（`key=value` 均为 Hydra override 语法）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `task.env.num_envs` | `100` | 并行环境数，越大训练越快，显存消耗越多 |
| `task.scenario_flag` | `goal_defense` | 场景：`goal_defense`/`empty`/`wall`/`narrow_gap`/`passage` |
| `task.num_agents` | `3` | 追捕无人机数量（1~4） |
| `model_dir` | `null` | 载入已有 checkpoint 继续训练，例如 `model_dir=/path/to/checkpoint.pt` |
| `total_frames` | `30000000000` | 总训练步数 |
| `eval_interval` | `-1` | 每隔多少 iteration 做一次评估（-1=禁用） |
| `save_interval` | `100` | 每隔多少 iteration 保存一次 checkpoint |
| `wandb.mode` | `disabled` | `disabled`/`online` |
| `sim.device` | `cuda:0` | 指定 GPU |

---

### 7.2 推荐：自动重启训练循环（长期无人值守）

```bash
cd /data/uavlab/multi-uav-pursuit2

# 后台启动，断开 SSH 不影响
nohup bash scripts/auto_train.sh > auto_train_monitor.log 2>&1 &
echo $! > auto_train.pid

# 查看进度
tail -f auto_train_monitor.log
tail -f formal_training.log

# 停止
kill $(cat auto_train.pid)
```

`auto_train.sh` 功能：
- 自动寻找最新 checkpoint 并 resume 训练
- 内置 watchdog（日志 600 秒无更新 → 判定挂起 → 自动 kill & 重启）
- 崩溃后 30 秒自动重启，最多 100 次
- 每次重启前将本次 checkpoint 归档到 `checkpoints/auto_train_<scenario>/`

---

### 7.3 监控训练

```bash
# TensorBoard（在项目根目录）
tensorboard --logdir runs/ --port 6006
# 然后浏览器打开 http://localhost:6006

# 直接看最近日志
tail -f formal_training.log | grep -E "success|capture|reward|loss"
```

---

## 8. 评估 & 生成视频

```bash
conda activate sim
cd /data/uavlab/multi-uav-pursuit2/scripts

python eval.py \
    headless=false \
    wandb.mode=disabled \
    task=HideAndSeek \
    task.use_eval=1 \
    task.use_random_cylinder=0 \
    task.scenario_flag=goal_defense \
    model_dir=/data/uavlab/multi-uav-pursuit2/checkpoints/<子目录>/checkpoint_XXXXX.pt
```

- **视频保存路径**：`eval_videos/<任务名>_<时间戳>.mp4`
- `headless=false` 时需要显示器或 VirtualGL，SSH 下建议保持 `headless=true` 然后再看视频文件
- 终止条件会打印在终端（捕获成功 / 目标进守区 / 坠机 / 超时）

---

## 9. 目录结构说明

```
multi-uav-pursuit2/
├── cfg/
│   ├── train.yaml              # 全局训练配置（随机种子、viewer、wandb 等）
│   ├── algo/
│   │   └── mappo.yaml          # MAPPO 超参（学习率、网络结构、entropy等）
│   └── task/
│       ├── HideAndSeek.yaml    # ★ 主任务配置（奖励系数、场景、课程等）
│       ├── HideAndSeek_envgen.yaml  # 自适应环境生成器版本
│       └── HideAndSeek_deploy.yaml  # 二阶段 Sim2Real 微调版本
├── conda_setup/
│   └── etc/conda/activate.d/
│       └── env_vars.sh         # conda activate 时自动 source Isaac Sim 路径
├── scripts/
│   ├── train.py                # 训练入口
│   ├── eval.py                 # 评估+录像入口
│   ├── auto_train.sh           # 自动重启训练循环（推荐长期训练用）
│   ├── train_deploy.py         # Sim2Real 二阶段微调入口
│   └── train_generator.py      # 自适应环境生成器训练入口
├── omni_drones/
│   ├── __init__.py             # CONFIG_PATH, init_simulation_app
│   ├── envs/
│   │   └── hide_and_seek/
│   │       └── hideandseek.py  # ★ 环境主逻辑（观测/奖励/重置）
│   ├── learning/               # MAPPO / PPO 算法实现
│   ├── controllers/            # PIDrate / Lee 等低级控制器
│   └── utils/                  # TorchRL 辅助、wandb 工具等
├── third_party/
│   ├── tensordict/             # git submodule，版本锁定
│   └── torchrl/                # git submodule，版本锁定
├── checkpoints/                # 训练产出的模型权重
├── runs/                       # TensorBoard 事件文件
├── eval_videos/                # 评估视频
└── docs/                       # 设计文档（含本文件）
```

---

## 10. 常见问题 & 已知坑

### ❌ `ModuleNotFoundError: No module named 'omni'`

**原因**：conda hook 没有成功挂载 Isaac Sim 路径。

**解决**：
```bash
# 检查 ISAACSIM_PATH 是否正确
echo $ISAACSIM_PATH
ls $ISAACSIM_PATH/setup_conda_env.sh   # 文件必须存在

# 重新复制 hook 并重新激活
cp -r conda_setup/etc $CONDA_PREFIX
conda deactivate && conda activate sim
# 应该看到 "Setup Isaac Sim Conda environment."
```

---

### ❌ conda activate 时报 pydantic ImportError

**现象**：
```
Error while loading conda entry point: conda-anaconda-tos 
(cannot import name 'ConfigDict' from 'pydantic' ...)
```

**原因**：Isaac Sim 内置的 pydantic 版本（v1）与 conda 新插件冲突。  
**影响**：**仅影响 conda 命令行提示，不影响 Python 代码执行**，可放心忽略。

---

### ❌ CUDA Out of Memory（OOM）

**原因**：GPU 有残留进程（Isaac Sim 崩溃后常见）。

**解决**：执行完整的显存清理流程（见[第 11 节](#11-gpu-显存清理)）。

---

### ❌ `RuntimeError: CUDA error: no kernel image is available for execution on the device`

**原因**：CUDA 版本与 PyTorch wheel 不匹配。  
**说明**：conda sim 环境内使用的是 **Isaac Sim 自带的 PyTorch 1.13.0+cu117**，  
这与系统安装的 CUDA 12.8 不同，但 NVIDIA 驱动向下兼容，正常情况下可以运行。  
若出现此错误，检查是否有其他 PyTorch 版本抢占了 PYTHONPATH。

---

### ❌ 训练启动后立即崩溃，日志极少

**检查步骤**：
```bash
# 1. 检查 Isaac Sim 路径
ls $ISAACSIM_PATH/setup_conda_env.sh

# 2. 检查 GPU 是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 3. 检查 headless 配置（服务器无显卡输出时必须 headless=true）
# 4. 查看完整错误日志（日志文件比终端输出更详细）
tail -200 formal_training.log
```

---

### ❌ `tensordict` 或 `torchrl` 版本错误

如果出现相关 API 报错，重新从 submodule 安装：
```bash
cd third_party/tensordict && pip install -e . --no-build-isolation && cd ../..
cd third_party/torchrl && pip install -e . --no-build-isolation && cd ../..
```

---

### ⚠️ SSH 断连后训练中断

**预防**：使用 `nohup` + `&` 启动，或用 `screen`/`tmux`：
```bash
# 方法 1：nohup（推荐配合 auto_train.sh）
nohup bash scripts/auto_train.sh > auto_train_monitor.log 2>&1 &

# 方法 2：tmux
tmux new -s train
# ... 在 tmux 中启动训练 ...
# Ctrl+B, D 脱离会话
```

---

### ⚠️ Isaac Sim 启动非常慢（正常现象）

Isaac Sim 首次启动编译着色器，可能需要 **5~15 分钟**，此后约 60~120 秒。  
在日志中看到类似 `Isaac Sim version: 2022.2.0` 之后才是真正的训练开始。

---

## 11. GPU 显存清理

Isaac Sim 崩溃后极易留下隐形僵尸进程（名称被改为 `dummy-igxwalhz` 类似形式），  
占用大量显存但无法被普通 `pkill` 清除。**每次重启训练前必须执行此流程**：

```bash
# 第一步：杀掉已知训练相关进程
pkill -9 -f auto_train.sh
pkill -9 -f scripts/train.py
pkill -9 -u uavlab -f kit

# 第二步：查看 GPU 进程列表
nvidia-smi

# 第三步：精确清理残留进程
# 在 nvidia-smi 输出的 Processes 列表中，找到占用几 GB 显存的未知进程
# 记录其 PID，然后：
kill -9 <PID1> <PID2> ...

# 第四步：二次确认显存已归零（单卡使用 < 1 GB 才算干净）
nvidia-smi
```

---

## 12. 本机实际配置速查

| 项目 | 值 |
|------|----|
| GPU 0 / GPU 1 | NVIDIA GeForce RTX 4090（各 24 GB） |
| NVIDIA 驱动版本 | 580.126.09 |
| Isaac Sim 版本 | 2022.2.0 |
| Isaac Sim 路径 | `/data/uavlab/isaac_sim_2022.2.0` |
| Conda 位置 | `/home/uavlab/miniconda3` |
| Conda 环境名 | `sim` |
| Python 版本（sim env） | 3.7.16 |
| PyTorch（sim env） | 1.13.0+cu117（Isaac Sim 内置） |
| CUDA（Isaac Sim） | 11.7 |
| tensordict 版本 | 0.1.2+5e6205c（git submodule） |
| torchrl 版本 | 0.1.1+e39e701（git submodule） |
| dgl 版本 | 1.1.3 |
| hydra-core | 1.3.2 |
| omegaconf | 2.3.0 |
| wandb | 0.18.7 |
| 项目根目录 | `/data/uavlab/multi-uav-pursuit2` |
| Miniconda base | `/home/uavlab/miniconda3` |
| 默认训练 GPU | `cuda:0`（可通过 `sim.device=cuda:1` 切换） |
| 默认并行环境数 | `num_envs=2048`（auto_train.sh）/ `100`（yaml 默认） |
| 默认场景 | `goal_defense` |

---

## 快速参考：最常用命令

```bash
# ── 激活环境 ──────────────────────────────────────────────────
conda activate sim
cd /data/uavlab/multi-uav-pursuit2

# ── 冒烟测试 ──────────────────────────────────────────────────
python scripts/train.py headless=true wandb.mode=disabled total_frames=50000 task=HideAndSeek

# ── 正式训练（后台、自动重启、2048 envs）────────────────────
nohup bash scripts/auto_train.sh > auto_train_monitor.log 2>&1 &

# ── 停止训练 ──────────────────────────────────────────────────
kill $(cat auto_train.pid)

# ── 评估最新权重 ──────────────────────────────────────────────
CKPT=$(ls -t checkpoints/auto_train_goal_defense/checkpoint_*.pt 2>/dev/null | head -1)
python scripts/eval.py headless=true wandb.mode=disabled task=HideAndSeek \
    task.use_eval=1 task.scenario_flag=goal_defense model_dir=${CKPT}

# ── TensorBoard ────────────────────────────────────────────────
tensorboard --logdir runs/ --port 6006

# ── 显存清理 ──────────────────────────────────────────────────
pkill -9 -f scripts/train.py; pkill -9 -u uavlab -f kit; nvidia-smi
```

---

*文档维护：请在做出重要环境变更（如更新 Isaac Sim、切换 GPU、修改路径）后同步更新本文件。*
