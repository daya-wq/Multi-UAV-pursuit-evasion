---
description: 如何彻底清理多智能体训练进程与释放僵尸显存
---
1. 首先，下达基础的批量清理指令，杀掉常规的 Python 和 Shell 父子进程：
   `pkill -9 -f auto_train.sh`
   `pkill -9 -f scripts/train.py`
   `pkill -9 -u uavlab -f kit`
   
2. **绝对不能停在这一步！** 由于 Isaac Sim 与 TorchRL 的工作机制，它们极易在后台残留被重命名为隐藏名称（如 `dummy-igxwalhz`）的僵尸进程。仅仅 `pkill` 根本无法扫荡干净，这会导致数十GB的显存被永久霸占，并在下次启动时瞬间触发 CUDA Out Of Memory 从而“假死”。

// turbo-all
3. 基础清理后，**必须强制**执行以下探测命令：
   `nvidia-smi`

4. 仔细核对 `nvidia-smi` 输出结果下面的 `Processes` 列表。如果发现任意未知进程（特别是吃掉了几 GB 或十多 GB 显存的 Python、kit 或 dummy 进程），必须立即提取对应 PID，使用精确打击：
   `kill -9 <残留进程PID>`

5. 在启动全新一轮训练的命令之前，必须二次确认所指定的 GPU (如 GPU 1) 显存已经彻底归零（剩余使用在 1GB 以下），这样才算彻底完成了环境重置。
