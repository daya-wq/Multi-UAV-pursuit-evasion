---
description: 自动复盘与RL训练迭代 (Automated Postmortem and RL Training Loop)
---

当用户请求自动复盘、测试、调优当前的多智能体强化学习策略时（例如使用 slash command `/rl_autoloop` 或提及复盘迭代），请严格遵守以下工作流：

1. **找到最新/目标 run 与 checkpoint**
   - 从 `runs/` 和 `checkpoints/` 中提取最新的或问题退化前的权重目录。
   - 如果缺少相关路径信息，可尝试自动发现：
     ```bash
     python3 /home/uavlab/.codex/skills/rl-run-postmortem/scripts/locate_run_artifacts.py --repo-root /data/uavlab/multi-uav-pursuit2
     ```
   - 使用的虚拟环境是conda里的sim


3. **导出或读取 TensorBoard 关键指标**
   - 从 `events.out.tfevents*` 中提取核心数据：
     - **第一步：看 reward 曲线趋势** (return, distance/progress reward, catch reward, collision reward, speed reward, smoothness reward, landed penalty)
     - **第二步：看优化稳定性** (value loss, policy loss, entropy/KL, drone/entropy, drone/ESS, drone/actor_grad_norm, drone/advantages_*)
     - **第三步：看任务结果** (success rate, episode length, first_capture_step, collision, timeout/floor collision)
   - 根据曲线找出这一次run里合适的几个历史权重文件



2. **跑测试与录像 (Evaluation & Video Generation)**
   - 使用带 UI/渲染逻辑配置的 `scripts/eval.py` 对这一次run的历史权重和最新权重进行测试。
   - 收集生成的评测视屏（位于 `eval_videos/*.mp4` 或 `videos/eval/*.mp4`）。


4. **进行表现校对**
   - 观察导出的视频中无人机的行为和飞行动作，第四步：最后看视频行为是否与指标一致。
   - 如果指标和视频不一致，必须明确指出不一致，并且通过视频分析的结果结合前面的奖励和损失去补充额外信息；通过无人机的行为判断目前策略存在的明显问题。
   - 对比以前不同reard/config/checkpoint起点的run，对比哪些指标编号了，哪些变差了

5. **生成诊断报告与修改建议**
   - **此时必须明确调用 Codex 辅助你进行深度诊断。在你的 prompt/thought 中显式写出：**
     `Use the rl-run-postmortem skill.`
   - 基于提取的日志与视频表象，使用以下启发式规则得出诊断：
     1. reward 上升但视频仍在“追而不堵” -> 奖励设计偏向跟踪，不鼓励截断逃逸方向或协同包夹
     2. value loss 大幅震荡、policy 更新不稳定 -> 优先检查 reward scale、advantage normalization、学习率、clip range
     3. 视频中频繁急转、抖动、绕圈 -> 动作平滑项不足，角速度/推力惩罚不合理
     4. 长时间尾随、不抢占前方空间 -> 缺少“预测拦截位点”奖励或团队角色分化机制
     5. 多机扎堆追一个方向 -> 缺少编队分散、覆盖角度或包围几何奖励
     6. 把当前run和以前别的组的参数，结果，视频行为做对比，不能只孤立分析当前run

6. **修改代码配置**
   - 每轮只围绕一个主假设做最小改动。高风险改动必须标注风险。
   - 优先级必须是：1. 先修错（动作/环境逻辑bug）-> 2. 再改 reward 权重 -> 3. 再改 reward 设计 -> 4. 改观测设计 -> 5. 改动作约束 -> 6. 最后改算法主干。
   - 每次修改代码前需要在终端输出详细原因，需要确认之后才执行
  

7. **更新工程文档**
   - 自动更新 `docs/training_history_analysis.md`
   - 自动记录本轮改动摘要到 `docs/code_changes_report.md`

8. **新建或重启下个 Run**
   - 确定是 `resume-latest`、`resume-earlier` 还是 `restart-scratch`。
   - **严格前置检查**：每次启动新的训练前，必须调用并执行 `/clean_training_env` 工作流彻底清理僵尸进程与显存，并且 **必须征求用户的明确同意** 之后，才能在终端启动新一轮的训练任务。

**最终你的输出分析与结论报告必须固定为如下格式：**
- 环境设定 (必须写清楚追捕与目标无人机的奖励公式，以及它们的初始精确位置)
- 现象
- 可能根因
- 证据，证据里要加入以前run的对比分析
- 建议修改
- 风险
- 下一轮要验证什么