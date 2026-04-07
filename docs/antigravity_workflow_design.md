# Antigravity RL 自动化工作流设计

这份文档不是 antigravity 官方 DSL 文档，而是一份可迁移的工作流蓝图。

原因很简单：我在本机没有找到 antigravity 的公开 workflow 语法说明，所以这里给你的不是“伪装成官方语法的假 YAML”，而是一份你可以直接翻译到 antigravity 工作流节点里的设计稿。

目标只有一个：

**让训练结束后，系统自动完成“评测 -> 诊断 -> 小修 -> 起下一轮”的闭环。**

---

## 1. 推荐架构

不要把所有事都塞进一个 agent。

最稳妥的是两层：

### 1. `workflow` 层

负责机械执行：

- 监控训练进程
- 检测训练是否停止、卡住、到达里程碑
- 找最新 run / checkpoint / 视频 / 日志
- 自动评测若干 checkpoint
- 调用 Codex
- 根据 Codex 产出决定是否自动改代码、是否起下一轮训练

### 2. `skill` 层

负责让 Codex 每次都按同一套方法分析：

- 先看 outcome，再看 chase quality，再看 collision，再看 optimizer
- 先看视频行为，再下结论
- 每次只选一个主失败模式
- 每次只做一组最小改动
- 每次明确回答：
  - 继续最新权重
  - 回滚早期权重
  - 从零开始

所以结论是：

- **workflow = 自动执行器**
- **skill = 标准化大脑**

---

## 2. 你现在这套流程，最适合拆成哪些节点

建议拆成 8 个节点。

### 节点 1. 训练监控 `monitor_run`

触发条件建议有 3 类：

1. 训练进程退出
2. 长时间没有新 checkpoint
3. 到达里程碑步数

推荐阈值：

- `on_exit`: 进程结束就触发
- `on_stall`: 90 分钟没有新 checkpoint 就触发
- `on_milestone`: 每 `20M` 或 `50M` frame 触发一次

### 节点 2. 工件定位 `locate_artifacts`

调用：

```bash
python3 /home/uavlab/.codex/skills/rl-run-postmortem/scripts/locate_run_artifacts.py --repo-root /data/uavlab/multi-uav-pursuit2
```

它会返回：

- 最新 run 目录
- 对应 checkpoint 目录
- 最新 checkpoint
- 最近视频
- 关键日志

### 节点 3. 选择评测 checkpoint `select_eval_targets`

不要只测最后一个 checkpoint。

推荐固定评测 3 个：

1. `latest`
2. `latest - 1`
3. “健康窗口候选”

健康窗口候选的规则：

- 如果你有历史分析，优先测早期成功出现的区间
- 否则测训练前 1/3、1/2、2/3 位置附近的 checkpoint

### 节点 4. 自动评测 `evaluate_targets`

对每个 checkpoint 做：

- 运行评测脚本
- 保存视频
- 记录 checkpoint 路径和视频路径

这里建议你后面再补一个统一评测脚本，例如：

- `scripts/eval_checkpoint.sh`
- 输入：checkpoint 路径
- 输出：视频路径 + 简要结果路径

### 节点 5. 收集上下文 `build_context`

整理给 Codex 的输入：

- run 名称
- run 目录
- checkpoint 列表
- 视频列表
- 日志路径
- 当前任务配置
- 当前算法配置
- 历史分析文档路径

### 节点 6. 调用 Codex 做 postmortem `codex_postmortem`

这一节点要显式调用 skill。

建议固定 prompt 模板，见下文。

### 节点 7. 决策分流 `decision_gate`

Codex 返回后，至少做这 3 类分支：

1. `safe_patch`
   - 只改 reward 系数、终止条件、日志补充
   - 可以自动执行
2. `risky_patch`
   - 涉及动作空间、网络结构、物理参数、训练入口
   - 需要你确认
3. `no_patch_resume`
   - 不改代码，只换 checkpoint 或直接继续跑

### 节点 8. 启动下一轮 `launch_next_run`

根据 Codex 的结果决定：

- `resume-latest`
- `resume-earlier`
- `restart-scratch`

并把这次分析结果写回文档。

---

## 3. 一版可迁移的 workflow 蓝图

下面这段是逻辑蓝图，不是官方语法。

```yaml
workflow:
  name: rl-autoloop

  inputs:
    repo_root: /data/uavlab/multi-uav-pursuit2
    milestone_frames: 50000000
    stall_minutes: 90
    auto_apply_safe_patches: true

  triggers:
    - type: process_exit
      match: scripts/train.py
    - type: checkpoint_stall
      path: ${repo_root}/checkpoints
      minutes: ${stall_minutes}
    - type: frame_milestone
      interval: ${milestone_frames}

  steps:
    - id: locate_artifacts
      type: shell
      run: |
        python3 /home/uavlab/.codex/skills/rl-run-postmortem/scripts/locate_run_artifacts.py \
          --repo-root ${repo_root} > .workflow/latest_artifacts.json

    - id: select_eval_targets
      type: script_or_logic
      input: .workflow/latest_artifacts.json
      output: .workflow/eval_targets.json
      logic:
        - choose latest checkpoint
        - choose previous checkpoint
        - choose one earlier "healthy phase" checkpoint

    - id: evaluate_targets
      type: shell
      foreach: .workflow/eval_targets.json
      run: |
        bash scripts/eval_checkpoint.sh ${item.checkpoint}

    - id: codex_postmortem
      type: codex
      model: codex
      prompt_file: .workflow/prompts/postmortem_prompt.txt
      attachments:
        - .workflow/latest_artifacts.json
        - .workflow/eval_targets.json
        - docs/training_history_analysis.md
        - cfg/task/HideAndSeek.yaml
        - cfg/algo/mappo.yaml
      output: .workflow/postmortem_report.md

    - id: parse_decision
      type: parser
      input: .workflow/postmortem_report.md
      extract:
        - primary_failure_mode
        - resume_strategy
        - code_changes
        - safe_or_risky

    - id: apply_patch_if_safe
      type: conditional
      when: safe_or_risky == safe && auto_apply_safe_patches == true
      then:
        - call codex_apply_patch

    - id: request_human_review_if_risky
      type: conditional
      when: safe_or_risky == risky
      then:
        - notify_human
        - wait_for_approval

    - id: launch_next_run
      type: shell
      run: |
        bash scripts/start_next_run.sh --strategy ${resume_strategy}
```

---

## 4. Codex 节点的 prompt 怎么写

你的关键点不在于“让 Codex 自由发挥”，而在于“强制它按固定协议分析”。

如果你想直接用一份现成的长提示词，见：

- [antigravity_autoloop_skill.md](/data/uavlab/multi-uav-pursuit2/docs/antigravity_autoloop_skill.md)

Codex 节点的 prompt 建议写成这样：

```text
Use the `rl-run-postmortem` skill.

Repository root: /data/uavlab/multi-uav-pursuit2

Inputs:
- latest_artifacts.json
- eval_targets.json
- docs/training_history_analysis.md
- cfg/task/HideAndSeek.yaml
- cfg/algo/mappo.yaml

Tasks:
1. Analyze the latest RL run using the skill workflow.
2. Review both TensorBoard outcome/control/optimizer signals and the evaluation videos.
3. Identify exactly one primary failure mode.
4. Propose the smallest coherent code/config change set.
5. Choose one resume strategy: resume-latest, resume-earlier, or restart-scratch.
6. Follow the report structure in references/report-template.md.
7. If the required changes are low-risk, implement them directly.
8. Update docs/training_history_analysis.md with a short new section for this run.

Constraints:
- Do not change more than one hypothesis per iteration.
- Do not rewrite the whole reward function unless strictly necessary.
- Prefer earlier checkpoints over latest if later checkpoints clearly regressed.
```

这段 prompt 的重点是两件事：

1. **直接点名 skill**
   - `Use the rl-run-postmortem skill`
2. **强制输出结构**
   - 不让 agent 每次都换一种说法

---

## 5. antigravity 里怎么调用这个 skill

这里要分清一件事：

**antigravity 不直接执行 skill。**

真正执行 skill 的是 **Codex**。  
antigravity 做的是：

- 触发一个 Codex 节点
- 在 prompt 里显式要求使用这个 skill

也就是说，调用链是：

`antigravity workflow -> Codex 节点 -> rl-run-postmortem skill`

### 前提条件

Codex 所在环境必须能看到这个 skill：

- [SKILL.md](/home/uavlab/.codex/skills/rl-run-postmortem/SKILL.md)

也就是：

- skill 必须放在该 Codex 实例的 `$CODEX_HOME/skills` 或默认 skills 目录里
- 你当前机器上就是 `/home/uavlab/.codex/skills/rl-run-postmortem`

### 最稳妥的调用方式

在 antigravity 的 Codex 节点 prompt 里明确写：

```text
Use the `rl-run-postmortem` skill.
```

因为 Codex 的技能触发机制本来就支持：

- 用户直接点名 skill
- 或任务和 skill 描述强匹配

但工程上不要赌“它会不会自动匹配成功”。  
**显式点名最稳。**

---

## 6. 你以后怎么把这一套自动化起来

建议分 3 个阶段做，不要一步到位。

### 阶段 A. 半自动闭环

先实现：

1. 训练结束自动触发 workflow
2. 自动找工件
3. 自动评测 checkpoint
4. 自动调用 Codex + skill
5. 自动生成报告
6. 由你人工决定是否应用 patch、是否启动下一轮

这是最值得先落地的版本，因为：

- 风险低
- 很快就能省掉 80% 的重复劳动

### 阶段 B. 小改动自动执行

再加：

- reward 系数改动自动执行
- 终止条件改动自动执行
- 文档更新自动执行
- checkpoint 选择自动执行
- 下一轮训练自动启动

但仍保留一条规则：

- 涉及动作空间、网络结构、物理参数、训练入口切换时，必须人工确认

### 阶段 C. 全自动实验代理

最后再做：

- 连续多轮实验
- 自动比较多组结果
- 自动回滚到最佳 checkpoint
- 自动生成训练日报 / 周报

只有前两阶段跑稳了，才值得上这一层。

---

## 7. 你下一步最应该补的 3 个自动化脚本

你现在已经有了工件定位脚本：

- `/home/uavlab/.codex/skills/rl-run-postmortem/scripts/locate_run_artifacts.py`

接下来最值得补的是这 3 个：

### 1. `scripts/eval_checkpoint.sh`

职责：

- 输入 checkpoint
- 跑评测
- 输出视频路径

### 2. `scripts/export_tb_scalars.py`

职责：

- 从 `events.out.tfevents.*` 里提取关键 tag
- 输出 JSON

推荐至少导出：

- `success`
- `catch_reward`
- `first_capture_step`
- `return`
- `distance_reward`
- `collision*`
- `smoothness*`
- `landed_penalty`
- `any_landed`
- `entropy`
- `actor_log_std_*`
- `action_norm`
- `cmd_norm`
- `ESS`

### 3. `scripts/start_next_run.sh`

职责：

- 根据策略决定从哪个 checkpoint 起跑
- 统一训练命令
- 统一日志保存位置

没有这三个脚本，workflow 会停留在“自动分析”；  
有了这三个脚本，workflow 才会真正变成“自动闭环”。

---

## 8. 你这套系统里，Codex 和 antigravity 的分工

建议固定成这样：

### antigravity

- 监控
- 调度
- 执行命令
- 串联节点
- 控制自动/人工审批

### Codex

- 读代码
- 读日志
- 读图和视频
- 诊断失败模式
- 生成最小改动
- 改代码
- 更新文档

### skill

- 固定 Codex 的方法论
- 固定输出结构
- 固定失败模式标签

一句话：

- **antigravity 管流程**
- **Codex 管分析和改代码**
- **skill 管分析标准**

---

## 9. 最短可落地版本

如果你现在只想尽快跑起来，最小实现建议就是：

1. antigravity 监听训练停止
2. 调用 `locate_run_artifacts.py`
3. 跑一次最新 checkpoint 评测
4. 把：
   - 最新 run
   - 最新 checkpoint
   - 评测视频
   - 训练历史文档
   - 当前配置
   喂给 Codex
5. prompt 里强制：
   - `Use the rl-run-postmortem skill`
6. Codex 输出：
   - 主失败模式
   - 最小改动
   - 下一轮从哪起跑
7. 你人工确认后启动下一轮

这版就已经非常有价值了。

---

## 10. 当前状态总结

你现在已经有：

- 一个可用的 Codex skill：
  - [SKILL.md](/home/uavlab/.codex/skills/rl-run-postmortem/SKILL.md)
- 一个失败模式参考：
  - [failure-modes.md](/home/uavlab/.codex/skills/rl-run-postmortem/references/failure-modes.md)
- 一个固定输出模板：
  - [report-template.md](/home/uavlab/.codex/skills/rl-run-postmortem/references/report-template.md)
- 一个工件定位脚本：
  - [locate_run_artifacts.py](/home/uavlab/.codex/skills/rl-run-postmortem/scripts/locate_run_artifacts.py)

你还缺的，是 workflow 侧的三个执行脚本和一个实际落地的 antigravity 节点编排。

这部分我下一步可以继续帮你写。
