# 项目审阅与维护清单（含论文对照）

最后更新：2026-03-22

## 1. 项目结构速览

这个仓库的主线比较清楚：

- `scripts/`：训练、评估、部署训练入口。
- `cfg/`：Hydra 配置，任务和算法参数基本都在这里。
- `omni_drones/envs/hide_and_seek/`：追逃环境主体，`hideandseek.py`、`hideandseek_envgen.py`、`hideandseek_deploy.py` 是三个主要变体。
- `omni_drones/learning/`：MAPPO 和 PPO 系列策略实现。
- `omni_drones/envs/isaac_env.py`：Isaac Sim 环境基类。
- `docs/source/`：Sphinx 文档，但当前内容仍以上游 `OmniDrones` 为主，和本项目不完全一致。

我本次重点阅读了 `README.md`、`cfg/`、`scripts/`、`omni_drones/envs/isaac_env.py`、`omni_drones/envs/hide_and_seek/`、`omni_drones/learning/mappo.py`、`docs/source/`。

## 2. 论文对照结论

本轮对照的论文是 `docs/Online Planning for Multi-UAV Pursuit-Evasion in Unknown Environments Using Deep Reinforcement Learning.pdf`。

### 2.1 已经和论文基本对齐的部分

1. OPEN 的三个核心模块在代码里都能找到对应实现。
   - Evader Prediction-Enhanced Network: `TP_net` 用 LSTM 预测未来轨迹，预测结果拼进 `state_self`。
   - Attention-based encoder: actor / critic 的 encoder 使用 `PartialAttentionEncoder`。
   - Adaptive Environment Generator: `HideAndSeek_envgen` 里有 buffer、局部扩展和全局随机采样逻辑。

2. 论文里一些关键任务设定在代码中也是一致的。
   - `history_step = 10`、`future_predcition_step = 5`。
   - `obs_max_cylinder = 3`，对应论文里的 `k = 3` 最近障碍。
   - 训练环境 `arena_size = 0.9`、`max_height = 1.2`、`catch_radius = 0.3`、`v_drone = 1.0`、`v_prey = 1.3`。
   - AEG 的 `ratio_unif = 0.3` 对应论文里的 `p = 0.7` 局部扩展 / `0.3` 全局探索。
   - `R_min = 0.5`、`R_max = 0.9` 对应论文里 active archive 的保留区间。

3. 论文中的部分观测细节在代码中也有体现。
   - 目标被遮挡时使用 `mask_value = -5`。
   - 目标检测通过“无遮挡 + 距离范围内”广播给所有无人机。
   - 动作侧确实走了 CTBR 风格接口，随后交给 `PIDrate` 控制器。

### 2.2 按论文来看，需要优先修正的内容

1. `HideAndSeek_deploy` 的默认配置并没有真正开启论文的第二阶段奖励。
   - 论文写的是第二阶段重新引入 smoothness reward。
   - 但当前 `cfg/task/HideAndSeek_deploy.yaml` 默认 `use_deployment: 0`，会把 smoothness reward 直接置零。
   - 这意味着用户按 README 直接跑 `train_deploy.py`，默认并不会得到论文里的 Two-stage reward refinement。

2. 第二阶段奖励参数和论文文字没有对齐，至少需要补“复现实验配置”。
   - 论文明确写 smoothness reward coefficient = `2.0`。
   - 当前代码默认是 `init_smoothness_coef = 20.0`、`max_smoothness_coef = 20.0`，而且 `smooth_lr = 0.0`，实际是常数而不是 refinement 过程。
   - 同时，`HideAndSeek_deploy` 还把 `catch_reward_coef` 从第一阶段的 `60.0` 改成了 `20.0`。这可能是作者调参，但不是论文里直接说明的内容。
   - 建议：单独提供一份“paper reproduction”配置，把论文数值和工程调参值明确区分开。

3. 当前评估脚本无法直接复现实验部分的统计结果。
   - 论文写的是 4 个测试场景、300 个 episode、每个 seed 100 个 episode。
   - 现在 `scripts/eval.py` 只做一次 rollout 聚合，没有三组 seed 的评测驱动，也没有一键跑 4 个场景的脚本。
   - 建议：补一个专门的 paper eval runner，输出 `capture rate / capture step / collision rate` 三个指标，并固定种子和 episode 数。

4. 论文中的 ablation 变体没有在仓库里形成可复现实验入口。
   - 论文区分了 `MAPPO`、`MAPPO + EPN`、`MAPPO + AEG`、`OPEN`。
   - 真实部署部分又区分了 `w/o Smoothness`、`One-stage`、`Two-stage`。
   - 当前仓库只有若干 task 配置，没有对应的命名配置或批处理脚本，不利于复现实验表格。
   - 建议：把这些论文变体各自落成 yaml override 或脚本入口。

5. 论文复现依赖的问题依旧存在，而且这次更明确了优先级。
   - 随机种子不一致、设备硬编码、`eval.py` 不强制 checkpoint，这些都会直接影响论文结果是否能被稳定复现。
   - 这几项即使不从工程卫生看，从论文复现角度也应该放在最高优先级。

## 3. 需要优先修改的内容

### P0: 直接影响正确性、复现性或使用体验

1. 修复随机种子配置不一致的问题。
   - `scripts/train.py`、`scripts/train_generator.py`、`scripts/train_deploy.py` 都先写死了 `seed = 42`，随后又调用 `env.set_seed(cfg.seed)`。
   - 这会导致一部分随机源由 42 控制，另一部分由配置里的 `cfg.seed` 控制，复现实验时容易出现“同一个 seed 跑不出同样结果”的情况。
   - 建议：统一只使用 `cfg.seed`，并把 Python / NumPy / Torch / Isaac Sim 的 seed 初始化收口到一个公共函数里。

2. 去掉设备硬编码，改为完全跟随配置。
   - `scripts/train.py`、`scripts/train_generator.py`、`scripts/train_deploy.py`、`scripts/eval.py` 创建策略时写死了 `device="cuda"`。
   - `omni_drones/envs/isaac_env.py` 创建 `SimulationContext` 时写死了 `device="cuda:0"`。
   - 这会让多 GPU、非 0 号卡、CPU 调试都变得不可靠，也和 `cfg/base/sim_base.yaml` 里的 `sim.device` 形成冲突。
   - 建议：统一透传 `cfg.sim.device`，不要在代码里再写常量。

3. 修复 `IsaacEnv.close()` 被直接短路的问题。
   - `omni_drones/envs/isaac_env.py` 的 `close()` 一进入就 `return`，后面的仿真停止、回调清理、stage 清理逻辑完全不会执行。
   - 长时间调试、批量实验、重复创建环境时，容易留下 Isaac Sim 资源和状态。
   - 建议：恢复关闭逻辑，并补一个最小化的资源释放验证。

4. 让 `eval.py` 在没有 checkpoint 时直接报错，而不是静默评估未训练策略。
   - `scripts/eval.py` 只有在 `cfg.model_dir is not None` 时才加载权重；默认配置里 `model_dir: null`。
   - `README.md` 当前写法是直接执行 `python eval.py`，这很容易让人误以为自己在评估训练好的模型。
   - 建议：`eval.py` 在 `model_dir` 为空时 `raise ValueError`，并同步改 README 的示例命令。

5. 把论文中的第二阶段部署训练默认行为修正到可直接复现。
   - 当前 `HideAndSeek_deploy` 默认不会启用 smoothness reward。
   - 建议至少做两件事：一是默认打开 `use_deployment`，二是单独提供论文复现配置，把第二阶段的 reward 参数写清楚。

### P1: 明显的维护性问题，后续改功能会越来越痛

1. 抽取 `scripts/` 下重复的训练入口逻辑。
   - `train.py`、`train_generator.py`、`train_deploy.py` 大部分代码是复制出来的，差异只有少量逻辑。
   - 当前重复部分包括：seed 初始化、环境包装、action transform、评估函数、日志与 checkpoint 流程。
   - 建议：抽一个公共 runner，例如 `scripts/common.py` 或 `omni_drones/runner/`，把差异点变成参数或 hook。

2. 合并三个追逃环境的大量重复代码。
   - `hideandseek.py`、`hideandseek_envgen.py`、`hideandseek_deploy.py` 都是超大单文件，而且前半段工具函数和主体逻辑高度重复。
   - 当前做法会导致一个 bug 修复要在 2 到 3 个文件里重复改，行为漂移风险很高。
   - 建议：抽成一个共享基类，三个环境只保留“任务生成策略”和“奖励/部署特化”差异。

3. 补全安装依赖，不要只靠 README 口头说明。
   - `setup.py` 目前只声明了少数依赖，但脚本和环境实际还依赖 `setproctitle`、`tqdm`、`numpy`、`matplotlib`，环境生成还依赖 `dgl`。
   - 现在仓库更像“熟悉项目的人可以手工配起来”，不太像“新环境按说明就能跑起来”。
   - 建议：把核心依赖写进安装配置；可选依赖用 extras，比如 `.[train]`、`.[envgen]`、`.[docs]`。

4. 增加最小化自动化测试。
   - 本次没有发现项目自己的测试集；当前只能靠手工跑 Isaac Sim。
   - 建议至少补三类 smoke test：
   - 配置加载测试：Hydra 配置能正常解析。
   - 导入测试：核心模块可导入。
   - 纯 Python 逻辑测试：任务生成、几何工具、buffer 操作等不依赖 Isaac 的部分。

5. 为论文结果补专门的复现实验入口。
   - 当前缺少四个测试场景的统一评估脚本。
   - 当前也缺少 `MAPPO / MAPPO+EPN / MAPPO+AEG / OPEN` 和 `w/o Smoothness / One-stage / Two-stage` 的标准配置。
   - 建议把论文表格对应的配置和命令直接固化到 `cfg/` 与 `scripts/`。

### P2: 工程卫生和文档一致性

1. 把文档从上游 `OmniDrones` 语境改成当前项目语境。
   - `docs/source/conf.py` 里的 `project`、仓库地址仍然指向 `OmniDrones`。
   - `docs/source/index.rst` 首页标题和介绍也还是上游平台描述。
   - 建议：至少先把项目名、仓库地址、论文引用和任务说明对齐。

2. 清理实验特定默认值，补一个安全的 smoke config。
   - `cfg/train.yaml` 里有明显实验阶段默认值，例如超大的 `total_frames`、固定的 wandb entity/project。
   - 这些值不适合作为仓库级默认配置。
   - 建议：仓库默认配置用于最小可运行验证，正式实验参数放到单独 override 文件。

3. 去掉残留的调试输出。
   - `omni_drones/envs/isaac_env.py` 启动时会 `pprint(self.fake_tensordict().shapes)`。
   - `omni_drones/learning/mappo.py` 初始化时会直接打印 observation/action spec。
   - `hideandseek_envgen.py` 等文件里也还有调试 `print`。
   - 建议：统一改成可控日志，默认关闭。

## 4. 推荐修改顺序

建议按下面顺序处理：

1. 先修论文复现相关的 P0：`seed`、`device`、`eval checkpoint`、`deploy stage 默认配置`。
2. 再补 paper eval runner 和论文 ablation 配置。
3. 然后把 `scripts/` 重构掉，再处理 `hide_and_seek` 三个环境的公共基类。
4. 接着补依赖和最小测试。
5. 最后统一清理文档与默认配置。

## 5. 本次已完成的核对

- 已确认仓库主线是“Isaac Sim + Hydra + MAPPO + 三个追逃环境变体”。
- 已确认 `scripts/` 和 `omni_drones/envs/hide_and_seek/` 是当前主要维护成本来源。
- 已运行 `python3 -m compileall scripts omni_drones`，纯语法检查通过。
- 当前环境没有 `python` 命令，只有 `python3`；README/脚本说明里最好明确环境前提。
- 没有发现项目自己的自动化测试入口。
- 已根据论文核对了 OPEN 的三大模块：EPN、AEG、Two-stage reward refinement。
- 已确认 EPN 和 AEG 的主体实现是存在的，当前最突出的论文对齐问题集中在部署阶段默认配置和评估复现流程。

## 6. 后续维护方式

这份文档建议持续更新，规则如下：

- 每次确认新的结构问题、高优先级 bug 或论文对齐问题，直接补到对应的 P0 / P1 / P2。
- 如果某项已修复，就在条目后面标记修复日期，或把它移动到“已完成”区块。
- 如果后面我继续帮你改代码，我会默认先对照这份清单更新状态，不再重复从零梳理。

## 7. 可作为下一步的具体改动包

如果下一轮直接开始改代码，我建议拆成三个小包：

1. `seed + device + eval checkpoint + deploy stage` 论文复现修复包。
2. `paper eval runner + ablation configs` 论文实验复现包。
3. `scripts/` 训练入口去重包。
4. `hide_and_seek` 环境基类重构包。
