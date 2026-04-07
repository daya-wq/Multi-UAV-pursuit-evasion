# Goal-Defense Task Redesign

## 1. 目标

本轮改动将原来的“纯追逃”任务，重构为一个更明确的“守区拦截”任务：

- 目标无人机的任务：进入右侧守区。
- 我方 pursuer 的任务：阻止目标进入守区，并在进入前完成抓捕。

这次重构的核心目的不是继续微调旧奖励，而是先把任务定义理顺：

- target 必须有明确目标，而不是只做抽象逃逸。
- target 必须有连续、受约束的运动学，而不是直接写速度的球体。
- 守区的几何必须和总场地几何一致。

## 2. 本次改动摘要

### 2.1 Target 改成有界二阶积分器

之前的 target 是 `DynamicSphere + set_velocities()`，本质上是一阶速度直接赋值：

- 速度可以瞬时改向
- 没有加速度约束
- 没有控制滞后

现在改成：

- target 本体仍然是 `DynamicSphere`
- 高层规则先生成目标加速度方向
- 再通过有界二阶积分器推进 target 运动

高层目标不是 RL，而是一个规则控制器：

- 朝守区产生吸引
- 受 pursuer 产生排斥
- 受障碍和场地边界产生排斥

也就是说，**target 仍然是规则驱动**，但它现在不再瞬时改速度，而是按下面的约束演化：

- `target_accel_limit`
  - 限制 target 最大加速度
- `target_command_tau`
  - 给期望加速度指令加入一阶滞后
- `target_velocity_damping`
  - 给速度加入线性阻尼，避免无限滑行
- `v_prey`
  - 作为 target 的速度上限
- `target_physics_max_velocity`
  - 作为 Isaac 刚体层的物理安全上限，用于防止异常数值

这版 target 的定位是：

- 比“直接写速度”真实得多
- 比“完整四旋翼 6DoF prey”更容易训练
- 更适合作为后续真机高层逃逸器的参考模型

### 2.2 场地扩大

当前环境原生是：

- `xy` 平面为圆形场地
- `z` 方向为有限高度

所以本次不是把环境改成真正的立方体边界，而是把它扩成：

- 圆形场地直径 `5m`
- 高度 `5m`

对应配置：

- `arena_size: 2.5`
- `max_height: 5.0`

这样做的原因是：

- 兼容当前环境已有的边界碰撞逻辑
- 不需要重写整套“圆边界”代码成“方盒子边界”
- 先解决“空间太小，动作稍激进就撞地/撞边”的问题

## 3. 新的守区定义

### 3.1 守区改成圆柱形

守区已经从之前的矩形盒改成了**圆柱形守区**。

配置含义：

- `goal_region_radius: 0.5`
- `goal_region_height: 2.0`

因此守区是一个：

- 半径 `0.5m`
- 高度 `2.0m`

的有限圆柱。

### 3.2 守区贴边

由于总场地是圆形，守区现在默认放在右侧并与外边界相切：

- 守区中心 `x = arena_size - goal_region_radius`
- `y = 0`
- `z = max_height / 2`

这样做有三个直接好处：

- 守区几何和总场地几何一致，解释干净
- 守区确实“贴边”，而不是一个悬在内部的人工盒子
- 不会出现“矩形角落伸出圆形场地外”的几何矛盾

## 4. 奖励与终止逻辑

### 4.1 Pursuer 成功条件

- 在 target 进入守区之前抓住它

### 4.2 Pursuer 失败条件

- target 进入守区
- 任意 pursuer 坠地
- episode 超时

### 4.3 当前主要奖励

- `distance_reward`
  - pursuer 接近 target 的进度奖励
- `goal_progress_reward`
  - 如果 target 被逼得离守区更远，则 pursuer 得正奖励
  - 如果 target 更接近守区，则 pursuer 得负奖励
- `goal_penalty`
  - target 一旦进入守区，pursuer 立即吃终止罚分
- `catch_reward`
  - 成功抓捕给大额正奖励
- `collision / speed / landed / smoothness`
  - 继续作为控制稳定性约束

### 4.4 当前被刻意弱化/关闭的项

为了避免新任务刚接入就被旧奖励拓扑拖偏，这一版先做了任务简化：

- `distance_predicted_coef: 0.0`
- `encircle_coef: 0.0`
- `intercept_coef: 0.3`

含义是：

- 先让策略学会“防守区 + 抓捕”这个主任务
- 暂时不强推复杂围捕或未来点 shaping

## 5. 初始场景

当前默认场景切到了：

- `scenario_flag: goal_defense`

固定站位逻辑是：

- pursuer 在右半侧靠近守区部署
- target 在左侧出发
- 初始高度默认对齐守区中部高度

这样做的原因是：

- 它确实形成了“入侵方 vs 防守方”的结构
- 不再像旧场景那样，一开始双方已经极近，导致训练退化成最后几厘米的控制问题

## 6. 为什么圆柱守区比矩形守区更合理

在当前环境里，圆柱守区更合理，原因非常直接：

1. 总场地本身就是圆形 `xy` 边界。
2. 矩形守区贴边时，角点会天然和圆边界发生冲突。
3. 圆柱守区可以自然地与圆场地做“内切/相切”关系。
4. 对 target 来说，接近守区的几何距离也更平滑，不会出现矩形角点导致的方向突变。

## 7. 当前已改文件

- `omni_drones/envs/hide_and_seek/hideandseek.py`
  - target 二阶积分器运动学
  - 圆柱守区判定
  - goal-defense reward/done
- `cfg/task/HideAndSeek.yaml`
  - 新场地参数
  - 新守区参数
  - target 运动学参数
  - 新任务默认场景

## 8. 当前未做的事情

这次改动只落在训练环境 `hideandseek.py`：

- 还没有同步到 `hideandseek_deploy.py`
- 还没有同步到 `hideandseek_envgen.py`
- 还没有给守区加可视化 debug draw

所以如果后续要做：

- deploy
- env generator
- 视频中直接显示守区

还需要再补一轮同步。

## 9. 下一步验证建议

建议按这个顺序验证：

1. 先跑 `1 env` 短 rollout
   - 确认 target 不会自己失控撞地
   - 确认 target 确实会朝守区移动
   - 确认进入守区会触发终止
2. 再跑 `scratch` 训练
   - 因为 prey 动力学、任务终止条件、奖励拓扑都变了
   - 旧 checkpoint 不应继续沿用
3. 训练时重点观察：
   - `goal_progress_reward`
   - `goal_penalty`
   - `catch_reward`
   - `any_landed`
   - `collision_floor`

## 10. 结论

这次改动的核心不是“调参”，而是把任务从一个容易学歪的纯追逃问题，改成一个：

- 目标明确
- 几何一致
- 运动学受约束
- 多机协作更自然

的守区拦截任务。

圆柱形贴边守区是这个重构里必要的一步，不是装饰性修改。
