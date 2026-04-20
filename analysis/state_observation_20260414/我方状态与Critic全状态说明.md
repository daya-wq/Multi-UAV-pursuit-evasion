# 我方状态与 Critic 全状态说明

本文档只说明当前代码里真正生效的状态，不讲旧版本。

对应代码：

- [hideandseek.py](/data/uavlab/multi-uav-pursuit2/omni_drones/envs/hide_and_seek/hideandseek.py)
- [HideAndSeek.yaml](/data/uavlab/multi-uav-pursuit2/cfg/task/HideAndSeek.yaml)

## 1. 总体结构

当前每架追捕无人机送给 actor 的观测分成三块：

```text
state_self   : [1, 23]
state_others : [2, 9]
cooperation  : [1, 24]
```

critic 用的是集中式状态：

```text
state_drones : [3, 47]
```

这里默认有 3 架追捕无人机，所以：

- `state_others` 里有 2 个队友
- `state_drones` 里有 3 行，每一行对应一架追捕机

## 2. Actor 的 `state_self`

`state_self` 表示“这架无人机自己的状态 + 它当前被分配到的角色目标点”。

拼接顺序如下：

```text
state_self = [
  self_pos_world          3
  self_vel_world          3
  self_body_rates         3
  self_quat_world         4
  assigned_waypoint_world 3
  close_waypoint_world    3
  close_triggered         1
  role_onehot             3
]
```

总维度：

```text
3 + 3 + 3 + 4 + 3 + 3 + 1 + 3 = 23
```

通俗解释：

- `self_pos_world`：我在世界坐标系的位置
- `self_vel_world`：我在世界坐标系的线速度
- `self_body_rates`：我机体系下的角速度
- `self_quat_world`：我的姿态四元数
- `assigned_waypoint_world`：专家策略给我分配的正常角色锚点
- `close_waypoint_world`：进入 close 收口后，我应该去的收口点
- `close_triggered`：我现在是否已经切到 close 模式，`0/1`
- `role_onehot`：我当前是后追捕者、前左拦截者还是前右拦截者

## 3. Actor 的 `state_others`

`state_others` 表示“另外两架队友相对我是什么状态”。

每个队友的特征是：

```text
other_j = [
  other_pos_world - self_pos_world  3
  other_vel_world - self_vel_world  3
  other_role_onehot                 3
]
```

总维度：

```text
3 + 3 + 3 = 9
```

因为当前是 3 架追捕机，所以：

```text
state_others : [2, 9]
```

通俗解释：

- 我能看到两个队友分别在我哪边
- 我能看到两个队友相对我是更快还是更慢
- 我能看到两个队友分别扮演什么角色

## 4. Actor 的 `cooperation`

`cooperation` 表示“全队共享的目标运动和围堵几何信息”。

拼接顺序如下：

```text
cooperation = [
  target_vel_world     3
  target_pred_world    3 * 5
  forward_dir_world    3
  lateral_world        3
]
```

总维度：

```text
3 + 15 + 3 + 3 = 24
```

其中：

- `target_vel_world`：目标当前世界系速度
- `target_pred_world`：目标未来 5 个预测点，每个点 3 维
- `forward_dir_world`：专家策略当前认为的目标前向/逃跑主方向
- `lateral_world`：和 `forward_dir_world` 垂直的侧向单位向量

通俗解释：

- `target_vel_world` 告诉网络目标现在往哪跑
- `target_pred_world` 告诉网络目标接下来 5 步大概率去哪
- `forward_dir_world` 告诉网络“拦截三角阵”的前后方向
- `lateral_world` 告诉网络“左右两侧拦截位”应该往哪分开

## 5. 角色编码的含义

当前 `role_onehot` 用的是 3 维 one-hot：

```text
[1, 0, 0] = 后追捕者 rear
[0, 1, 0] = 前左拦截者 front-left
[0, 0, 1] = 前右拦截者 front-right
```

注意：角色不是固定绑死在 `drone0/1/2` 上的。

当前代码每一步都会根据 expert2 的三个锚点，重新做一次最小总代价分配，让三架机自动决定谁当前更适合当 rear、谁更适合当前左前/右前。

## 6. Critic 的 `state_drones`

critic 当前吃的是集中式状态：

```text
state_drones : [3, 47]
```

每一行的拼接方式是：

```text
state_drones[i] = [
  state_self_i   23
  cooperation_i  24
]
```

总维度：

```text
23 + 24 = 47
```

也就是说，critic 对每一架追捕机都能看到：

- 这架机自己的位置、速度、角速度、姿态
- 这架机当前的 normal waypoint / close waypoint
- 这架机当前角色
- 目标当前速度
- 目标未来 5 个预测点
- 当前 forward / lateral 几何方向

然后 critic 一次同时看到三架机的这三行。

## 7. 这版状态和你要求的对应关系

你提的状态定义已经对应进去了：

```text
state_self:
  self_pos_world
  self_vel_world
  self_body_rates
  四元数
  assigned_waypoint_world
  close_waypoint_world
  close_triggered
  role_onehot

state_others:
  other_pos_world - self_pos_world
  other_vel_world - self_vel_world
  other_role_onehot

cooperation:
  目标速度
  目标预测 3*5
  forward_dir_world
  lateral_world
```

## 8. 当前实际维度

当前代码跑起来以后，实际 shape 已验证为：

```text
state_self   : [num_envs, 3, 1, 23]
state_others : [num_envs, 3, 2, 9]
cooperation  : [num_envs, 3, 1, 24]
state_drones : [num_envs, 3, 47]
```

## 9. 一个重要提醒

这次改动已经改变了 actor/critic 的输入维度，所以旧 checkpoint 不能直接严格加载。

如果后面你想继续用旧的 BC 或 RL 权重，有两个办法：

- 重新从头训练
- 做兼容加载：把旧输入层权重拷过去，新加的输入维度初始化为 0 或很小的值

如果你要，我下一步可以直接帮你把“旧 checkpoint 兼容加载到新版状态输入”也补上。
