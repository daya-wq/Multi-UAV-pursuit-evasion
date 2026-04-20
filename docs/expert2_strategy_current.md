# Expert2 策略说明（当前工作区）

本文档整理当前 `expert2` 专家策略的实际代码逻辑和主要参数。策略实现主要来自 `scripts/expert_strategy_test.py` 的共享专家控制器，以及 `scripts/expert_isaac_eval.py` 的 Isaac Sim batched 评测/采集版本；`scripts/run_expert2_actor_imitation_pipeline.sh` 是当前 Expert2 模仿学习流水线入口。

## 1. 当前默认组合

Expert2 的几何意图是 **双前拦截 + 单后追捕**：

- 1 架后追捕者，分配到 anchor 0，也就是 `rear/chaser`。
- 2 架前拦截者，分配到 anchor 1/2，也就是 `front interceptors`。
- 默认关闭 `goal_mode`，默认开启 `close_mode`，默认关闭 `rush_mode`。
- 默认前拦截布局是 `symmetric`。

当前默认开关来自 `apply_strategy_defaults()` 和评测/采集脚本：

| 参数 | 当前 Expert2 默认值 | 含义 |
| --- | --- | --- |
| `strategy_variant` | `expert2` | 使用 Expert2 几何策略 |
| `forward_dir_mode` | `default` | 方向混合模式；但在默认 `goal_mode=false` 时会被 Expert2 强制改成纯运动方向 |
| `enable_goal_mode` | `false` | 关闭目标守区方向干预，不启用 goal emergency |
| `enable_close_mode` | `true` | 开启近距收拢逻辑 |
| `enable_rush_mode` | `false` | 关闭最近无人机直接冲刺目标逻辑 |
| `expert2_front_layout` | `symmetric` | 两个前拦截点左右对称 |

注意：`scripts/expert_isaac_eval.sh` 自身默认 `STRATEGY_VARIANT=baseline`，只有显式设置 `STRATEGY_VARIANT=expert2` 才会走 Expert2。`scripts/run_expert2_actor_imitation_pipeline.sh` 会固定设置 `STRATEGY_VARIANT=expert2`，并使用上表 Expert2 默认组合。

## 2. 目标预测

策略先得到 `target_pos_pred`，也就是目标下一步或短期位置预测：

| `pred_mode` | 行为 |
| --- | --- |
| `noise` | 使用真实下一步目标位置 `target_next_pos`，再沿随机方向加 `0.0` 到 `0.5 m` 噪声 |
| `tp_net` | 用 TP-Net 历史序列预测；历史长度 `history_step=10`，预测步数配置 `future_predcition_step=5`，实际取输出第 0 个未来点 |
| `oracle_pos` | 直接使用目标当前位置 |
| `oracle_next` | 直接使用真实下一步目标位置 |

评测脚本 `scripts/expert_isaac_eval.sh` 默认 `pred_mode=tp_net`，找不到 `tp_only_*.pt` 时回退到 `noise`。当前 Expert2 模仿学习流水线默认 `PRED_MODE=noise`。

## 3. 阵型构造

先构造预测拦截种子点：

```text
goal_vec = goal_region_center - target_pos
goal_dist = ||goal_vec||
lookahead_time = 0.18, if goal_dist > 1.4
lookahead_time = 0.10, otherwise
intercept_seed = target_pos_pred + target_vel * lookahead_time
```

方向构造：

- `motion_dir` 优先取目标速度方向。
- 如果目标速度太小，则用 `intercept_seed - target_pos`。
- 如果仍然太小，则回退到 `goal_dir`。
- 当前默认 `expert2 + goal_mode=false` 时，`forward_dir = motion_dir`，不混合守区方向。
- 如果手动开启 `goal_mode=true`，并且 `forward_dir_mode=default`，Expert2 使用 `goal_weight=0.92`（紧急）或 `0.70`（非紧急）混合 `goal_dir` 和 `motion_dir`。
- 如果手动开启 `goal_mode=true`，并且 `forward_dir_mode=motion_biased`，Expert2 使用 `goal_weight=0.80`（紧急）或 `0.45`（非紧急）。
- `goal_emergency = goal_dist < 1.10`，但默认 `goal_mode=false` 时强制为 false。

横向方向：

```text
lateral = normalize(cross([0, 0, 1], forward_dir))
```

如果叉乘退化，则回退到 `[1, 0, 0]`。

### 3.1 symmetric 布局（默认）

```text
rear  = intercept_seed - 0.30 * forward_dir
left  = intercept_seed + 0.40 * lateral
right = intercept_seed - 0.40 * lateral
anchors = [rear, left, right]
```

其中 `rear` 是后追捕者，`left/right` 是两个前拦截者。因为 `intercept_seed` 本身已经是目标短期预测位置，两个前拦截点会围绕预测位置左右展开。

### 3.2 staggered 布局（可选）

```text
rear  = intercept_seed - 0.30 * forward_dir
left  = intercept_seed
right = intercept_seed + 0.20 * forward_dir + 0.40 * lateral
anchors = [rear, left, right]
```

该布局把一个前拦截者放到预测点，另一个放到更靠前且侧偏的位置。最近评测中它的综合效果低于默认 `symmetric`。

### 3.3 角色分配

每一步对 3 架无人机和 3 个 anchor 做全排列匹配，选择总代价最小的分配。代价包含：

```text
cost = sum(distance(drone_i, assigned_anchor_i)) + 0.20 * changed_assignment_count
```

`0.20` 是角色切换惩罚，用于减少左右/前后角色频繁抖动。

## 4. 正常飞行控制律

对每架无人机，先根据分配结果判断角色：

- `anchor_idx == 0`：后追捕者 `chaser`。
- `anchor_idx != 0`：前拦截者 `front_interceptor`。

Expert2 正常模式参数：

| 角色 | `forward_bias` | `kp` |
| --- | ---: | ---: |
| 后追捕者 | `-0.08` | `1.70` |
| 前拦截者 | `+0.10` | `1.80` |

速度指令：

```text
feedforward = target_vel + forward_bias * forward_dir
v_des = kp * (waypoint - drone_pos) + 0.85 * feedforward - 0.26 * drone_vel
```

高度控制：

```text
desired_height = target_pos_pred.z + (0.10 if close_mode else 0.05)
desired_height = clamp(desired_height, 1.2, max_height - 0.4)
v_des.z = 0.90 * (desired_height - drone_z) - 0.45 * drone_vz
```

在当前 `max_height=5.0` 下，高度 clamp 是 `[1.2, 4.6]`。

最终速度会限幅到 `v_drone`。当前评测/采集脚本通常把 `v_drone_test` 覆盖为 `1.5 m/s`，虽然 `cfg/task/HideAndSeek.yaml` 里的任务默认值是 `1.0 m/s`。

## 5. Close Mode 近距收拢

当前 Expert2 默认开启 `close_mode`。触发条件：

```text
chaser:            dist_to_waypoint < 0.45 or dist_to_target < 0.95
front_interceptor: dist_to_waypoint < 0.45 or dist_to_target < 0.75
```

收拢点：

```text
close_wp_chaser      = target_pos_pred - 0.10 * forward_dir
close_wp_interceptor = target_pos_pred + 0.03 * forward_dir + 0.24 * side_axis
```

其中：

```text
side_axis = normalize(project_to_plane(waypoint - target_pos, forward_dir))
```

向心增益：

| 角色 | `inward_gain` |
| --- | ---: |
| 后追捕者 | `0.16` |
| 前拦截者 | `0.05` |
| rush 最近无人机（仅 rush 开启时） | `0.35` |

Close velocity：

```text
inward = normalize(target_pos_pred - drone_pos)
v_close =
    2.05 * (close_wp - drone_pos)
  + 0.92 * target_vel
  + inward_gain * inward * v_drone
  - 0.34 * drone_vel

v_close.z = 0.90 * (target_pos_pred.z - drone_z) - 0.55 * drone_vz
```

当 `close_mode=true` 时，`v_des` 被替换为 `v_close`。

## 6. Rush Mode（默认关闭）

如果手动开启 `enable_rush_mode=true`：

```text
rush_mode = min_dist_to_target < catch_radius * 1.8
```

当前 `catch_radius=0.4`，所以阈值是 `0.72 m`。触发后，离目标最近的无人机会被标记为 `rush_idx`，在 close 收拢里直接把 `close_wp` 设为 `target_pos_pred`，并把 `inward_gain` 设为 `0.35`。当前默认关闭它，因为最近消融结果显示 rush 单独开启或与 goal 一起开启都没有默认 close-only 稳。

## 7. 无人机间避碰和低空保护

基础避碰：

```text
safe_dist = 4.0 * collision_radius = 0.28 m
repulse_gain = 0.5
```

Expert2 在 close mode 下会加强避碰：

```text
safe_dist = 0.36 m
repulse_gain = 0.65
closing_boost = 1.0 + 0.50 * clamp(closing_speed / v_drone, 0.0, 1.0)
repulse_mag = v_drone * repulse_gain * (1 - d / safe_dist) * closing_boost
```

低空保护：

```text
low_altitude_guard_z = 0.40 m
if drone_z < 0.40:
    v_des.z = max(v_des.z, 3.0 * (0.40 - drone_z))
```

修正后再次按 `v_drone` 限幅。

## 8. 速度到 PIDRate action 的转换

Expert2 先输出世界系期望速度，再转成 PIDRate action。转换流程：

```text
desired_vel = clip_norm(desired_vel, v_drone)
vel_error = desired_vel - current_vel
acc_cmd = 3.2 * vel_error
acc_cmd.z = clamp(acc_cmd.z, -3.5, 3.5)
||acc_cmd.xy|| <= max(0.1, (gravity + acc_cmd.z) * tan(25 deg))
thrust_world = acc_cmd + gravity * [0, 0, 1]
t = hover_thrust_ratio * ||thrust_world|| / gravity
t = clamp(t, min_thrust_ratio, max_thrust_ratio)
```

姿态/角速度：

```text
z_des = normalize(thrust_world)
x_ref = project current body-x onto plane normal to z_des
x_ref = 0.75 * x_ref + 0.25 * heading_hint_projected, if valid
e_rot = 0.5 * (cross(x_cur, x_des) + cross(y_cur, y_des) + cross(z_cur, z_des))
omega_world = 2.0 * e_rot
omega_body = rotate_inverse(current_quat, omega_world)
||omega_body|| <= 0.65 * max_body_rate_rad_s
```

PIDRate 归一化输出：

```text
rate_norm = clamp(omega_body / max_body_rate_rad_s, -0.999, 0.999)
thrust_norm = clamp(2.0 * t - 1.0, -0.999, 0.999)
action = [rate_norm_x, rate_norm_y, rate_norm_z, thrust_norm]
```

当前 `cfg/algo/mappo.yaml` 里 actor 使用 `tanh: true`，`PIDRateController` 以 `actor_has_tanh=True` 接收已经归一化到 `[-1, 1]` 的 action。因此专家采集的数据标记为 `action_label_type = "pidrate_normalized"`，BC/DAgger 训练时直接使用这个归一化 action 标签，不再做 `atanh`。

## 9. 主要参数总表

### 9.1 环境和目标参数

| 参数 | 当前值 | 来源/说明 |
| --- | ---: | --- |
| `num_agents` | `3` | 3 架追捕无人机 |
| `arena_size` | `2.5` | 平面场地半径/尺度 |
| `max_height` | `5.0` | 最大高度 |
| `sim.dt` | `0.01 s` | Isaac Sim 运行时；简化测试器默认 `0.02 s` |
| `episode_length` | `1000` | 评测/采集脚本默认覆盖；任务 YAML 里默认是 `800` |
| `v_drone_test` | `1.5 m/s` | 评测/采集脚本默认 |
| `v_prey_test` | `1.5 m/s` | 评测脚本默认 |
| `v_prey_schedule` | `1.2,1.35,1.5,1.65` | 当前 Expert2 模仿学习流水线默认 |
| `catch_radius` | `0.4 m` | 抓捕半径 |
| `collision_radius` | `0.07 m` | 单机碰撞半径 |
| `goal_region_center` | `[2.0, 0.0, 2.5]` | 守区中心 |
| `goal_region_radius` | `0.5 m` | 守区半径 |
| `goal_region_height` | `2.0 m` | 守区高度 |
| `target_accel_limit` | `2.0` | 目标 APF 加速度限幅 |
| `target_velocity_damping` | `0.25` | 目标速度阻尼 |
| `target_command_tau` | `0.15` | 目标加速度命令时间常数 |
| `target_physics_max_velocity` | `5.0` | 目标物理速度上限 |
| `target_min_z` | `0.2` | 目标最低高度保护 |

### 9.2 Expert2 几何参数

| 参数 | 当前值 | 含义 |
| --- | ---: | --- |
| `expert2_rear_back` | `0.30` | 后追捕 anchor 在 `forward_dir` 反方向后退距离 |
| `expert2_side_width` | `0.40` | symmetric 两个前拦截 anchor 的左右侧偏 |
| `expert2_front_layout` | `symmetric` | 默认双侧对称布局 |
| `expert2_front_lead_forward` | `0.20` | staggered 中前侧拦截点的前向偏移 |
| `expert2_front_lead_side` | `0.40` | staggered 中前侧拦截点的侧向偏移 |
| `lookahead_time_far` | `0.18` | `goal_dist > 1.4` 时使用 |
| `lookahead_time_near` | `0.10` | `goal_dist <= 1.4` 时使用 |
| `assignment_switch_penalty` | `0.20` | 角色切换惩罚 |
| `goal_emergency_dist` | `1.10` | 仅 `goal_mode=true` 时生效 |
| `trap_mode_dist` | `1.55` | Expert2 构造后会把 `trap_mode` 清为 false，仅 baseline 使用 |

### 9.3 Expert2 close/避碰参数

| 参数 | 当前值 | 含义 |
| --- | ---: | --- |
| `expert2_chaser_close_back` | `0.10` | 后追捕者 close 点在目标后方的距离 |
| `expert2_front_close_forward` | `0.03` | 前拦截者 close 点前向偏移 |
| `expert2_front_close_side` | `0.24` | 前拦截者 close 点侧向偏移 |
| `expert2_chaser_inward_gain` | `0.16` | 后追捕者向心增益 |
| `expert2_front_inward_gain` | `0.05` | 前拦截者向心增益 |
| `expert2_chaser_close_target_threshold` | `0.95` | 后追捕者对目标近距阈值 |
| `expert2_front_close_target_threshold` | `0.75` | 前拦截者对目标近距阈值 |
| `expert2_close_safe_dist` | `0.36` | close mode 下无人机间安全距离 |
| `expert2_close_repulse_gain` | `0.65` | close mode 下斥力增益 |
| `expert2_close_repulse_closing_boost` | `0.50` | 相向接近时额外斥力系数 |
| `low_altitude_guard_z` | `0.40` | 低空保护阈值 |

### 9.4 控制器参数

| 参数 | 当前值 | 含义 |
| --- | ---: | --- |
| normal `kp` chaser | `1.70` | 后追捕者 waypoint 比例增益 |
| normal `kp` front | `1.80` | 前拦截者 waypoint 比例增益 |
| normal feedforward coef | `0.85` | 目标速度前馈系数 |
| normal velocity damping | `0.26` | 当前速度阻尼 |
| close waypoint coef | `2.05` | close 点比例项 |
| close target velocity coef | `0.92` | close 模式目标速度前馈 |
| close velocity damping | `0.34` | close 模式速度阻尼 |
| height normal kp | `0.90` | 高度跟随比例项 |
| height normal damping | `0.45` | 高度速度阻尼 |
| height close damping | `0.55` | close 模式高度速度阻尼 |
| velocity-to-acc gain | `3.2` | 速度误差转加速度增益 |
| z acceleration clamp | `[-3.5, 3.5]` | 垂直加速度限幅 |
| max tilt angle | `25 deg` | 水平加速度耦合限幅 |
| attitude error gain | `2.0` | `omega_world = 2.0 * e_rot` |
| omega norm cap | `0.65 * max_body_rate_rad_s` | 角速度范数限幅 |
| `target_clip` | 通常 `0.8` | `max_body_rate_rad_s = radians(180 * target_clip)` |
| `max_body_rate_rad_s` | 通常 `2.513 rad/s` | `target_clip=0.8` 时 |
| `min_thrust_ratio` | 通常 `0.0` | PIDRate 控制器读取 |
| `max_thrust_ratio` | 通常 `0.9` | PIDRate 控制器读取 |
| `hover_thrust_ratio` | 运行时计算，约 `0.52` | 由无人机重力/最大推力计算 |

## 10. 最近评测结果

最近一次策略套件评测目录：

```text
analysis/expert_strategy_suite/20260411_003646/
```

统一设置：Isaac Sim 6-DoF，`pred_mode=noise`，`v_prey=1.5`，512 并行环境，1024 episodes。

| 策略 | 捕获率 | 进守区失败 | 落地 | 超时 | 任意碰撞 | 机间碰撞 | 平均捕获步 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_best` | `77.7%` | `8.8%` | `0.0%` | `13.5%` | `65.1%` | `28.7%` | `355.2` |
| `expert2_both_on` | `74.7%` | `8.2%` | `0.0%` | `17.1%` | `61.3%` | `26.6%` | `300.5` |
| `expert2_none` | `72.9%` | `10.1%` | `0.0%` | `17.0%` | `60.4%` | `16.5%` | `310.3` |
| `expert2_close_only` | `75.9%` | `8.8%` | `0.0%` | `15.3%` | `61.1%` | `24.8%` | `307.2` |
| `expert2_rush_only` | `72.9%` | `10.3%` | `0.0%` | `16.8%` | `60.4%` | `16.7%` | `312.4` |
| `expert2_goal_off_close_only` | `77.4%` | `7.5%` | `0.0%` | `15.0%` | `61.1%` | `24.8%` | `325.2` |
| `expert2_goal_off_close_only_staggered` | `73.6%` | `7.2%` | `0.0%` | `19.1%` | `63.9%` | `32.5%` | `303.7` |

注意：这批 suite 里 `expert2_none`、`expert2_close_only`、`expert2_rush_only`、`expert2_both_on` 都是 `goal_mode=true` 的消融名；当前默认 Expert2 组合对应额外标了 `goal_off` 的 `expert2_goal_off_close_only`。

当前默认 Expert2 组合等价于 `expert2_goal_off_close_only`：`goal=false`，`close=true`，`rush=false`，`front_layout=symmetric`。

## 11. 常用调用

只评测 Expert2：

```bash
STRATEGY_VARIANT=expert2 \
ENABLE_GOAL_MODE=false \
ENABLE_CLOSE_MODE=true \
ENABLE_RUSH_MODE=false \
EXPERT2_FRONT_LAYOUT=symmetric \
bash scripts/expert_isaac_eval.sh noise 1.5 0 1024
```

运行当前 Expert2 采集 + BC + DAgger 流水线：

```bash
bash scripts/run_expert2_actor_imitation_pipeline.sh
```

常用覆盖项：

```bash
PRED_MODE=noise \
V_DRONE_TEST=1.5 \
V_PREY_SCHEDULE=1.2,1.35,1.5,1.65 \
GENERIC_BATCH_ENVS=1024 \
NUM_WAVES=50 \
DAGGER_WAVES=50 \
bash scripts/run_expert2_actor_imitation_pipeline.sh
```
