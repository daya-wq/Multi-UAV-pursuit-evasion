"""
expert_strategy_test.py
========================
独立专家策略测试脚本 —— 无需 Isaac Sim，不干扰任何正在运行的训练。

专家策略输出：
  - t      : 归一化推力标量  (0 ~ 1)
  - omega  : 角速度向量      [omega_x, omega_y, omega_z]  (rad/s)

合成速度超过 v_max=1.5 m/s 时在速度层面截断。

目标预测模式（--pred_mode 参数）：
  "noise"   : 直接读取目标下一帧位置，叠加上限 0.5 的均匀球面噪声
  "tp_net"  : 载入 tp_only_1690959872.pt，用 TP_net 预测下一时刻目标位置

测试配置与现有 HideAndSeek 环境完全一致（见 cfg/task/HideAndSeek.yaml）。
运行 50 轮，统计：
  1. 成功率（追逐者捕获目标）
  2. 捕获步数分布
  3. 目标到达守区的概率
并生成图表保存到 figures/expert_test_*.png。

用法：
  python scripts/expert_strategy_test.py
  python scripts/expert_strategy_test.py --pred_mode tp_net
  python scripts/expert_strategy_test.py --n_episodes 50 --seed 42
"""

import argparse
import math
import os
import sys
import collections
import itertools
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────
# 0.  TP_net (从 mappo.py 中独立摘出，保持权重兼容)
# ─────────────────────────────────────────────────────────────
class TP_net(nn.Module):
    def __init__(self, input_dim, output_dim, future_predcition_step, window_step):
        super().__init__()
        self.hidden_dim = 64
        self.num_layers = 1
        self.future_predcition_step = future_predcition_step
        self.window_step = window_step
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return torch.tanh(self.fc(out[:, -1, :]))


# ─────────────────────────────────────────────────────────────
# 1.  环境参数（与 HideAndSeek.yaml / hideandseek.py 保持一致）
# ─────────────────────────────────────────────────────────────
class EnvCfg:
    # layout
    arena_size: float = 2.5
    max_height: float = 5.0
    num_agents: int   = 3
    max_episode_length: int = 1000
    dt: float = 0.02             # 50 Hz sim step
    gravity: float = 9.81

    # agents
    v_drone: float = 1.5
    catch_radius: float = 0.4
    collision_radius: float = 0.07
    target_clip: float = 0.8
    min_thrust_ratio: float = 0.0
    max_thrust_ratio: float = 0.9
    hover_thrust_ratio: float = 0.52
    max_body_rate_rad_s: float = math.radians(180.0 * 0.8)

    # goal / defense zone
    goal_region_center = torch.tensor([2.0, 0.0, 2.5])
    goal_region_radius: float = 0.5
    goal_region_height: float = 2.0

    # target APF dynamics  (mirrors _pre_sim_step)
    v_prey: float = 1.5
    target_accel_limit: float = 2.0
    target_velocity_damping: float = 0.25
    target_command_tau: float = 0.15
    target_goal_attraction_coef: float = 1.0
    target_physics_max_velocity: float = 5.0

    # TP net (for "tp_net" mode)
    max_agents: int = 4
    history_step: int = 10
    future_predcition_step: int = 5
    window_step: int = 1

    # initial positions (use_eval=1 layout, matches _reset_idx)
    goal_z: float = 2.5
    # drones at fixed positions + small noise
    drone_init_pos = torch.tensor([
        [1.2000, 0.0000, 2.5],
        [1.5000, 0.6000, 2.5],
        [1.5000,-0.6000, 2.5],
    ])
    target_init_pos = torch.tensor([[-1.6000, 0.0000, 2.5]])

    pos_noise_xy: float = 0.15
    pos_noise_z: float  = 0.10


CFG = EnvCfg()


# ─────────────────────────────────────────────────────────────
# 2.  几何工具
# ─────────────────────────────────────────────────────────────
def safe_normalize(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + eps)

def clip_vector_norm(v: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = v.norm(dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / (norm + 1e-6), max=1.0)
    return v * scale


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_w = q[..., :1]
    q_vec = q[..., 1:]
    a = v * (2.0 * q_w.square() - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * (q_vec * v).sum(dim=-1, keepdim=True) * 2.0
    return a + b + c


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_w = q[..., :1]
    q_vec = q[..., 1:]
    a = v * (2.0 * q_w.square() - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * (q_vec * v).sum(dim=-1, keepdim=True) * 2.0
    return a - b + c


def safe_atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.atanh(x.clamp(-1.0 + eps, 1.0 - eps))


def project_to_plane(v: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    return v - (v * normal).sum(dim=-1, keepdim=True) * normal

def point_to_cylinder_distance(points, center, radius, height):
    radial = torch.norm(points[..., :2] - center[:2], dim=-1)
    rdelta = torch.relu(radial - radius)
    vdelta = torch.relu((points[..., 2] - center[2]).abs() - height / 2.0)
    return torch.sqrt(rdelta**2 + vdelta**2)

def point_in_cylinder(points, center, radius, height):
    radial = torch.norm(points[..., :2] - center[:2], dim=-1)
    vert_ok = (points[..., 2] - center[2]).abs() <= height / 2.0
    return (radial <= radius) & vert_ok

def closest_point_on_cylinder(points, center, radius, height):
    off_xy = points[..., :2] - center[:2]
    rdist  = off_xy.norm(dim=-1, keepdim=True)
    safe_r = torch.where(rdist > 1e-5, rdist, torch.ones_like(rdist))
    clamped = torch.minimum(rdist, torch.full_like(rdist, radius))
    cxy = center[:2] + off_xy / safe_r * clamped
    cz  = points[..., 2:3].clamp(center[2] - height / 2.0, center[2] + height / 2.0)
    return torch.cat([cxy, cz], dim=-1)


# ─────────────────────────────────────────────────────────────
# 3.  目标 APF 动力学（完全复制 _get_dummy_policy_prey）
# ─────────────────────────────────────────────────────────────
def apf_target_force(target_pos: torch.Tensor,
                     drone_pos: torch.Tensor,
                     cfg: EnvCfg) -> torch.Tensor:
    """
    target_pos : [1, 3]
    drone_pos  : [n_agents, 3]
    returns    : force [1, 3]
    """
    force = torch.zeros(1, 3)

    # goal attraction
    goal_center = cfg.goal_region_center.view(1, 3)
    goal_closest = closest_point_on_cylinder(
        target_pos, cfg.goal_region_center, cfg.goal_region_radius, cfg.goal_region_height
    )
    goal_offset = goal_closest - target_pos
    goal_dist   = goal_offset.norm(dim=-1, keepdim=True)
    force += cfg.target_goal_attraction_coef * goal_offset / (goal_dist + 1e-5)

    # pursuer repulsion
    target_rpos = drone_pos.unsqueeze(0) - target_pos.unsqueeze(1)  # [1, n, 3]
    dist_to_pursuer = target_rpos.norm(dim=-1, keepdim=True)        # [1, n, 1]
    force_dir = -target_rpos / (dist_to_pursuer + 1e-5)
    force_p   = force_dir * (1.0 / (dist_to_pursuer + 1e-5))
    force += force_p.sum(dim=1)

    # boundary repulsion
    xy_dist = target_pos[..., :2].norm(dim=-1, keepdim=True)
    xy_dir  = -target_pos[..., :2] / (xy_dist + 1e-5)
    out     = (target_pos[..., 0]**2 + target_pos[..., 1]**2) > cfg.arena_size**2

    force_bx = (out.float()  * xy_dir[..., 0] / 1e-5
              + (~out).float() * xy_dir[..., 0] / ((cfg.arena_size - xy_dist[..., 0]) + 1e-5))
    force_by = (out.float()  * xy_dir[..., 1] / 1e-5
              + (~out).float() * xy_dir[..., 1] / ((cfg.arena_size - xy_dist[..., 0]) + 1e-5))

    high = target_pos[..., 2] > cfg.max_height
    low  = target_pos[..., 2] < 0.0
    dz_h = (cfg.max_height - target_pos[..., 2]).clamp(min=1e-5)
    dz_l = (0.0 - target_pos[..., 2]).clamp(max=-1e-5)

    force_bz = (high.float() * (-1.0 / 1e-5)
              + (~high).float() * (-dz_h / (dz_h**2 + 1e-5))
              +  low.float()  * ( 1.0 / 1e-5)
              + (~low).float()  * (-dz_l / (dz_l**2 + 1e-5)))

    # force: [1,3], force_bx/y/z: scalar or [1] or [1,1] → reshape to [1,1] before +=
    force[..., 0] += force_bx.reshape(1)
    force[..., 1] += force_by.reshape(1)
    force[..., 2] += force_bz.reshape(1)

    return force.float()


def step_target(target_pos: torch.Tensor,
                target_vel: torch.Tensor,
                target_acc_cmd: torch.Tensor,
                drone_pos: torch.Tensor,
                cfg: EnvCfg):
    """Advance target one dt using APF + second-order integrator."""
    force = apf_target_force(target_pos, drone_pos, cfg)    # [1,3]
    desired_acc = cfg.target_accel_limit * force / (force.norm(dim=-1, keepdim=True) + 1e-5)
    accel_alpha = min(1.0, cfg.dt / max(cfg.target_command_tau, cfg.dt))
    target_acc_cmd = target_acc_cmd + accel_alpha * (desired_acc - target_acc_cmd)
    next_vel = target_vel + (target_acc_cmd - cfg.target_velocity_damping * target_vel) * cfg.dt
    next_vel = clip_vector_norm(next_vel, cfg.v_prey)
    next_pos = target_pos + next_vel * cfg.dt
    return next_pos, next_vel, target_acc_cmd


# ─────────────────────────────────────────────────────────────
# 4.  TP_net 预测器
# ─────────────────────────────────────────────────────────────
class TPNetPredictor:
    """
    维护 history_step 帧的状态历史，用 TP_net 预测下一时刻目标绝对位置。
    需要归一化约定与 hideandseek.py 中 _compute_state_and_obs() 完全一致。
    """
    def __init__(self, weight_path: str, cfg: EnvCfg, device: str = "cpu"):
        self.cfg    = cfg
        self.device = device
        self.net = TP_net(
            input_dim=1 + 3 + 3 + 3 * cfg.max_agents,
            output_dim=3 * cfg.future_predcition_step,
            future_predcition_step=cfg.future_predcition_step,
            window_step=cfg.window_step,
        ).to(device)
        state = torch.load(weight_path, map_location=device)
        self.net.load_state_dict(state)
        self.net.eval()
        self.history = collections.deque(maxlen=cfg.history_step)
        self.position_scale = torch.tensor(
            [cfg.arena_size, cfg.arena_size, cfg.max_height], device=device
        )

    def reset(self):
        self.history.clear()

    def _make_frame(self, step: int, target_pos, target_vel, drone_pos):
        """Build one frame tensor matching frame_state in hideandseek.py."""
        t_norm  = torch.tensor([step / max(self.cfg.max_episode_length, 1)],
                                device=self.device)
        tp_norm = (target_pos.squeeze(0) / self.position_scale).unsqueeze(0)  # [1,3]
        tv_norm = target_vel.squeeze(0)                                         # [3]
        # pad drone_pos to max_agents
        dp_norm = drone_pos / self.position_scale.view(1, 3)                   # [n,3]
        n = dp_norm.shape[0]
        if n < self.cfg.max_agents:
            pad = torch.full((self.cfg.max_agents - n, 3), -5.0, device=self.device)
            dp_norm = torch.cat([dp_norm, pad], dim=0)
        frame = torch.cat([t_norm,
                           tp_norm.reshape(-1),
                           tv_norm.reshape(-1),
                           dp_norm.reshape(-1)], dim=0)
        return frame  # [1 + 3 + 3 + max_agents*3]

    def predict_next_pos(self, step, target_pos, target_vel, drone_pos):
        """Returns predicted absolute position [3] for next 1 step."""
        frame = self._make_frame(step, target_pos, target_vel, drone_pos)
        self.history.append(frame)
        # fill history if not enough frames yet
        while len(self.history) < self.cfg.history_step:
            self.history.appendleft(frame)

        history_tensor = torch.stack(list(self.history), dim=0).unsqueeze(0)  # [1, H, D]
        with torch.no_grad():
            out = self.net(history_tensor)  # [1, future_step*3], tanh output
        out = out.reshape(self.cfg.future_predcition_step, 3)
        # denormalize (mirrors hideandseek.py lines 910-913)
        pred = out[0].clone()
        pred[:2] = pred[:2] * self.cfg.arena_size
        pred[2]  = (pred[2] + 1.0) / 2.0 * self.cfg.max_height
        return pred


# ─────────────────────────────────────────────────────────────
# 5.  专家策略核心：Blocker + Flanker 分层策略
# ─────────────────────────────────────────────────────────────
class ExpertPolicy:
    """
    输出 (t, omega) per agent：
      t     ∈ [0, 1]    归一化推力（用于表征合速度量级）
      omega ∈ R^3        角速度 (rad/s)，用于表征转向率

    实际速度通过 first-order filter 积分，限幅至 v_drone=1.0 m/s。
    """

    def __init__(self, cfg: EnvCfg, pred_mode: str = "noise",
                 tp_weight_path: str = "",
                 tp_device: str = "cpu"):
        self.cfg = cfg
        self.pred_mode = pred_mode
        self.prev_assignment = None
        if pred_mode == "tp_net":
            assert os.path.isfile(tp_weight_path), \
                f"TP weight not found: {tp_weight_path}"
            self.tp_predictor = TPNetPredictor(tp_weight_path, cfg, device=tp_device)
        else:
            self.tp_predictor = None

    def reset(self):
        self.prev_assignment = None
        if self.tp_predictor is not None:
            self.tp_predictor.reset()

    # ── 目标预测 ──────────────────────────────────────────────
    def predict_target(self, step: int,
                       target_pos: torch.Tensor,
                       target_vel: torch.Tensor,
                       drone_pos: torch.Tensor,
                       target_next_pos: torch.Tensor) -> torch.Tensor:
        """返回预测的目标下一时刻绝对位置 [3]。"""
        if self.pred_mode == "tp_net":
            return self.tp_predictor.predict_next_pos(
                step, target_pos, target_vel, drone_pos
            )
        else:  # "noise" mode
            noise_mag = torch.empty(1).uniform_(0.0, 0.5).item()
            direction = torch.randn(3, device=target_pos.device, dtype=target_pos.dtype)
            direction = direction / (direction.norm() + 1e-6)
            return target_next_pos.squeeze(0) + noise_mag * direction

    # ── 角色分配 ──────────────────────────────────────────────
    @staticmethod
    def assign_roles(drone_pos: torch.Tensor,
                     target_pos: torch.Tensor,
                     escape_dir: torch.Tensor,
                     n_agents: int) -> torch.Tensor:
        """
        返回 role_ids [n_agents]: 0 = Blocker, 1 = Flanker
        适合度：沿 escape_dir 投影最大（且侧向最小）的无人机做 Blocker
        """
        rel       = drone_pos - target_pos  # [n,3]
        ahead     = (rel * escape_dir).sum(dim=-1)
        lateral   = (rel - ahead.unsqueeze(-1) * escape_dir).norm(dim=-1)
        dist      = rel.norm(dim=-1)
        fitness   = ahead / (dist + 1e-6) * torch.exp(-lateral / 0.5)
        blocker   = fitness.argmax().item()
        roles     = torch.ones(n_agents, dtype=torch.long)
        roles[blocker] = 0
        return roles

    def _build_formation_targets(
        self,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor,
        target_pos_pred: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        goal_pos = self.cfg.goal_region_center.to(target_pos.device, target_pos.dtype)
        goal_vec = goal_pos - target_pos
        goal_dist = goal_vec.norm()
        goal_dir = safe_normalize(goal_vec.unsqueeze(0)).squeeze(0)

        lookahead_time = 0.18 if goal_dist > 1.4 else 0.10
        intercept_seed = target_pos_pred + target_vel * lookahead_time

        motion_hint = target_vel
        if motion_hint.norm() < 1e-4:
            motion_hint = intercept_seed - target_pos
        if motion_hint.norm() < 1e-4:
            motion_hint = goal_dir
        motion_dir = safe_normalize(motion_hint.unsqueeze(0)).squeeze(0)
        forward_dir = safe_normalize((0.7 * goal_dir + 0.3 * motion_dir).unsqueeze(0)).squeeze(0)

        up = torch.tensor([0.0, 0.0, 1.0], device=target_pos.device, dtype=target_pos.dtype)
        lateral = torch.linalg.cross(up, forward_dir)
        if lateral.norm() < 1e-5:
            lateral = torch.tensor([1.0, 0.0, 0.0], device=target_pos.device, dtype=target_pos.dtype)
        lateral = safe_normalize(lateral.unsqueeze(0)).squeeze(0)

        trap_mode = bool(goal_dist < 1.55)
        goal_emergency = bool(goal_dist < 1.10)
        front_cap = max(0.45, float(goal_dist.item()) - self.cfg.goal_region_radius - 0.12)
        # Only keep one true blocker in front; the other two should squeeze from
        # the sides or slightly behind. The previous "all three ahead" wall was
        # stable but often failed to reduce min target distance below catch_radius.
        base_front = min(front_cap, 1.00 if not trap_mode else 0.62)
        side_front = -0.10 if not trap_mode else min(front_cap, 0.32 if goal_emergency else 0.18)
        side_width = 0.64 if not trap_mode else (0.28 if goal_emergency else 0.34)

        center = intercept_seed + forward_dir * base_front
        left = intercept_seed + forward_dir * side_front + lateral * side_width
        right = intercept_seed + forward_dir * side_front - lateral * side_width
        anchors = torch.stack([center, left, right], dim=0)
        return anchors, forward_dir, trap_mode

    def _assign_anchors(self, drone_pos: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        n_agents = drone_pos.shape[0]
        anchor_count = anchors.shape[0]
        assert n_agents == anchor_count, "Current expert assumes 3 drones and 3 anchors."
        dist = torch.cdist(drone_pos.unsqueeze(0), anchors.unsqueeze(0)).squeeze(0)
        best_perm = None
        best_cost = None
        switch_penalty = 0.20
        for perm in itertools.permutations(range(anchor_count)):
            perm_tensor = torch.tensor(perm, device=drone_pos.device)
            cost = dist[torch.arange(n_agents, device=drone_pos.device), perm_tensor].sum()
            if self.prev_assignment is not None:
                changed = (perm_tensor != self.prev_assignment.to(drone_pos.device)).float().sum()
                cost = cost + switch_penalty * changed
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_perm = perm_tensor
        self.prev_assignment = best_perm.clone().cpu()
        return best_perm

    # ── 各角色目标位置 ──────────────────────────────────────────
    @staticmethod
    def blocker_target(target_pos, target_pos_pred, escape_dir, goal_pos,
                       cfg: EnvCfg) -> torch.Tensor:
        """Blocker：飞到 target→goal 路径上，在目标预测位置前方插上。"""
        goal_dist = (goal_pos - target_pos).norm()
        urgency   = torch.sigmoid((cfg.goal_region_radius + 1.5 - goal_dist) / 0.3)
        lead_dist = 0.3 + 0.4 * (1 - urgency)
        # 优先指向预测位置方向
        pred_dir  = safe_normalize((target_pos_pred - target_pos).unsqueeze(0)).squeeze(0)
        # 混合：逼近 goal 越急，越沿 escape_dir
        blended   = safe_normalize((1 - urgency) * pred_dir + urgency * escape_dir)
        return target_pos + blended * lead_dist

    @staticmethod
    def flanker_target(target_pos, escape_dir, flank_idx, n_flankers,
                       cfg: EnvCfg) -> torch.Tensor:
        """Flanker：在 escape_dir 垂直面上均匀分布，包围目标。"""
        up = torch.tensor([0.0, 0.0, 1.0], device=target_pos.device, dtype=target_pos.dtype)
        cross = torch.linalg.cross(escape_dir, up)
        if cross.norm() < 1e-5:
            cross = torch.tensor([1.0, 0.0, 0.0], device=target_pos.device, dtype=target_pos.dtype)
        perp1 = safe_normalize(cross.unsqueeze(0)).squeeze(0)
        perp2 = torch.linalg.cross(escape_dir, perp1)
        angle = 2 * math.pi * (flank_idx + 0.5) / n_flankers
        offset = math.cos(angle) * perp1 + math.sin(angle) * perp2
        encircle_r = max(cfg.catch_radius * 1.5, 0.5)
        goal_dist  = (cfg.goal_region_center - target_pos).norm()
        urgency    = torch.sigmoid((cfg.goal_region_radius + 1.5 - goal_dist) / 0.3)
        lead       = 0.15 * (1 - urgency)
        return target_pos + escape_dir * lead + offset * encircle_r

    # ── 主推力 + 角速度计算 ─────────────────────────────────────
    @staticmethod
    def vel_to_t_omega(
        desired_vel: torch.Tensor,
        current_vel: torch.Tensor,
        v_max: float,
        current_quat: Optional[torch.Tensor] = None,
        heading_hint_world: Optional[torch.Tensor] = None,
        hover_thrust_ratio: float = 0.52,
        min_thrust_ratio: float = 0.0,
        max_thrust_ratio: float = 0.9,
        max_body_rate_rad_s: float = math.radians(180.0 * 0.8),
        gravity: float = 9.81,
    ) -> tuple:
        """
        将期望速度向量转换为更接近真实四旋翼控制逻辑的 (t, omega)：
          1. 用速度误差构造期望加速度
          2. 用期望加速度 + 重力构造期望推力方向
          3. 再由姿态误差生成机体系角速度命令
        """
        desired_vel = clip_vector_norm(desired_vel.unsqueeze(0), v_max).squeeze(0)
        if desired_vel.norm() < 1e-6:
            t_hover = torch.tensor(hover_thrust_ratio, device=desired_vel.device, dtype=desired_vel.dtype)
            return t_hover, torch.zeros_like(desired_vel), desired_vel

        vel_error = desired_vel - current_vel
        acc_cmd = 3.2 * vel_error
        acc_cmd[:2] = clip_vector_norm(acc_cmd[:2].unsqueeze(0), 4.2).squeeze(0)
        acc_cmd[2] = acc_cmd[2].clamp(-3.5, 3.5)

        e3 = torch.tensor([0.0, 0.0, 1.0], device=desired_vel.device, dtype=desired_vel.dtype)
        thrust_world = acc_cmd + gravity * e3
        thrust_norm = thrust_world.norm().clamp(min=1e-6)
        z_des = thrust_world / thrust_norm
        t = hover_thrust_ratio * (thrust_norm / gravity)
        t = t.clamp(min_thrust_ratio, max_thrust_ratio)

        if current_quat is None:
            return t, torch.zeros_like(desired_vel), desired_vel

        basis_x = torch.tensor([1.0, 0.0, 0.0], device=desired_vel.device, dtype=desired_vel.dtype)
        basis_y = torch.tensor([0.0, 1.0, 0.0], device=desired_vel.device, dtype=desired_vel.dtype)
        basis_z = torch.tensor([0.0, 0.0, 1.0], device=desired_vel.device, dtype=desired_vel.dtype)

        x_cur = quat_rotate(current_quat.unsqueeze(0), basis_x.unsqueeze(0)).squeeze(0)
        y_cur = quat_rotate(current_quat.unsqueeze(0), basis_y.unsqueeze(0)).squeeze(0)
        z_cur = quat_rotate(current_quat.unsqueeze(0), basis_z.unsqueeze(0)).squeeze(0)

        x_ref = project_to_plane(x_cur, z_des)
        if x_ref.norm() < 1e-5:
            x_ref = basis_x
        if heading_hint_world is not None and heading_hint_world.norm() > 1e-5:
            hint = project_to_plane(heading_hint_world, z_des)
            if hint.norm() > 1e-5:
                x_ref = 0.75 * x_ref + 0.25 * hint

        x_des = safe_normalize(x_ref.unsqueeze(0)).squeeze(0)
        y_des = torch.linalg.cross(z_des, x_des)
        if y_des.norm() < 1e-5:
            y_des = basis_y
        y_des = safe_normalize(y_des.unsqueeze(0)).squeeze(0)
        x_des = safe_normalize(torch.linalg.cross(y_des, z_des).unsqueeze(0)).squeeze(0)

        e_rot = 0.5 * (
            torch.linalg.cross(x_cur, x_des)
            + torch.linalg.cross(y_cur, y_des)
            + torch.linalg.cross(z_cur, z_des)
        )
        omega_world = 2.7 * e_rot
        omega_body = quat_rotate_inverse(current_quat.unsqueeze(0), omega_world.unsqueeze(0)).squeeze(0)
        omega_body = clip_vector_norm(
            omega_body.unsqueeze(0), 0.85 * max_body_rate_rad_s
        ).squeeze(0)
        if torch.isnan(omega_body).any():
            omega_body = torch.zeros_like(desired_vel)
        if torch.isnan(t):
            t = torch.tensor(hover_thrust_ratio, device=desired_vel.device, dtype=desired_vel.dtype)
        return t, omega_body, desired_vel

    @staticmethod
    def t_omega_to_pidrate_raw(
        t: torch.Tensor,
        omega: torch.Tensor,
        max_body_rate_rad_s: float,
    ) -> torch.Tensor:
        rate_norm = (omega / max(max_body_rate_rad_s, 1e-6)).clamp(-0.999, 0.999)
        thrust_norm = (2.0 * t - 1.0).clamp(-0.999, 0.999).unsqueeze(-1)
        return safe_atanh(torch.cat([rate_norm, thrust_norm], dim=-1))

    # ── 核心推断接口 ───────────────────────────────────────────
    def get_actions(self, step: int,
                    drone_pos: torch.Tensor,     # [n,3]
                    drone_vel: torch.Tensor,     # [n,3]
                    target_pos: torch.Tensor,    # [1,3]
                    target_vel: torch.Tensor,    # [1,3]
                    target_next_pos: torch.Tensor,  # [1,3]  真实下一帧位置（用于noise模式）
                    drone_quat: Optional[torch.Tensor] = None,  # [n,4]
                    ) -> tuple:
        """
        返回:
          t_list     [n]   每架无人机的归一化推力
          omega_list [n,3] 每架无人机的角速度
          vel_cmd    [n,3] 限幅后的速度命令（用于仿真积分）
        """
        n = drone_pos.shape[0]
        target_pos_flat = target_pos.squeeze(0)   # [3]
        goal_pos        = self.cfg.goal_region_center  # [3]
        escape_dir      = safe_normalize((goal_pos - target_pos_flat).unsqueeze(0)).squeeze(0)

        # 目标预测位置
        target_pos_pred = self.predict_target(
            step, target_pos, target_vel, drone_pos, target_next_pos
        )  # [3]

        anchors, forward_dir, trap_mode = self._build_formation_targets(
            target_pos_flat,
            target_vel.squeeze(0),
            target_pos_pred,
        )
        assignment = self._assign_anchors(drone_pos, anchors)
        target_vel_ff = target_vel.squeeze(0)
        min_target_dist = torch.norm(drone_pos - target_pos_flat, dim=-1).min().item()
        goal_dist = torch.norm(goal_pos - target_pos_flat).item()

        t_list     = []
        omega_list = []
        vel_cmd    = []

        for i in range(n):
            anchor_idx = int(assignment[i].item())
            waypoint = anchors[anchor_idx]
            to_wp = waypoint - drone_pos[i]
            dist_wp = to_wp.norm()

            feedforward = target_vel_ff.clone()
            if anchor_idx == 0:
                feedforward = feedforward + 0.22 * forward_dir
            else:
                feedforward = feedforward - 0.05 * forward_dir
            kp = 1.90 if anchor_idx == 0 else 1.65
            if trap_mode:
                kp += 0.20
            kd = 0.26
            v_des = kp * to_wp + 0.85 * feedforward - kd * drone_vel[i]

            # Keep altitude close to the target plane to avoid aggressive dives.
            # If we are close to the goal and missed the first interception window,
            # rebuild the frontal barrier instead of tail-chasing forever.
            close_mode = dist_wp < 0.45 or min_target_dist < 0.95
            desired_height = float(target_pos_pred[2].item()) + (0.10 if close_mode else 0.05)
            desired_height = max(1.2, min(self.cfg.max_height - 0.4, desired_height))
            v_des[2] = 0.90 * (desired_height - drone_pos[i, 2]) - 0.45 * drone_vel[i, 2]

            if goal_dist < 1.4 and anchor_idx == 0 and dist_wp > 0.45:
                frontal_wp = target_pos_pred + 0.95 * forward_dir
                v_des = 1.60 * (frontal_wp - drone_pos[i]) + 0.70 * feedforward - 0.35 * drone_vel[i]
                v_des[2] = 0.95 * (desired_height - drone_pos[i, 2]) - 0.45 * drone_vel[i, 2]

            # Once a drone is close enough, collapse from its assigned side
            # instead of orbiting around the target forever.
            if close_mode:
                if anchor_idx == 0:
                    close_wp = target_pos_pred + 0.03 * forward_dir
                    inward_gain = 0.18
                else:
                    side_axis = project_to_plane(waypoint - target_pos_flat, forward_dir)
                    if side_axis.norm() > 1e-5:
                        side_axis = safe_normalize(side_axis.unsqueeze(0)).squeeze(0)
                    else:
                        side_axis = torch.zeros_like(forward_dir)
                    close_wp = target_pos_pred - 0.03 * forward_dir + side_axis * (0.12 if trap_mode else 0.18)
                    inward_gain = 0.10
                inward = safe_normalize((target_pos_pred - drone_pos[i]).unsqueeze(0)).squeeze(0)
                v_des = (
                    2.05 * (close_wp - drone_pos[i])
                    + 0.92 * target_vel_ff
                    + inward_gain * inward * self.cfg.v_drone
                    - 0.34 * drone_vel[i]
                )
                v_des[2] = 0.90 * (float(target_pos_pred[2].item()) - drone_pos[i, 2]) - 0.55 * drone_vel[i, 2]

            speed_cap = self.cfg.v_drone if (not close_mode or anchor_idx == 0) else 0.96 * self.cfg.v_drone
            v_des = clip_vector_norm(v_des.unsqueeze(0), speed_cap).squeeze(0)

            quat_i = None if drone_quat is None else drone_quat[i]
            heading_hint = to_wp
            ti, oi, vi = self.vel_to_t_omega(
                v_des,
                drone_vel[i],
                self.cfg.v_drone,
                current_quat=quat_i,
                heading_hint_world=heading_hint,
                hover_thrust_ratio=self.cfg.hover_thrust_ratio,
                min_thrust_ratio=self.cfg.min_thrust_ratio,
                max_thrust_ratio=self.cfg.max_thrust_ratio,
                max_body_rate_rad_s=self.cfg.max_body_rate_rad_s,
                gravity=self.cfg.gravity,
            )
            t_list.append(ti)
            omega_list.append(oi)
            vel_cmd.append(vi)

        return (torch.stack(t_list),
                torch.stack(omega_list),
                torch.stack(vel_cmd))


# ─────────────────────────────────────────────────────────────
# 6.  仿真引擎（简化动力学，不依赖 Isaac Sim）
# ─────────────────────────────────────────────────────────────
class SimpleSim:
    """
    点质量模拟器：
      - 无人机 = 一阶速度跟随 + 速度限幅 (v_max=1.0)
      - 目标   = APF 二阶积分器（与 hideandseek.py 一致）
    """
    def __init__(self, cfg: EnvCfg):
        self.cfg = cfg

    def reset(self, seed: int = 0):
        rng = torch.Generator()
        rng.manual_seed(seed)

        drone_noise = torch.zeros(self.cfg.num_agents, 3)
        drone_noise[:, :2].uniform_(-self.cfg.pos_noise_xy, self.cfg.pos_noise_xy,
                                    generator=rng)
        drone_noise[:, 2].uniform_(-self.cfg.pos_noise_z, self.cfg.pos_noise_z,
                                   generator=rng)
        self.drone_pos = (self.cfg.drone_init_pos.clone() + drone_noise).clamp(
            -self.cfg.arena_size + 0.05, self.cfg.arena_size - 0.05
        )
        self.drone_pos[:, 2].clamp_(0.2, self.cfg.max_height - 0.1)

        t_noise = torch.zeros(1, 3)
        t_noise[:, :2].uniform_(-self.cfg.pos_noise_xy, self.cfg.pos_noise_xy,
                                generator=rng)
        t_noise[:, 2].uniform_(-self.cfg.pos_noise_z, self.cfg.pos_noise_z,
                               generator=rng)
        self.target_pos = (self.cfg.target_init_pos.clone() + t_noise).clamp(
            -self.cfg.arena_size + 0.05, self.cfg.arena_size - 0.05
        )
        self.target_pos[:, 2].clamp_(0.2, self.cfg.max_height - 0.1)

        self.drone_vel    = torch.zeros(self.cfg.num_agents, 3)
        self.target_vel   = torch.zeros(1, 3)
        self.target_acc   = torch.zeros(1, 3)
        self.step_count   = 0
        return self._make_obs()

    def _make_obs(self):
        return dict(
            drone_pos=self.drone_pos.clone(),
            drone_vel=self.drone_vel.clone(),
            target_pos=self.target_pos.clone(),
            target_vel=self.target_vel.clone(),
        )

    def step(self, vel_cmd: torch.Tensor):
        """
        vel_cmd: [n,3] 专家指令速度，已限幅至 v_drone。
        一阶追踪 + 更新无人机位置。
        """
        cfg = self.cfg
        dt  = cfg.dt

        # ── 先保存目标"下一帧"真实位置（供 noise 预测使用）
        t_next, tv_next, ta_next = step_target(
            self.target_pos, self.target_vel, self.target_acc, self.drone_pos, cfg
        )

        # ── 无人机动力学：一阶速度跟踪
        alpha_v = min(1.0, dt / 0.05)   # 时间常数 ~50ms
        self.drone_vel = self.drone_vel + alpha_v * (vel_cmd - self.drone_vel)
        self.drone_vel = clip_vector_norm(self.drone_vel, cfg.v_drone)   # << 关键截断
        self.drone_pos = self.drone_pos + self.drone_vel * dt

        # 边界裁减
        xy_r = self.drone_pos[:, :2].norm(dim=-1, keepdim=True)
        out_mask = xy_r > (cfg.arena_size - 0.05)
        scale = torch.where(out_mask,
                            (cfg.arena_size - 0.05) / (xy_r + 1e-6),
                            torch.ones_like(xy_r))
        self.drone_pos[:, :2] *= scale
        self.drone_pos[:, 2].clamp_(0.2, cfg.max_height - 0.1)

        # ── 目标前进
        self.target_pos = t_next
        self.target_vel = tv_next
        self.target_acc = ta_next

        self.step_count += 1

        # ── 终止条件
        target_dist  = (self.drone_pos - self.target_pos).norm(dim=-1)  # [n]
        captured     = bool((target_dist < cfg.catch_radius).any())
        goal_reached = bool(point_in_cylinder(
            self.target_pos.squeeze(0),
            cfg.goal_region_center,
            cfg.goal_region_radius,
            cfg.goal_region_height
        ))
        timeout = self.step_count >= cfg.max_episode_length

        done = captured or goal_reached or timeout
        info = dict(
            captured=captured,
            goal_reached=goal_reached,
            timeout=timeout,
            target_next_pos=t_next,   # 供 noise 模式预测
        )
        return self._make_obs(), done, info


# ─────────────────────────────────────────────────────────────
# 7.  测试循环
# ─────────────────────────────────────────────────────────────
def run_test(n_episodes: int = 50,
             pred_mode: str = "noise",
             tp_weight_path: str = "",
             seed: int = 0) -> dict:
    cfg    = CFG
    sim    = SimpleSim(cfg)
    expert = ExpertPolicy(cfg, pred_mode=pred_mode, tp_weight_path=tp_weight_path)

    results = []
    print(f"\n{'='*60}")
    print(f"  专家策略测试 | 预测模式: {pred_mode} | 轮数: {n_episodes}")
    print(f"{'='*60}")

    for ep in range(n_episodes):
        obs = sim.reset(seed=seed + ep)
        expert.reset()
        done = False
        ep_steps = 0
        ep_info  = {}

        while not done:
            target_next_pos = obs.get("target_next_pos",
                                      obs["target_pos"])  # fallback

            ts, omegas, vel_cmd = expert.get_actions(
                step          = sim.step_count,
                drone_pos     = obs["drone_pos"],
                drone_vel     = obs["drone_vel"],
                target_pos    = obs["target_pos"],
                target_vel    = obs["target_vel"],
                target_next_pos = obs.get("target_next_pos", obs["target_pos"])
            )

            obs, done, info = sim.step(vel_cmd)
            # make next target position available in obs for next step
            obs["target_next_pos"] = info["target_next_pos"]
            ep_steps += 1
            ep_info   = info

        results.append({
            "episode":      ep,
            "steps":        ep_steps,
            "captured":     ep_info["captured"],
            "goal_reached": ep_info["goal_reached"],
            "timeout":      ep_info["timeout"],
            "t_sample":     ts.mean().item(),       # 最后一步推力均值
            "omega_norm":   omegas.norm(dim=-1).mean().item(),  # 角速度均值
        })

        outcome = ("✅ 捕获" if ep_info["captured"]     else
                   "❌ 守区" if ep_info["goal_reached"] else
                   "⏱  超时")
        print(f"  EP {ep+1:03d} | {outcome} | steps={ep_steps:4d} "
              f"| t={ts.mean():.3f} | ω={omegas.norm(dim=-1).mean():.3f}")

    return results


# ─────────────────────────────────────────────────────────────
# 8.  结果统计 & 图表
# ─────────────────────────────────────────────────────────────
def compute_stats(results: list) -> dict:
    n          = len(results)
    captured   = [r for r in results if r["captured"]]
    goal       = [r for r in results if r["goal_reached"]]
    timeout    = [r for r in results if r["timeout"]]
    cap_steps  = [r["steps"] for r in captured]

    stats = {
        "n_episodes":     n,
        "success_rate":   len(captured) / n,
        "goal_rate":      len(goal) / n,
        "timeout_rate":   len(timeout) / n,
        "capture_count":  len(captured),
        "goal_count":     len(goal),
        "timeout_count":  len(timeout),
        "cap_steps_mean": float(np.mean(cap_steps)) if cap_steps else float("nan"),
        "cap_steps_std":  float(np.std(cap_steps))  if cap_steps else float("nan"),
        "cap_steps_min":  float(np.min(cap_steps))  if cap_steps else float("nan"),
        "cap_steps_max":  float(np.max(cap_steps))  if cap_steps else float("nan"),
        "cap_steps_list": cap_steps,
        "all_steps":      [r["steps"] for r in results],
    }
    return stats


def make_figures(results: list, stats: dict, pred_mode: str, out_dir: str = "figures"):
    os.makedirs(out_dir, exist_ok=True)

    ep_nums  = [r["episode"] + 1 for r in results]
    outcomes = [("success" if r["captured"] else
                 "goal"    if r["goal_reached"] else
                 "timeout") for r in results]
    steps    = [r["steps"] for r in results]

    color_map = {"success": "#2ecc71", "goal": "#e74c3c", "timeout": "#95a5a6"}
    label_map = {"success": f"Capture ({stats['success_rate']:.0%})",
                 "goal":    f"Goal Zone ({stats['goal_rate']:.0%})",
                 "timeout": f"Timeout ({stats['timeout_rate']:.0%})"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Expert Policy Test Results  |  Prediction: {pred_mode}  |  N={len(results)} episodes",
        fontsize=15, fontweight="bold", y=0.98
    )

    # -- Plot 1: Steps per episode (color-coded by outcome)
    ax = axes[0, 0]
    bar_colors = [color_map[o] for o in outcomes]
    bars = ax.bar(ep_nums, steps, color=bar_colors, edgecolor="none", alpha=0.85, width=0.8)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Steps", fontsize=11)
    ax.set_title("Steps per Episode (by Outcome)", fontsize=12)
    ax.set_xlim(0, len(results) + 1)
    ax.axhline(stats["cap_steps_mean"], color="#2ecc71", linestyle="--",
               linewidth=1.5, label=f"Avg capture steps: {stats['cap_steps_mean']:.0f}")
    patches = [mpatches.Patch(color=color_map[k], label=label_map[k])
               for k in color_map]
    ax.legend(handles=patches + [ax.get_lines()[0]], loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # -- Plot 2: Pie chart - outcome distribution
    ax = axes[0, 1]
    sizes  = [stats["capture_count"], stats["goal_count"], stats["timeout_count"]]
    labels = [f"Capture\n{stats['capture_count']} eps",
              f"Goal Zone\n{stats['goal_count']} eps",
              f"Timeout\n{stats['timeout_count']} eps"]
    palette = [color_map["success"], color_map["goal"], color_map["timeout"]]
    mask = [s > 0 for s in sizes]
    wedges, texts, autotexts = ax.pie(
        [s for s, m in zip(sizes, mask) if m],
        labels    = [l for l, m in zip(labels, mask) if m],
        colors    = [c for c, m in zip(palette, mask) if m],
        autopct   = "%1.1f%%",
        startangle= 90,
        pctdistance=0.75,
        textprops = {"fontsize": 11},
    )
    ax.set_title(f"Outcome Distribution (n={len(results)})", fontsize=12)

    # -- Plot 3: Capture steps histogram
    ax = axes[1, 0]
    if stats["cap_steps_list"]:
        n_bins = min(20, max(5, len(stats["cap_steps_list"]) // 3))
        ax.hist(stats["cap_steps_list"], bins=n_bins, color="#2ecc71",
                edgecolor="white", linewidth=0.5, alpha=0.85)
        ax.axvline(stats["cap_steps_mean"], color="#e74c3c", linestyle="--",
                   linewidth=2, label=f"Mean {stats['cap_steps_mean']:.1f}")
        ax.axvline(stats["cap_steps_mean"] - stats["cap_steps_std"],
                   color="#e74c3c", linestyle=":", linewidth=1.2)
        ax.axvline(stats["cap_steps_mean"] + stats["cap_steps_std"],
                   color="#e74c3c", linestyle=":", linewidth=1.2,
                   label=f"±1sigma ({stats['cap_steps_std']:.1f})")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No successful episodes", ha="center", va="center",
                fontsize=13, color="gray", transform=ax.transAxes)
    ax.set_xlabel("Capture Steps", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Capture Steps Distribution (successful episodes)", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # -- Plot 4: Cumulative success / goal rate curve
    ax = axes[1, 1]
    cum_success = np.cumsum([1 if r["captured"] else 0 for r in results])
    cum_rate    = cum_success / np.arange(1, len(results) + 1)
    cum_goal    = np.cumsum([1 if r["goal_reached"] else 0 for r in results])
    cum_goal_r  = cum_goal / np.arange(1, len(results) + 1)

    ax.plot(ep_nums, cum_rate,   color="#2ecc71", linewidth=2.0,
            label=f"Cumulative success rate -> {stats['success_rate']:.1%}")
    ax.plot(ep_nums, cum_goal_r, color="#e74c3c", linewidth=2.0,
            linestyle="--", label=f"Cumulative goal rate -> {stats['goal_rate']:.1%}")
    ax.fill_between(ep_nums, 0, cum_rate, alpha=0.15, color="#2ecc71")
    ax.fill_between(ep_nums, 0, cum_goal_r, alpha=0.12, color="#e74c3c")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Cumulative Rate", fontsize=11)
    ax.set_title("Cumulative Success / Goal-Zone Rate", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.35)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(out_dir, f"expert_test_{pred_mode}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart saved: {out_path}")
    return out_path


def print_summary(stats: dict, pred_mode: str):
    print(f"\n{'='*60}")
    print(f"  专家策略测试摘要  [预测模式: {pred_mode}]")
    print(f"{'='*60}")
    print(f"  测试轮数     : {stats['n_episodes']}")
    print(f"  ✅ 成功率     : {stats['success_rate']:.1%}  ({stats['capture_count']} 轮)")
    print(f"  ❌ 守区率     : {stats['goal_rate']:.1%}  ({stats['goal_count']} 轮)")
    print(f"  ⏱  超时率     : {stats['timeout_rate']:.1%}  ({stats['timeout_count']} 轮)")
    if stats["cap_steps_list"]:
        print(f"  捕获步数")
        print(f"    均值       : {stats['cap_steps_mean']:.1f}")
        print(f"    标准差     : {stats['cap_steps_std']:.1f}")
        print(f"    最小/最大  : {stats['cap_steps_min']:.0f} / {stats['cap_steps_max']:.0f}")
    else:
        print("  捕获步数     : 无成功轮次")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────
# 9.  入口
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="独立专家策略测试（无需 Isaac Sim）"
    )
    parser.add_argument("--n_episodes", type=int, default=50,
                        help="测试轮数（默认 50）")
    parser.add_argument("--pred_mode", choices=["noise", "tp_net"], default="noise",
                        help="目标预测模式: noise（加噪GT）或 tp_net（网络预测）")
    parser.add_argument("--tp_weight", type=str,
                        default="checkpoints/HideAndSeek_20260403_001241/tp_only_1690959872.pt",
                        help="TP_net 权重文件路径（pred_mode=tp_net 时使用）")
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子基础值（默认 0）")
    parser.add_argument("--out_dir", type=str, default="figures",
                        help="图表输出目录（默认 figures/）")
    args = parser.parse_args()

    # 确保在项目根目录下运行
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    print(f"  工作目录: {os.getcwd()}")

    tp_path = os.path.abspath(args.tp_weight)
    if args.pred_mode == "tp_net" and not os.path.isfile(tp_path):
        print(f"[ERROR] tp_net 模式需要权重文件，但未找到: {tp_path}")
        print("  请切换为 --pred_mode noise 或提供正确路径。")
        sys.exit(1)

    torch.manual_seed(args.seed)
    results = run_test(
        n_episodes    = args.n_episodes,
        pred_mode     = args.pred_mode,
        tp_weight_path= tp_path,
        seed          = args.seed,
    )
    stats   = compute_stats(results)
    print_summary(stats, args.pred_mode)
    make_figures(results, stats, args.pred_mode, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
