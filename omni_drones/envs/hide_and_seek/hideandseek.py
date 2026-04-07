import torch
import numpy as np
import functorch
import logging
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec
from tensordict.tensordict import TensorDict, TensorDictBase
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import wandb
import time
from functorch import vmap
from omni_drones.utils.torch import cpos, off_diag, quat_axis, others
import torch.distributions as D
from torch.masked import masked_tensor, as_masked_tensor

import omni.isaac.core.objects as objects
# from omni.isaac.core.objects import VisualSphere, DynamicSphere, FixedCuboid, VisualCylinder, FixedCylinder, DynamicCylinder
# from omni.isaac.core.prims import RigidPrimView, GeometryPrimView
import omni.isaac.core.prims as prims
from omni_drones.views import RigidPrimView
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
from omni_drones.robots.config import RobotCfg
from omni_drones.robots.drone import MultirotorBase
import omni_drones.utils.kit as kit_utils
# import omni_drones.utils.restart_sampling as rsp
from pxr import UsdGeom, Usd, UsdPhysics
import omni.isaac.core.utils.prims as prim_utils
import omni.physx.scripts.utils as script_utils
from omni_drones.utils.scene import design_scene
import pdb
import copy
from omni_drones.utils.torch import euler_to_quaternion

from omni.isaac.debug_draw import _debug_draw
import os

from .draw import draw_traj, draw_detection, draw_catch, draw_court
from .draw_circle import Float3, _COLOR_ACCENT, _carb_float3_add, draw_court_circle
import time
import collections
from omni_drones.learning import TP_net
import math


def polygon_area_xy(points_xy: torch.Tensor) -> torch.Tensor:
    x = points_xy[..., 0]
    y = points_xy[..., 1]
    cross = x * torch.roll(y, shifts=-1, dims=-1) - y * torch.roll(x, shifts=-1, dims=-1)
    return 0.5 * torch.abs(cross.sum(dim=-1))


def point_to_cylinder_distance(
    points: torch.Tensor, center: torch.Tensor, radius: float, height: float
) -> torch.Tensor:
    radial_dist = torch.norm(points[..., :2] - center[..., :2], dim=-1)
    radial_delta = torch.relu(radial_dist - radius)
    vertical_delta = torch.relu(torch.abs(points[..., 2] - center[..., 2]) - height / 2.0)
    return torch.sqrt(radial_delta.square() + vertical_delta.square())


def point_in_cylinder(points: torch.Tensor, center: torch.Tensor, radius: float, height: float) -> torch.Tensor:
    radial_dist = torch.norm(points[..., :2] - center[..., :2], dim=-1)
    vertical_ok = torch.abs(points[..., 2] - center[..., 2]) <= height / 2.0
    return (radial_dist <= radius) & vertical_ok


def closest_point_on_cylinder(
    points: torch.Tensor, center: torch.Tensor, radius: float, height: float
) -> torch.Tensor:
    offset_xy = points[..., :2] - center[..., :2]
    radial_dist = torch.norm(offset_xy, dim=-1, keepdim=True)
    safe_radial = torch.where(radial_dist > 1e-5, radial_dist, torch.ones_like(radial_dist))
    clamped_radial = torch.minimum(radial_dist, torch.full_like(radial_dist, radius))
    closest_xy = center[..., :2] + offset_xy / safe_radial * clamped_radial
    closest_z = points[..., 2:3].clamp(center[..., 2:3] - height / 2.0, center[..., 2:3] + height / 2.0)
    return torch.cat([closest_xy, closest_z], dim=-1)


def clip_vector_norm(v: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = torch.norm(v, dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / (norm + 1e-6), max=1.0)
    return v * scale


def safe_normalize(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return v / (torch.norm(v, dim=-1, keepdim=True) + eps)


def compute_phi_team(
    block_i: torch.Tensor,
    dist_i: torch.Tensor,
    norm_dir: torch.Tensor,
    g_urg: torch.Tensor,
    pressure_scale: float,
):
    n = block_i.shape[1]
    device = block_i.device
    batch = block_i.shape[0]
    if n == 0:
        zeros = torch.zeros(batch, device=device)
        return zeros, zeros, zeros, zeros

    topk = min(n, 2)
    top_block = block_i.topk(topk, dim=1).values
    phi_block = 0.7 * block_i.max(dim=1).values + 0.3 * top_block.mean(dim=1)

    top_dist = dist_i.topk(topk, dim=1, largest=False).values
    phi_pressure = torch.exp(-top_dist.mean(dim=1) / max(pressure_scale, 1e-6))

    if n == 1:
        phi_spread = torch.zeros(batch, device=device)
    elif n == 2:
        cos12 = (norm_dir[:, 0, :] * norm_dir[:, 1, :]).sum(dim=-1).clamp(-1.0, 1.0)
        phi_spread = 0.5 * (1.0 - cos12)
    else:
        phi_spread = 1.0 - norm_dir.mean(dim=1).norm(dim=-1)

    phi_team = (
        0.50 * (0.4 + 0.6 * g_urg) * phi_block
        + 0.30 * phi_pressure
        + 0.20 * (1.0 - g_urg) * phi_spread
    )
    return phi_block, phi_pressure, phi_spread, phi_team


def compute_leave_one_out_di(
    block_i: torch.Tensor,
    dist_i: torch.Tensor,
    norm_dir: torch.Tensor,
    g_urg: torch.Tensor,
    pressure_scale: float,
    di_scale: float,
) -> torch.Tensor:
    n = block_i.shape[1]
    if n <= 1:
        return torch.zeros_like(block_i)

    _, _, _, phi_full = compute_phi_team(
        block_i, dist_i, norm_dir, g_urg, pressure_scale
    )
    d_i = torch.zeros_like(block_i)
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        _, _, _, phi_without_i = compute_phi_team(
            block_i[:, idx],
            dist_i[:, idx],
            norm_dir[:, idx],
            g_urg,
            pressure_scale,
        )
        d_i[:, i] = phi_full - phi_without_i
    return torch.tanh(d_i / max(di_scale, 1e-6))

class HideAndSeek(IsaacEnv): 
    """
    Obstacle-free goal-defense environment with curriculum learning.

    Internal functions:

        _set_specs(self): 
            Set environment specifications for observations, states, actions, 
            rewards, statistics, infos, and initialize agent specifications

        _design_scenes(self): 
            Generate simulation scene and initialize all required objects
            
        _reset_idx(self, env_ids: torch.Tensor): 
            Reset poses of all objects, statistics and infos

        _pre_sim_step(self, tensordict: TensorDictBase):
            Process need to be completed before each step of simulation,
            including updating the APF-driven second-order target velocity

        _compute_state_and_obs(self):
            Obtain the observations and states tensor from drone state data
            Observations are organized into:
                state_self: target relative state, target prediction, self state, one-dimensional time encoding
                state_others: teammate relative state plus defense-geometry features
                cooperation: shared goal-defense coordination features
            States contain centralized drone features for the critic

        _compute_reward_and_done(self):
            Compute obstacle-free goal-defense rewards from pursuer-target geometry,
            goal progress, collisions, and terminal events

        _get_dummy_policy_prey(self):
            Get APF forces for the target to move
            Force = goal attraction + pursuer repulsion + region-boundary repulsion

    """
    def __init__(self, cfg, headless):
        self.max_agents = int(cfg.task.max_agents)
        self.cooperation_dim = 10
        self.state_others_dim = 8
        super().__init__(cfg, headless)
        self.drone.initialize()
        self.target = RigidPrimView(
            "/World/envs/env_*/target",
            reset_xform_properties=False,
            shape=[self.num_envs, -1],
        )
        self.target.initialize()
        
        self.time_encoding = self.cfg.task.time_encoding

        self.target_init_vel = self.target.get_velocities(clone=True)
        self.env_ids = torch.from_numpy(np.arange(0, cfg.env.num_envs))
        self.arena_size = self.cfg.task.arena_size
        self.returns = self.progress_buf * 0
        self.collision_radius = self.cfg.task.collision_radius
        self.init_poses = self.drone.get_world_poses(clone=True)
        self.v_prey = self.cfg.task.v_drone * self.cfg.task.v_prey
        self.base_target_speed = float(self.v_prey)
        self.current_target_speed = float(self.base_target_speed)
        self.catch_reward_coef = self.cfg.task.catch_reward_coef
        self.timeout_penalty_coef = float(getattr(self.cfg.task, "timeout_penalty_coef", 2.0))
        self.collision_coef = float(self.cfg.task.collision_coef)
        self.speed_coef = self.cfg.task.speed_coef
        self.capture_progress_coef = float(getattr(self.cfg.task, "capture_progress_coef", 0.2))
        self.coop_phi_coef = float(getattr(self.cfg.task, "coop_phi_coef", 0.15))
        self.coop_diff_coef = float(getattr(self.cfg.task, "coop_diff_coef", 0.35))
        self.coop_reward_init = float(getattr(self.cfg.task, "coop_reward_init", 0.60))
        self.coop_reward_final = float(getattr(self.cfg.task, "coop_reward_final", 0.15))
        self.coop_anneal_portion = float(getattr(self.cfg.task, "coop_anneal_portion", 0.50))
        self.goal_progress_scale = float(getattr(self.cfg.task, "goal_progress_scale", 0.05))
        self.capture_progress_scale = float(getattr(self.cfg.task, "capture_progress_scale", 0.05))
        self.di_scale = float(getattr(self.cfg.task, "di_scale", 0.15))
        self.urgent_goal_dist = float(getattr(self.cfg.task, "urgent_goal_dist", 0.8))
        self.urgent_goal_tau = float(getattr(self.cfg.task, "urgent_goal_tau", 0.15))
        self.block_margin = float(getattr(self.cfg.task, "block_margin", 0.15))
        self.block_softness = float(getattr(self.cfg.task, "block_softness", 0.10))
        self.block_between_softness = float(getattr(self.cfg.task, "block_between_softness", 0.15))
        self.block_lateral_scale = float(getattr(self.cfg.task, "block_lateral_scale", 0.35))
        self.pressure_dist_scale = float(getattr(self.cfg.task, "pressure_dist_scale", 0.8))
        self.time_penalty_coef = float(getattr(self.cfg.task, "time_penalty_coef", 0.002))
        self.smoothness_coef = self.cfg.task.smoothness_coef
        self.ground_collision_height = self.cfg.task.ground_collision_height
        self.landed_z_threshold = self.cfg.task.landed_z_threshold
        self.landed_speed_threshold = self.cfg.task.landed_speed_threshold
        self.landed_penalty_coef = self.cfg.task.landed_penalty_coef
        self.goal_progress_coef = float(getattr(self.cfg.task, "goal_progress_coef", 1.0))
        self.goal_penalty_coef = float(getattr(self.cfg.task, "goal_penalty_coef", 200.0))
        self.target_goal_attraction_coef = float(getattr(self.cfg.task, "target_goal_attraction_coef", 1.0))
        self.target_accel_limit = float(getattr(self.cfg.task, "target_accel_limit", 2.0))
        self.target_velocity_damping = float(getattr(self.cfg.task, "target_velocity_damping", 0.25))
        self.target_command_tau = float(getattr(self.cfg.task, "target_command_tau", 0.15))
        self.use_eval = self.cfg.task.use_eval
        self.use_partial_obs = self.cfg.task.use_partial_obs
        self.goal_region_radius = float(getattr(self.cfg.task, "goal_region_radius", 0.5))
        self.goal_region_height = float(getattr(self.cfg.task, "goal_region_height", 2.0))
        goal_region_center_cfg = getattr(self.cfg.task, "goal_region_center", None)
        if goal_region_center_cfg is None:
            self.goal_region_center = torch.tensor(
                [
                    self.arena_size - self.goal_region_radius,
                    0.0,
                    self.max_height / 2.0,
                ],
                device=self.device,
                dtype=torch.float32,
            )
        else:
            self.goal_region_center = torch.tensor(
                goal_region_center_cfg, device=self.device, dtype=torch.float32
            )
        self.training_progress = 0.0
        self.position_scale = torch.tensor(
            [self.arena_size, self.arena_size, self.max_height],
            device=self.device,
            dtype=torch.float32,
        )
        self.goal_distance_norm = max(self.arena_size, 1e-6)
        self.velocity_scale = max(float(self.cfg.task.v_drone), 1e-6)
        self.target_velocity_scale = max(float(self.current_target_speed), 1e-6)
        curr_cfg = getattr(self.cfg.task, "curriculum", None)
        default_curriculum_stages = [
            {"distance_range": (0.3, 0.8), "target_speed": 0.5, "promote_success": 0.60, "promote_capture_step": 400.0},
            {"distance_range": (0.5, 1.5), "target_speed": 0.8, "promote_success": 0.50},
            {"distance_range": (1.0, 2.5), "target_speed": self.base_target_speed},
        ]
        self.curriculum_enabled = bool(getattr(curr_cfg, "enabled", False)) and self.scenario_flag == "goal_defense"
        self.curriculum_eval_uses_fixed_layout = bool(getattr(curr_cfg, "eval_uses_fixed_layout", True))
        self.curriculum_ema_alpha = float(getattr(curr_cfg, "ema_alpha", 0.10))
        self.curriculum_min_stage_episodes = int(getattr(curr_cfg, "min_stage_episodes", max(self.num_envs, 100)))
        self.curriculum_target_spawn_x = tuple(getattr(curr_cfg, "target_spawn_x", (-0.9, -0.2)))
        self.curriculum_target_spawn_y = tuple(getattr(curr_cfg, "target_spawn_y", (-1.0, 1.0)))
        self.curriculum_target_z_jitter = float(getattr(curr_cfg, "target_z_jitter", 0.15))
        self.curriculum_drone_z_jitter = float(getattr(curr_cfg, "drone_z_jitter", 0.15))
        self.curriculum_drone_min_separation = float(
            getattr(curr_cfg, "drone_min_separation", max(4.0 * self.collision_radius, 0.30))
        )
        self.curriculum_max_sample_attempts = int(getattr(curr_cfg, "max_sample_attempts", 64))
        if curr_cfg is not None and getattr(curr_cfg, "stages", None):
            self.curriculum_stages = []
            for stage_cfg in curr_cfg.stages:
                dist_range = tuple(float(v) for v in stage_cfg.pursuer_target_dist)
                parsed_stage = {
                    "distance_range": dist_range,
                    "target_speed": float(getattr(stage_cfg, "target_speed", self.base_target_speed)),
                }
                if hasattr(stage_cfg, "promote_success"):
                    parsed_stage["promote_success"] = float(stage_cfg.promote_success)
                if hasattr(stage_cfg, "promote_capture_step"):
                    parsed_stage["promote_capture_step"] = float(stage_cfg.promote_capture_step)
                self.curriculum_stages.append(parsed_stage)
        else:
            self.curriculum_stages = default_curriculum_stages
        self.curriculum_stage = 0
        self.curriculum_stage_episodes = 0
        self.curriculum_success_ema = 0.0
        self.curriculum_capture_step_ema = float(self.max_episode_length)
        self.curriculum_capture_ema_initialized = False
        self._set_curriculum_stage(
            0 if self.curriculum_enabled else len(self.curriculum_stages) - 1,
            reset_metrics=True,
            announce=False,
        )
        self.capture = torch.zeros(self.num_envs, self.num_agents, device=self.device)
        self.min_dist = torch.ones(self.num_envs, 1, device=self.device) * float(torch.inf) # for teacher evaluation
        # prev_target_dist for distance progress reward: [num_envs, num_agents]
        self.prev_target_dist = torch.zeros(self.num_envs, self.num_agents, device=self.device)
        self.prev_goal_dist = torch.zeros(self.num_envs, 1, device=self.device)
        self.target_acc_cmd = torch.zeros(self.num_envs, 1, 3, device=self.device)
        
        self.central_env_pos = Float3(
            *self.envs_positions[self.central_env_idx].tolist()
        )

        self.init_drone_pos_dist = D.Uniform(
            torch.tensor([0.1, -self.arena_size / math.sqrt(2.0) + 0.1], device=self.device),
            torch.tensor([self.arena_size / math.sqrt(2.0) - 0.1, self.arena_size / math.sqrt(2.0) - 0.1], device=self.device)
        )
        self.init_target_pos_dist = D.Uniform(
            torch.tensor([-self.arena_size / math.sqrt(2.0) + 0.1, -self.arena_size / math.sqrt(2.0) + 0.1], device=self.device),
            torch.tensor([-0.1, self.arena_size / math.sqrt(2.0) - 0.1], device=self.device)
        )

        self.init_drone_pos_dist_z = D.Uniform(
            torch.tensor([self.max_height / 2 - 0.1], device=self.device),
            torch.tensor([self.max_height / 2 + 0.1], device=self.device)
        )
        self.init_target_pos_dist_z = D.Uniform(
            torch.tensor([self.max_height / 2 - 0.1], device=self.device),
            torch.tensor([self.max_height / 2 + 0.1], device=self.device)
        )

        self.init_rpy_dist = D.Uniform(
            torch.tensor([-0.2, -0.2, 0.0], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 0.2], device=self.device) * torch.pi
        )

        if self.use_eval:
            self.init_rpy_dist = D.Uniform(
                torch.tensor([0.0, 0.0, 0.0], device=self.device) * torch.pi,
                torch.tensor([0.0, 0.0, 0.0], device=self.device) * torch.pi
            )

        self.mask_value = -5
        self.draw = _debug_draw.acquire_debug_draw_interface()
        
        # use self.masked_drone_pos to expand drone_pos
        self.masked_drone_pos = self.mask_value * torch.ones(self.num_envs, self.max_agents - self.num_agents, 3, device=self.device)

        # TP net
        # t, target pos, target vel, padded drone_pos
        self.TP = TP_net(
            input_dim=1 + 3 + 3 + 3 * self.max_agents,
            output_dim=3 * self.future_predcition_step,
            future_predcition_step=self.future_predcition_step,
            window_step=self.window_step,
        ).to(self.device)
        self.history_step = self.cfg.task.history_step
        self.history_data = collections.deque(maxlen=self.history_step)
        # self.debug_list = []

        # for deployment
        self.prev_actions = torch.zeros(self.num_envs, self.num_agents, 4, device=self.device)

    def set_training_progress(self, fraction: float):
        self.training_progress = float(max(0.0, min(1.0, fraction)))

    def _set_curriculum_stage(self, stage: int, reset_metrics: bool = False, announce: bool = True):
        stage = int(max(0, min(stage, len(self.curriculum_stages) - 1)))
        self.curriculum_stage = stage
        stage_cfg = self.curriculum_stages[stage]
        if self.curriculum_enabled:
            self.current_target_speed = float(stage_cfg["target_speed"])
        else:
            # When curriculum is disabled, respect the caller-provided base target speed
            # instead of silently snapping to the YAML stage preset.
            self.current_target_speed = float(self.base_target_speed)
        self.target_velocity_scale = max(self.current_target_speed, 1e-6)
        if reset_metrics:
            self.curriculum_stage_episodes = 0
            self.curriculum_success_ema = 0.0
            self.curriculum_capture_step_ema = float(self.max_episode_length)
            self.curriculum_capture_ema_initialized = False
        if announce and self.curriculum_enabled:
            logging.info(
                "[Curriculum] switched to stage %d | pursuer-target dist=(%.2f, %.2f) | target_speed=%.2f",
                stage + 1,
                stage_cfg["distance_range"][0],
                stage_cfg["distance_range"][1],
                self.current_target_speed,
            )

    def _maybe_advance_curriculum(self):
        if not self.curriculum_enabled:
            return
        if self.curriculum_stage >= len(self.curriculum_stages) - 1:
            return
        if self.curriculum_stage_episodes < self.curriculum_min_stage_episodes:
            return

        stage_cfg = self.curriculum_stages[self.curriculum_stage]
        success_ready = self.curriculum_success_ema >= float(stage_cfg.get("promote_success", 1.1))
        capture_ready = True
        if "promote_capture_step" in stage_cfg:
            capture_ready = (
                self.curriculum_capture_ema_initialized
                and self.curriculum_capture_step_ema <= float(stage_cfg["promote_capture_step"])
            )
        if success_ready and capture_ready:
            self._set_curriculum_stage(self.curriculum_stage + 1, reset_metrics=True, announce=True)

    def _update_curriculum_metrics(self, done: torch.Tensor, capture_before_goal: torch.Tensor):
        if not self.curriculum_enabled or not self.training:
            return
        done_mask = done.squeeze(-1)
        if not torch.any(done_mask):
            return

        alpha = self.curriculum_ema_alpha
        success_batch = capture_before_goal.squeeze(-1)[done_mask].float().mean().item()
        if self.curriculum_stage_episodes == 0:
            self.curriculum_success_ema = success_batch
        else:
            self.curriculum_success_ema = (1.0 - alpha) * self.curriculum_success_ema + alpha * success_batch
        self.curriculum_stage_episodes += int(done_mask.sum().item())

        successful_done = done_mask & capture_before_goal.squeeze(-1)
        if torch.any(successful_done):
            capture_step_batch = self.progress_buf[successful_done].float().mean().item()
            if not self.curriculum_capture_ema_initialized:
                self.curriculum_capture_step_ema = capture_step_batch
                self.curriculum_capture_ema_initialized = True
            else:
                self.curriculum_capture_step_ema = (
                    (1.0 - alpha) * self.curriculum_capture_step_ema + alpha * capture_step_batch
                )
        self._maybe_advance_curriculum()

    def _sample_goal_defense_curriculum_positions(self, env_ids: torch.Tensor):
        num_envs = len(env_ids)
        goal_z = float(self.goal_region_center[2].item())
        stage_cfg = self.curriculum_stages[self.curriculum_stage]
        dist_min, dist_max = stage_cfg["distance_range"]
        target_x_min, target_x_max = self.curriculum_target_spawn_x
        target_y_min, target_y_max = self.curriculum_target_spawn_y

        target_pos = torch.zeros(num_envs, 1, 3, device=self.device)
        target_pos[..., 0].uniform_(min(target_x_min, target_x_max), max(target_x_min, target_x_max))
        target_pos[..., 1].uniform_(min(target_y_min, target_y_max), max(target_y_min, target_y_max))
        target_pos[..., 2].uniform_(goal_z - self.curriculum_target_z_jitter, goal_z + self.curriculum_target_z_jitter)
        target_pos[..., 2].clamp_(0.3, self.max_height - 0.1)

        drone_pos = torch.zeros(num_envs, self.num_agents, 3, device=self.device)
        fallback_offsets = torch.linspace(
            -0.5 * math.pi, 0.5 * math.pi, self.num_agents, device=self.device
        ) if self.num_agents > 1 else torch.zeros(1, device=self.device)

        for env_idx in range(num_envs):
            target_xy = target_pos[env_idx, 0, :2]
            target_z = target_pos[env_idx, 0, 2]
            placed = False
            for _ in range(self.curriculum_max_sample_attempts):
                radii = torch.empty(self.num_agents, device=self.device).uniform_(dist_min, dist_max)
                angles = torch.empty(self.num_agents, device=self.device).uniform_(0.0, 2.0 * math.pi)
                z_offsets = torch.empty(self.num_agents, device=self.device).uniform_(
                    -self.curriculum_drone_z_jitter, self.curriculum_drone_z_jitter
                )
                candidate = torch.zeros(self.num_agents, 3, device=self.device)
                candidate[:, 0] = target_xy[0] + radii * torch.cos(angles)
                candidate[:, 1] = target_xy[1] + radii * torch.sin(angles)
                candidate[:, 2] = target_z + z_offsets

                in_arena = (candidate[:, 0] ** 2 + candidate[:, 1] ** 2) <= (self.arena_size - 0.10) ** 2
                valid_height = (candidate[:, 2] >= 0.3) & (candidate[:, 2] <= self.max_height - 0.1)
                if self.num_agents > 1:
                    pairwise = torch.cdist(candidate[:, :2], candidate[:, :2])
                    pairwise.fill_diagonal_(float("inf"))
                    separated = bool((pairwise >= self.curriculum_drone_min_separation).all())
                else:
                    separated = True
                if bool(in_arena.all() and valid_height.all()) and separated:
                    drone_pos[env_idx] = candidate
                    placed = True
                    break

            if not placed:
                goal_dir = safe_normalize(
                    (self.goal_region_center[:2] - target_xy).unsqueeze(0)
                ).squeeze(0)
                base_angle = torch.atan2(goal_dir[1], goal_dir[0])
                radii = torch.empty(self.num_agents, device=self.device).uniform_(dist_min, dist_max)
                jitter = torch.empty(self.num_agents, device=self.device).uniform_(-0.20, 0.20)
                angles = base_angle + fallback_offsets + jitter
                candidate = torch.zeros(self.num_agents, 3, device=self.device)
                candidate[:, 0] = target_xy[0] + radii * torch.cos(angles)
                candidate[:, 1] = target_xy[1] + radii * torch.sin(angles)
                candidate[:, 2] = target_z + torch.empty(self.num_agents, device=self.device).uniform_(
                    -self.curriculum_drone_z_jitter, self.curriculum_drone_z_jitter
                )
                candidate[:, 2].clamp_(0.3, self.max_height - 0.1)
                radial_norm = torch.norm(candidate[:, :2], dim=-1, keepdim=True)
                radial_scale = torch.clamp((self.arena_size - 0.10) / (radial_norm + 1e-6), max=1.0)
                candidate[:, :2] = candidate[:, :2] * radial_scale
                drone_pos[env_idx] = candidate

        return drone_pos, target_pos

    def _set_specs(self):        
        drone_state_dim = self.drone.state_spec.shape.numel()
        self.time_encoding_dim = 1 if self.cfg.task.time_encoding else 0
        self.future_predcition_step = self.cfg.task.future_predcition_step
        self.history_step = self.cfg.task.history_step
        self.window_step = self.cfg.task.window_step

        if self.use_TP_net:
            observation_spec = CompositeSpec({
                "state_self": UnboundedContinuousTensorSpec((1, 3 + 3 * self.future_predcition_step + self.time_encoding_dim + 13)),
                "state_others": UnboundedContinuousTensorSpec((self.drone.n-1, self.state_others_dim)),
                "cooperation": UnboundedContinuousTensorSpec((1, self.cooperation_dim)),
            }).to(self.device)
            state_spec = CompositeSpec({
                "state_drones": UnboundedContinuousTensorSpec((self.drone.n, 3 + 3 * self.future_predcition_step + self.time_encoding_dim + 13 + self.cooperation_dim)),
            }).to(self.device)
        else:
            observation_spec = CompositeSpec({
                "state_self": UnboundedContinuousTensorSpec((1, 3 + self.time_encoding_dim + 13)),
                "state_others": UnboundedContinuousTensorSpec((self.drone.n-1, self.state_others_dim)),
                "cooperation": UnboundedContinuousTensorSpec((1, self.cooperation_dim)),
            }).to(self.device)
            state_spec = CompositeSpec({
                "state_drones": UnboundedContinuousTensorSpec((self.drone.n, 3 + self.time_encoding_dim + 13 + self.cooperation_dim)),
            }).to(self.device)
        
        TP_spec = CompositeSpec({
            "TP_input": UnboundedContinuousTensorSpec((self.history_step, 1 + 3 + 3 + self.max_agents * 3)),
            "TP_groundtruth": UnboundedContinuousTensorSpec(3),
            "TP_done": UnboundedContinuousTensorSpec(1),
        }).to(self.device)
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": observation_spec.expand(self.drone.n),
                "state": state_spec,
                "TP": TP_spec
            })
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": torch.stack([self.drone.action_spec]*self.drone.n, dim=0),
            })
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((self.drone.n, 1)),                
            })
        }).expand(self.num_envs).to(self.device)

        self.agent_spec["drone"] = AgentSpec(
            "drone", self.drone.n,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "state"),
        )

        # stats and infos
        stats_spec = CompositeSpec({
            "success": UnboundedContinuousTensorSpec(1),
            "collision": UnboundedContinuousTensorSpec(1),
            "terminal_reward": UnboundedContinuousTensorSpec(1),
            "goal_progress_reward": UnboundedContinuousTensorSpec(1),
            "capture_progress_reward": UnboundedContinuousTensorSpec(1),
            "catch_reward": UnboundedContinuousTensorSpec(1),
            "goal_penalty": UnboundedContinuousTensorSpec(1),
            "goal_reached": UnboundedContinuousTensorSpec(1),
            "speed_reward": UnboundedContinuousTensorSpec(1),
            "time_penalty": UnboundedContinuousTensorSpec(1),
            "coop_reward": UnboundedContinuousTensorSpec(1),
            "phi_team": UnboundedContinuousTensorSpec(1),
            "phi_block": UnboundedContinuousTensorSpec(1),
            "phi_pressure": UnboundedContinuousTensorSpec(1),
            "phi_spread": UnboundedContinuousTensorSpec(1),
            "d_i_mean": UnboundedContinuousTensorSpec(1),
            "d_i_min": UnboundedContinuousTensorSpec(1),
            "urgent_block_rate": UnboundedContinuousTensorSpec(1),
            "n_agents_ahead_mean": UnboundedContinuousTensorSpec(1),
            "collision_penalty": UnboundedContinuousTensorSpec(1),
            "collision_wall": UnboundedContinuousTensorSpec(1),
            "collision_floor": UnboundedContinuousTensorSpec(1),
            "collision_drone": UnboundedContinuousTensorSpec(1),
            "smoothness_reward": UnboundedContinuousTensorSpec(1),
            "landed_penalty": UnboundedContinuousTensorSpec(1),
            "any_landed": UnboundedContinuousTensorSpec(1),
            "smoothness_mean": UnboundedContinuousTensorSpec(1),
            "smoothness_max": UnboundedContinuousTensorSpec(1),
            "first_capture_step": UnboundedContinuousTensorSpec(1),
            "curriculum_stage": UnboundedContinuousTensorSpec(1),
            "curriculum_success_ema": UnboundedContinuousTensorSpec(1),
            "curriculum_capture_step_ema": UnboundedContinuousTensorSpec(1),
            "curriculum_target_speed": UnboundedContinuousTensorSpec(1),
            "sum_detect_step": UnboundedContinuousTensorSpec(1),
            "return": UnboundedContinuousTensorSpec(1),
            "action_error_order1_mean": UnboundedContinuousTensorSpec(1),
            "action_error_order1_max": UnboundedContinuousTensorSpec(1),
            "target_predicted_error": UnboundedContinuousTensorSpec(1),
            "distance_threshold_L": UnboundedContinuousTensorSpec(1),
            "out_of_arena": UnboundedContinuousTensorSpec(1),
            "smoothness_coef": UnboundedContinuousTensorSpec(1),
            "pursuer_collisions_count": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
            "prev_action": torch.stack([self.drone.action_spec] * self.drone.n, 0).to(self.device),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()
        
    def _design_scene(self): # for render
        self.use_local_usd = self.cfg.use_local_usd
        self.num_agents = self.cfg.task.num_agents
        self.drone_detect_radius = self.cfg.task.drone_detect_radius
        self.target_detect_radius = self.cfg.task.target_detect_radius
        self.catch_radius = self.cfg.task.catch_radius
        self.arena_size = self.cfg.task.arena_size
        self.max_height = self.cfg.task.max_height
        self.scenario_flag = self.cfg.task.scenario_flag
        self.boundary = self.arena_size - 0.1
        self.use_TP_net = self.cfg.algo.use_TP_net

        # init
        drone_pos = torch.tensor([
                            [0.6000,  0.0000, 0.5],
                            [0.8000,  0.0000, 0.5],
                            [0.8000, -0.2000, 0.5],
                            [0.8000,  0.2000, 0.5],
                        ], device=self.device)[:self.num_agents]
        target_pos = torch.tensor([
                            [-0.8000,  0.0000, 0.5],
                        ], device=self.device)
        if self.scenario_flag != "goal_defense":
            raise NotImplementedError(
                f"Obstacle-free build only supports scenario_flag='goal_defense', got {self.scenario_flag!r}."
            )

        # init drone
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        cfg.rigid_props.max_linear_velocity = self.cfg.task.v_drone
        self.drone: MultirotorBase = drone_model(cfg=cfg)
        self.drone.spawn(drone_pos)

        # init prey as a bounded double-integrator point mass.
        objects.DynamicSphere(
            prim_path="/World/envs/env_0/target",
            name="target",
            translation=target_pos,
            radius=0.05,
            color=torch.tensor([1.0, 0.0, 0.0]),
            mass=1.0,
        )
        target_physics_max_velocity = float(
            getattr(
                self.cfg.task,
                "target_physics_max_velocity",
                max(5.0, self.cfg.task.v_drone * self.cfg.task.v_prey * 3.0),
            )
        )
        kit_utils.set_rigid_body_properties(
            prim_path="/World/envs/env_0/target",
            disable_gravity=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=target_physics_max_velocity,
        )
    
        # kit_utils.create_ground_plane(
        #     "/World/defaultGroundPlane",
        #     static_friction=1.0,
        #     dynamic_friction=1.0,
        #     restitution=0.0,
        # )

        if self.use_local_usd:
            # use local usd resources
            usd_path = os.path.join(os.path.dirname(__file__), os.pardir, "assets", "default_environment.usd")
            kit_utils.create_ground_plane(
                "/World/defaultGroundPlane",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
                usd_path=usd_path
            )
        else:
            # use online usd resources
            kit_utils.create_ground_plane(
                "/World/defaultGroundPlane",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            )

        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)

        use_goal_defense_curriculum = (
            self.curriculum_enabled
            and self.scenario_flag == "goal_defense"
            and (self.training or not self.curriculum_eval_uses_fixed_layout)
        )

        # init, fixed xy and randomize z
        if use_goal_defense_curriculum:
            drone_pos, target_pos = self._sample_goal_defense_curriculum_positions(env_ids)
        elif not self.use_eval:
            # random pos
            drone_pos = self.init_drone_pos_dist.sample((*env_ids.shape, self.num_agents))
            target_pos =  self.init_target_pos_dist.sample((*env_ids.shape, 1))
            drone_pos_z = self.init_drone_pos_dist_z.sample((*env_ids.shape, self.num_agents))
            target_pos_z = self.init_target_pos_dist_z.sample((*env_ids.shape, 1))
            drone_pos = torch.concat([drone_pos, drone_pos_z], dim=-1)
            target_pos = torch.concat([target_pos, target_pos_z], dim=-1)
        else:
            goal_z = float(self.goal_region_center[2].item())
            drone_pos = torch.tensor(
                [
                    [1.2000, 0.0000, goal_z],
                    [1.5000, 0.6000, goal_z],
                    [1.5000, -0.6000, goal_z],
                    [1.8000, 0.0000, goal_z],
                ],
                device=self.device,
            )[: self.num_agents]
            target_pos = torch.tensor(
                [[-1.6000, 0.0000, goal_z]],
                device=self.device,
            )

        # Add position noise to fixed initial positions for exploration diversity
        if self.use_eval and not use_goal_defense_curriculum:
            pos_noise_xy = 0.15  # ±0.15m noise on x,y
            pos_noise_z = 0.1   # ±0.10m noise on z
            drone_noise = torch.zeros(len(env_ids), self.num_agents, 3, device=self.device)
            drone_noise[..., :2].uniform_(-pos_noise_xy, pos_noise_xy)
            drone_noise[..., 2].uniform_(-pos_noise_z, pos_noise_z)
            drone_pos = drone_pos.unsqueeze(0).expand(len(env_ids), -1, -1) + drone_noise
            # clamp to arena bounds
            drone_pos[..., :2].clamp_(-self.arena_size + 0.05, self.arena_size - 0.05)
            drone_pos[..., 2].clamp_(0.2, self.max_height - 0.1)

            target_noise = torch.zeros(len(env_ids), 1, 3, device=self.device)
            target_noise[..., :2].uniform_(-pos_noise_xy, pos_noise_xy)
            target_noise[..., 2].uniform_(-pos_noise_z, pos_noise_z)
            target_pos = target_pos.unsqueeze(0).expand(len(env_ids), -1, -1) + target_noise
            target_pos[..., :2].clamp_(-self.arena_size + 0.05, self.arena_size - 0.05)
            target_pos[..., 2].clamp_(0.2, self.max_height - 0.1)

        # drone_pos = self.init_drone_pos_dist.sample((*env_ids.shape, self.num_agents))
        rpy = self.init_rpy_dist.sample((*env_ids.shape, self.num_agents))
        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(
            drone_pos + self.envs_positions[env_ids].unsqueeze(1), rot, env_ids
        )
        drone_init_velocities = torch.zeros(len(env_ids) * self.num_agents, 6, device=self.device)
        self.drone.set_velocities(drone_init_velocities, env_ids)

        goal_offset = self.goal_region_center.view(1, 1, 3) - target_pos
        target_rpy = torch.zeros(len(env_ids), 1, 3, device=self.device)
        target_rpy[..., 2] = torch.atan2(goal_offset[..., 1], goal_offset[..., 0])
        target_rot = euler_to_quaternion(target_rpy)
        self.target.set_world_poses(
            positions=target_pos + self.envs_positions[env_ids].unsqueeze(1),
            orientations=target_rot,
            env_indices=env_ids,
        )
        target_init_velocities = torch.zeros(len(env_ids), 6, device=self.device)
        self.target.set_velocities(target_init_velocities, env_ids)
        self.target_acc_cmd[env_ids] = 0.0

        # reset stats
        self.stats[env_ids] = 0.
        self.stats['first_capture_step'][env_ids] = self.max_episode_length
        self.stats["curriculum_stage"][env_ids] = float(self.curriculum_stage + 1)
        self.stats["curriculum_success_ema"][env_ids] = float(self.curriculum_success_ema)
        self.stats["curriculum_capture_step_ema"][env_ids] = float(self.curriculum_capture_step_ema)
        self.stats["curriculum_target_speed"][env_ids] = float(self.current_target_speed)

        # init prev_actions: hover
        cmd_init = 2.0 * (self.drone.throttle[env_ids]) ** 2 - 1.0
        self.info['prev_action'][env_ids, :, 3] = cmd_init.mean(dim=-1)
        self.prev_actions[env_ids] = self.info['prev_action'][env_ids].clone()

        # reset prev_target_dist for distance progress reward
        # drone_pos: [len(env_ids), num_agents, 3], target_pos: [len(env_ids), 1, 3] — broadcasts naturally
        self.prev_target_dist[env_ids] = torch.norm(target_pos - drone_pos, dim=-1)
        goal_dist = point_to_cylinder_distance(
            target_pos.squeeze(1), self.goal_region_center, self.goal_region_radius, self.goal_region_height
        )
        self.prev_goal_dist[env_ids] = goal_dist.unsqueeze(-1)
        
        if self.use_eval and self._should_render(0):
            self._draw_court_circle()
        
        for substep in range(1):
            self.sim.step(self._should_render(substep))

    def _pre_sim_step(self, tensordict: TensorDictBase):   
        actions = tensordict[("agents", "action")]
        
        # for deployment
        self.info["prev_action"] = tensordict[("info", "prev_action")]
        self.prev_actions = self.info["prev_action"].clone()
        self.action_error_order1 = tensordict[("stats", "action_error_order1")].clone()
        self.stats["action_error_order1_mean"].add_(self.action_error_order1.mean(dim=-1).unsqueeze(-1))
        self.stats["action_error_order1_max"].copy_(
            torch.max(
                self.stats["action_error_order1_max"],
                self.action_error_order1.mean(dim=-1).unsqueeze(-1),
            )
        )

        self.effort = self.drone.apply_action(actions)

        target_vel = self.target.get_velocities()
        target_lin_vel = target_vel[..., :3]
        forces_target = self._get_dummy_policy_prey()
        desired_acc = self.target_accel_limit * forces_target / (torch.norm(forces_target, dim=-1, keepdim=True) + 1e-5)
        accel_alpha = min(1.0, self.dt / max(self.target_command_tau, self.dt))
        self.target_acc_cmd.lerp_(desired_acc, accel_alpha)
        next_target_vel = target_lin_vel + (self.target_acc_cmd - self.target_velocity_damping * target_lin_vel) * self.dt
        next_target_vel = clip_vector_norm(next_target_vel, self.current_target_speed)
        target_vel[..., :3] = next_target_vel
        target_vel[..., 3:] = 0.0
        self.target.set_velocities(target_vel.type(torch.float32), self.env_ids)
     
    def _compute_state_and_obs(self):
        self.drone_states = self.drone.get_state()
        self.info["drone_state"][:] = self.drone_states[..., :13]
        drone_pos, _ = self.get_env_poses(self.drone.get_world_poses())
        drone_vel = self.drone.get_velocities()[..., :3]
        self.drone_rpos = vmap(off_diag)(vmap(cpos)(drone_pos, drone_pos))
        self.drone_rvel = vmap(off_diag)(vmap(cpos)(drone_vel, drone_vel))

        obs = TensorDict({}, [self.num_envs, self.drone.n])

        target_pos, _ = self.get_env_poses(self.target.get_world_poses())
        target_vel = self.target.get_velocities()
        target_rpos = vmap(cpos)(drone_pos, target_pos)
        in_detection_range = torch.norm(target_rpos, dim=-1) < self.drone_detect_radius
        self.broadcast_detect = torch.any(in_detection_range, dim=1)

        target_rpos_norm = target_rpos / self.position_scale.view(1, 1, 1, 3)
        target_rpos_mask = (~self.broadcast_detect).unsqueeze(-1).unsqueeze(-1).expand_as(target_rpos_norm)
        target_rpos_masked = target_rpos_norm.clone()
        target_rpos_masked.masked_fill_(target_rpos_mask, self.mask_value)

        t = (self.progress_buf / self.max_episode_length).unsqueeze(-1).unsqueeze(-1)

        target_mask = (~self.broadcast_detect).unsqueeze(-1).expand_as(target_pos)
        target_pos_norm = target_pos / self.position_scale.view(1, 1, 3)
        target_pos_masked = target_pos_norm.clone()
        target_pos_masked.masked_fill_(target_mask, self.mask_value)
        target_vel_norm = target_vel[..., :3] / self.target_velocity_scale
        target_vel_masked = target_vel_norm.clone()
        target_vel_masked.masked_fill_(target_mask, self.mask_value)

        TP = TensorDict({}, [self.num_envs])
        target_rpos_predicted = torch.zeros(
            self.num_envs,
            self.num_agents,
            3 * self.future_predcition_step,
            device=self.device,
        )

        if self.use_TP_net:
            expanded_drone_pos = torch.concat(
                [drone_pos / self.position_scale.view(1, 1, 3), self.masked_drone_pos], dim=1
            )
            frame_state = torch.concat(
                [
                    self.progress_buf.unsqueeze(-1) / max(self.max_episode_length, 1),
                    target_pos_masked.reshape(self.num_envs, -1),
                    target_vel_masked.squeeze(1),
                    expanded_drone_pos.reshape(self.num_envs, -1),
                ],
                dim=-1,
            )
            if len(self.history_data) < self.history_step:
                for _ in range(self.history_step):
                    self.history_data.append(frame_state)
            else:
                self.history_data.append(frame_state)

            TP["TP_input"] = torch.stack(list(self.history_data), dim=1).to(self.device)
            self.target_pos_predicted = self.TP(TP["TP_input"]).reshape(
                self.num_envs, self.future_predcition_step, -1
            )
            self.target_pos_predicted[..., :2] = self.target_pos_predicted[..., :2] * self.arena_size
            self.target_pos_predicted[..., 2] = (
                (self.target_pos_predicted[..., 2] + 1.0) / 2.0 * self.max_height
            )
            self.stats["target_predicted_error"].add_(
                torch.norm(target_pos.squeeze(1) - self.target_pos_predicted[:, 0], dim=-1).unsqueeze(-1)
            )
            TP["TP_done"] = (
                self.progress_buf <= (self.max_episode_length - self.future_predcition_step)
            ).unsqueeze(-1)
            TP["TP_groundtruth"] = target_pos.squeeze(1).clone()
            TP["TP_groundtruth"][..., :2] = TP["TP_groundtruth"][..., :2] / self.arena_size
            TP["TP_groundtruth"][..., 2] = TP["TP_groundtruth"][..., 2] / self.max_height * 2.0 - 1.0
            target_rpos_predicted = (
                (drone_pos.unsqueeze(2) - self.target_pos_predicted.unsqueeze(1))
                / self.position_scale.view(1, 1, 1, 3)
            ).reshape(self.num_envs, self.num_agents, -1)
            tp_escape_dir = safe_normalize(self.target_pos_predicted[:, 0] - target_pos.squeeze(1))
        else:
            self.target_pos_predicted = None
            tp_escape_dir = None

        target_pos_flat = target_pos.squeeze(1)
        goal_vec = self.goal_region_center.view(1, 3) - target_pos_flat
        goal_threat_dir = safe_normalize(goal_vec)
        if tp_escape_dir is None:
            tp_escape_dir = goal_threat_dir
        else:
            tp_escape_dir = torch.where(
                self.broadcast_detect.expand_as(goal_threat_dir),
                tp_escape_dir,
                goal_threat_dir,
            )

        agent_from_target = drone_pos - target_pos
        self.target_dist = torch.norm(agent_from_target, dim=-1)
        norm_dir = safe_normalize(agent_from_target)
        coverage_quality = 1.0 - norm_dir.mean(dim=1).norm(dim=-1, keepdim=True)
        coverage_quality = coverage_quality.expand(-1, self.num_agents)

        goal_expanded = goal_threat_dir.unsqueeze(1).expand(-1, self.num_agents, -1)
        tp_expanded = tp_escape_dir.unsqueeze(1).expand(-1, self.num_agents, -1)
        self.my_ahead_proj_goal = (agent_from_target * goal_expanded).sum(dim=-1)
        self.my_ahead_proj_tp = (agent_from_target * tp_expanded).sum(dim=-1)
        lateral_vec_goal = agent_from_target - self.my_ahead_proj_goal.unsqueeze(-1) * goal_expanded
        self.my_lateral_dist_goal = lateral_vec_goal.norm(dim=-1)
        goal_to_target_dist = goal_vec.norm(dim=-1)
        goal_dist_boundary = point_to_cylinder_distance(
            target_pos_flat, self.goal_region_center, self.goal_region_radius, self.goal_region_height
        )
        self.g_urg = torch.sigmoid(
            (self.urgent_goal_dist - goal_dist_boundary) / max(self.urgent_goal_tau, 1e-6)
        )
        self.block_i = (
            torch.sigmoid((self.my_ahead_proj_goal - self.block_margin) / max(self.block_softness, 1e-6))
            * torch.sigmoid(
                (goal_to_target_dist.unsqueeze(1) - self.my_ahead_proj_goal)
                / max(self.block_between_softness, 1e-6)
            )
            * torch.exp(-self.my_lateral_dist_goal / max(self.block_lateral_scale, 1e-6))
        )
        self.phi_block, self.phi_pressure, self.phi_spread, self.phi_team = compute_phi_team(
            self.block_i,
            self.target_dist,
            norm_dir,
            self.g_urg,
            self.pressure_dist_scale,
        )
        self.d_i_clip = compute_leave_one_out_di(
            self.block_i,
            self.target_dist,
            norm_dir,
            self.g_urg,
            self.pressure_dist_scale,
            self.di_scale,
        )

        goal_threat_dir_expand = goal_threat_dir.unsqueeze(1).expand(-1, self.num_agents, -1)
        tp_escape_dir_expand = tp_escape_dir.unsqueeze(1).expand(-1, self.num_agents, -1)
        cooperation = torch.cat(
            [
                coverage_quality.unsqueeze(-1),
                goal_threat_dir_expand,
                tp_escape_dir_expand,
                (self.my_ahead_proj_goal / self.goal_distance_norm).unsqueeze(-1),
                (self.my_ahead_proj_tp / self.goal_distance_norm).unsqueeze(-1),
                (self.my_lateral_dist_goal / self.goal_distance_norm).unsqueeze(-1),
            ],
            dim=-1,
        ).unsqueeze(2)
        obs["cooperation"] = cooperation

        obs["state_self"] = torch.cat(
            [
                target_rpos_masked.reshape(self.num_envs, self.num_agents, -1),
                target_rpos_predicted,
                self.drone_states[..., 3:10],
                self.drone_states[..., 13:19],
                t.expand(-1, self.num_agents, self.time_encoding_dim),
            ],
            dim=-1,
        ).unsqueeze(2)

        if self.drone.n > 1:
            j_ahead_proj_goal = vmap(others)(self.my_ahead_proj_goal) / self.goal_distance_norm
            j_dist_to_target = vmap(others)(self.target_dist) / self.goal_distance_norm
            obs["state_others"] = torch.cat(
                [
                    self.drone_rpos / self.position_scale.view(1, 1, 1, 3),
                    self.drone_rvel / self.velocity_scale,
                    j_ahead_proj_goal.unsqueeze(-1),
                    j_dist_to_target.unsqueeze(-1),
                ],
                dim=-1,
            )
        else:
            obs["state_others"] = torch.zeros(
                self.num_envs,
                self.num_agents,
                0,
                self.state_others_dim,
                device=self.device,
            )

        state = TensorDict({}, [self.num_envs])
        state["state_drones"] = torch.cat(
            [
                target_rpos_norm.reshape(self.num_envs, self.num_agents, -1),
                target_rpos_predicted,
                self.drone_states[..., 3:10],
                self.drone_states[..., 13:19],
                t.expand(-1, self.num_agents, self.time_encoding_dim),
                cooperation.squeeze(2),
            ],
            dim=-1,
        )

        if self.use_TP_net:
            return TensorDict(
                {
                    "agents": {
                        "observation": obs,
                        "state": state,
                        "TP": TP,
                    },
                    "stats": self.stats,
                    "info": self.info,
                },
                self.batch_size,
            )
        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "state": state,
                },
                "stats": self.stats,
                "info": self.info,
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        drone_pos, _ = self.get_env_poses(self.drone.get_world_poses())
        target_pos, _ = self.get_env_poses(self.target.get_world_poses())

        target_dist = torch.norm(target_pos - drone_pos, dim=-1)
        goal_dist = point_to_cylinder_distance(
            target_pos.squeeze(1), self.goal_region_center, self.goal_region_radius, self.goal_region_height
        )
        goal_progress_reward_val = self.goal_progress_coef * torch.tanh(
            (goal_dist.unsqueeze(-1) - self.prev_goal_dist) / max(self.goal_progress_scale, 1e-6)
        )
        self.prev_goal_dist.copy_(goal_dist.unsqueeze(-1))
        goal_reached = point_in_cylinder(
            target_pos.squeeze(1), self.goal_region_center, self.goal_region_radius, self.goal_region_height
        ).unsqueeze(-1)
        self.stats["goal_reached"].copy_(
            torch.logical_or(self.stats["goal_reached"].bool(), goal_reached).float()
        )

        topk = min(self.num_agents, 2)
        prev_team_dist = self.prev_target_dist.topk(topk, dim=1, largest=False).values.mean(dim=1, keepdim=True)
        curr_team_dist = target_dist.topk(topk, dim=1, largest=False).values.mean(dim=1, keepdim=True)
        capture_progress_reward_val = self.capture_progress_coef * torch.tanh(
            (prev_team_dist - curr_team_dist) / max(self.capture_progress_scale, 1e-6)
        )
        self.prev_target_dist.copy_(target_dist)

        self.capture = target_dist < self.catch_radius
        capture_done = torch.any(self.capture, dim=-1).unsqueeze(-1)

        drone_vel = self.drone.get_velocities()
        drone_speed_norm = torch.norm(drone_vel[..., :3], dim=-1)
        speed_excess = torch.relu(drone_speed_norm / max(self.cfg.task.v_drone, 1e-6) - 1.0)
        speed_reward = -self.speed_coef * speed_excess

        drone_pos_dist = torch.norm(self.drone_rpos, dim=-1)
        collision_drone = (drone_pos_dist < 2.0 * self.collision_radius).float().sum(-1)
        self.stats["collision_drone"].add_(collision_drone.mean(-1).unsqueeze(-1))
        self.stats["pursuer_collisions_count"].add_((collision_drone.sum(-1) / 2).unsqueeze(-1))

        collision_wall = (
            (drone_pos[..., -1] > self.max_height).float()
            + ((drone_pos[..., 0] ** 2 + drone_pos[..., 1] ** 2) > self.arena_size ** 2).float()
        )
        collision_floor = (drone_pos[..., -1] < self.ground_collision_height).float()
        collision_binary = (
            (collision_drone > 0)
            | (collision_wall > 0)
            | (collision_floor > 0)
        ).float()
        collision_penalty = -self.collision_coef * collision_binary
        collision_flag = collision_binary.any(dim=1, keepdim=True)
        self.stats["collision"].add_(collision_flag.float())
        self.stats["collision_wall"].add_(collision_wall.mean(-1).unsqueeze(-1))
        self.stats["collision_floor"].add_(collision_floor.mean(-1).unsqueeze(-1))

        landed = (
            (drone_pos[..., -1] < self.landed_z_threshold)
            & (drone_speed_norm < self.landed_speed_threshold)
        )
        any_landed = landed.any(dim=1, keepdim=True)
        landed_penalty = -self.landed_penalty_coef * any_landed.expand_as(target_dist).float()
        self.stats["any_landed"].add_(any_landed.float())

        timeout = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
            & ~capture_done
            & ~goal_reached
            & ~any_landed
        )
        capture_before_goal = capture_done & ~goal_reached & ~any_landed
        catch_reward = self.catch_reward_coef * capture_before_goal.expand_as(target_dist).float()
        goal_penalty_val = -self.goal_penalty_coef * goal_reached.expand_as(target_dist).float()
        timeout_penalty_val = -self.timeout_penalty_coef * timeout.expand_as(target_dist).float()
        terminal_reward = catch_reward + goal_penalty_val + timeout_penalty_val

        coop_anneal = 1.0
        if self.coop_anneal_portion > 0:
            coop_anneal = min(self.training_progress / self.coop_anneal_portion, 1.0)
        coop_weight = self.coop_reward_init + (
            self.coop_reward_final - self.coop_reward_init
        ) * coop_anneal
        coop_reward = coop_weight * (
            self.coop_diff_coef * self.d_i_clip
            + self.coop_phi_coef * self.phi_team.unsqueeze(-1).expand(-1, self.num_agents)
        )

        airborne_mask = (drone_pos[..., -1] > self.landed_z_threshold).float()
        smoothness_reward = self.smoothness_coef * (torch.exp(-self.action_error_order1) - 1.0) * airborne_mask
        time_penalty = -self.time_penalty_coef * torch.ones_like(target_dist)

        reward = (
            terminal_reward
            + goal_progress_reward_val.expand(-1, self.num_agents)
            + capture_progress_reward_val.expand(-1, self.num_agents)
            + coop_reward
            + collision_penalty
            + landed_penalty
            + speed_reward
            + smoothness_reward
            + time_penalty
        )

        self.stats["success"].copy_(
            torch.logical_or(capture_before_goal, self.stats["success"].bool()).float()
        )
        progress_step = self.progress_buf.unsqueeze(-1).float()
        current_capture_step = (
            capture_before_goal.float() * progress_step
            + (~capture_before_goal).float() * self.max_episode_length
        )
        self.stats["first_capture_step"].copy_(
            torch.min(self.stats["first_capture_step"], current_capture_step)
        )
        self.stats["terminal_reward"].add_(terminal_reward.mean(-1).unsqueeze(-1))
        self.stats["goal_progress_reward"].add_(goal_progress_reward_val)
        self.stats["capture_progress_reward"].add_(capture_progress_reward_val)
        self.stats["catch_reward"].add_(catch_reward.mean(-1).unsqueeze(-1))
        self.stats["goal_penalty"].add_(goal_penalty_val.mean(-1).unsqueeze(-1))
        self.stats["speed_reward"].add_(speed_reward.mean(-1).unsqueeze(-1))
        self.stats["time_penalty"].add_(time_penalty.mean(-1).unsqueeze(-1))
        self.stats["coop_reward"].add_(coop_reward.mean(-1).unsqueeze(-1))
        self.stats["phi_team"].add_(self.phi_team.unsqueeze(-1))
        self.stats["phi_block"].add_(self.phi_block.unsqueeze(-1))
        self.stats["phi_pressure"].add_(self.phi_pressure.unsqueeze(-1))
        self.stats["phi_spread"].add_(self.phi_spread.unsqueeze(-1))
        self.stats["d_i_mean"].add_(self.d_i_clip.mean(dim=-1, keepdim=True))
        self.stats["d_i_min"].add_(self.d_i_clip.min(dim=-1, keepdim=True).values)
        urgent_block = ((self.g_urg > 0.5) & (self.phi_block > 0.6)).float().unsqueeze(-1)
        self.stats["urgent_block_rate"].add_(urgent_block)
        n_agents_ahead = (self.my_ahead_proj_goal > 0).float().sum(dim=1, keepdim=True)
        self.stats["n_agents_ahead_mean"].add_(n_agents_ahead)
        self.stats["collision_penalty"].add_(collision_penalty.mean(-1).unsqueeze(-1))
        self.stats["landed_penalty"].add_(landed_penalty.mean(-1).unsqueeze(-1))
        self.stats["smoothness_reward"].add_(smoothness_reward.mean(-1).unsqueeze(-1))
        self.stats["smoothness_mean"].add_(self.drone.throttle_difference.mean(-1).unsqueeze(-1))
        self.stats["smoothness_max"].copy_(
            torch.max(
                self.drone.throttle_difference.max(-1).values.unsqueeze(-1),
                self.stats["smoothness_max"],
            )
        )

        done = timeout | capture_before_goal | goal_reached | any_landed
        self._update_curriculum_metrics(done, capture_before_goal)

        ep_len = torch.clamp(self.progress_buf.unsqueeze(-1), min=1)
        averaged_stats = [
            "collision",
            "action_error_order1_mean",
            "target_predicted_error",
            "smoothness_mean",
            "smoothness_reward",
            "terminal_reward",
            "goal_progress_reward",
            "capture_progress_reward",
            "catch_reward",
            "goal_penalty",
            "speed_reward",
            "time_penalty",
            "coop_reward",
            "phi_team",
            "phi_block",
            "phi_pressure",
            "phi_spread",
            "d_i_mean",
            "d_i_min",
            "urgent_block_rate",
            "n_agents_ahead_mean",
            "collision_penalty",
            "collision_wall",
            "collision_floor",
            "collision_drone",
            "landed_penalty",
        ]
        for key in averaged_stats:
            self.stats[key].div_(torch.where(done, ep_len, torch.ones_like(ep_len)))

        self.stats["return"] += reward.mean(-1).unsqueeze(-1)

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1),
                },
                "done": done,
            },
            self.batch_size,
        )
        
    def _get_dummy_policy_prey(self):
        drone_pos, _ = self.get_env_poses(self.drone.get_world_poses(False))
        target_pos, _ = self.get_env_poses(self.target.get_world_poses())
        target_rpos = vmap(cpos)(drone_pos, target_pos)

        force = torch.zeros(self.num_envs, 1, 3, device=self.device)

        goal_center = self.goal_region_center.view(1, 1, 3)
        goal_closest = closest_point_on_cylinder(
            target_pos, goal_center, self.goal_region_radius, self.goal_region_height
        )
        goal_offset = goal_closest - target_pos
        goal_dist = torch.norm(goal_offset, dim=-1, keepdim=True)
        force_goal = self.target_goal_attraction_coef * goal_offset / (goal_dist + 1e-5)
        force += force_goal

        # pursuer repulsion
        dist_pos = torch.norm(target_rpos, dim=-1).squeeze(1).unsqueeze(-1)
        force_r_xy_direction = - target_rpos / (dist_pos + 1e-5)
        force_p = force_r_xy_direction * (1 / (dist_pos + 1e-5))
        force += torch.sum(force_p, dim=1)

        # region-boundary repulsion
        force_region = torch.zeros_like(force)
        target_origin_dist = torch.norm(target_pos[..., :2],dim=-1)
        force_r_xy_direction = - target_pos[..., :2] / (target_origin_dist.unsqueeze(-1) + 1e-5)
        out_of_arena = target_pos[..., 0]**2 + target_pos[..., 1]**2 > self.arena_size**2
        self.stats['out_of_arena'] = torch.logical_or(self.stats['out_of_arena'].bool(), out_of_arena).float()

        force_region[..., 0] = out_of_arena.float() * force_r_xy_direction[..., 0] * (1 / 1e-5) + \
            (~out_of_arena).float() * force_r_xy_direction[..., 0] * (1 / ((self.arena_size - target_origin_dist) + 1e-5))
        force_region[..., 1] = out_of_arena.float() * force_r_xy_direction[..., 1] * (1 / 1e-5) + \
            (~out_of_arena).float() * force_r_xy_direction[..., 1] * (1 / ((self.arena_size - target_origin_dist) + 1e-5))
        
        higher_than_z = (target_pos[..., 2] > self.max_height)
        force_region[...,2] = higher_than_z.float() * (-1 / 1e-5) + \
            (~higher_than_z).float() * - (self.max_height - target_pos[..., 2]) / ((self.max_height - target_pos[..., 2])**2 + 1e-5)
        lower_than_ground = (target_pos[..., 2] < 0.0)
        force_region[...,2] += (lower_than_ground.float() * (1 / 1e-5) + \
            (~lower_than_ground).float() * - (0.0 - target_pos[..., 2]) / ((0.0 - target_pos[..., 2])**2 + 1e-5))
        force += force_region

        return force.type(torch.float32)

    # visualize functions
    def _draw_court_circle(self):
        self.draw.clear_lines()

        arena_p1, arena_p2, arena_colors, arena_sizes = draw_court_circle(
            self.arena_size,
            self.max_height,
            color_edge=(0.82, 0.82, 0.82, 1.0),
            color_wall=(0.82, 0.82, 0.82, 1.0),
            line_size=3.0,
            wall_alpha=0.015,
        )

        goal_halo_edge = (1.0, 0.78, 0.28, 1.0)
        goal_halo_wall = (1.0, 0.74, 0.24, 1.0)
        halo_p1, halo_p2, halo_colors, halo_sizes = draw_court_circle(
            self.goal_region_radius + 0.06,
            self.goal_region_height,
            color_edge=goal_halo_edge,
            color_wall=goal_halo_wall,
            line_size=10.0,
            wall_alpha=0.18,
        )

        goal_edge = (1.0, 0.35, 0.12, 1.0)
        goal_wall = (1.0, 0.42, 0.14, 1.0)
        goal_p1, goal_p2, goal_colors, goal_sizes = draw_court_circle(
            self.goal_region_radius,
            self.goal_region_height,
            color_edge=goal_edge,
            color_wall=goal_wall,
            line_size=9.0,
            wall_alpha=0.42,
        )
        # `draw_court_circle` draws a cylinder spanning z in [0, H], while the
        # task logic treats `goal_region_center` as the midpoint with height H.
        # Shift the draw origin down by H/2 so the rendered guard zone matches
        # the actual in-cylinder test used for rewards/done conditions.
        goal_draw_origin_world = _carb_float3_add(
            self.central_env_pos,
            Float3(
                float(self.goal_region_center[0].item()),
                float(self.goal_region_center[1].item()),
                float((self.goal_region_center[2] - self.goal_region_height / 2.0).item()),
            ),
        )
        goal_center_world = _carb_float3_add(
            self.central_env_pos,
            Float3(
                float(self.goal_region_center[0].item()),
                float(self.goal_region_center[1].item()),
                float(self.goal_region_center[2].item()),
            ),
        )
        goal_top_world = Float3(
            goal_center_world.x,
            goal_center_world.y,
            float(goal_center_world.z + self.goal_region_height / 2.0),
        )
        beacon_top_world = Float3(
            goal_center_world.x,
            goal_center_world.y,
            float(self.central_env_pos.z + self.max_height + 0.8),
        )
        beacon_arm = float(self.goal_region_radius * 0.55)
        floor_marker_z = float(self.central_env_pos.z + 0.05)
        floor_arm = float(self.goal_region_radius * 1.2)

        arena_p1 = [
            _carb_float3_add(p, self.central_env_pos) for p in arena_p1
        ]
        arena_p2 = [
            _carb_float3_add(p, self.central_env_pos) for p in arena_p2
        ]
        halo_p1 = [
            _carb_float3_add(p, goal_draw_origin_world) for p in halo_p1
        ]
        halo_p2 = [
            _carb_float3_add(p, goal_draw_origin_world) for p in halo_p2
        ]
        goal_p1 = [
            _carb_float3_add(p, goal_draw_origin_world) for p in goal_p1
        ]
        goal_p2 = [
            _carb_float3_add(p, goal_draw_origin_world) for p in goal_p2
        ]
        beacon_p1 = [
            goal_top_world,
            Float3(goal_center_world.x - beacon_arm, goal_center_world.y, beacon_top_world.z),
            Float3(goal_center_world.x, goal_center_world.y - beacon_arm, beacon_top_world.z),
            Float3(goal_center_world.x - beacon_arm * 0.75, goal_center_world.y - beacon_arm * 0.75, beacon_top_world.z),
            Float3(goal_center_world.x - floor_arm, goal_center_world.y, floor_marker_z),
            Float3(goal_center_world.x, goal_center_world.y - floor_arm, floor_marker_z),
        ]
        beacon_p2 = [
            beacon_top_world,
            Float3(goal_center_world.x + beacon_arm, goal_center_world.y, beacon_top_world.z),
            Float3(goal_center_world.x, goal_center_world.y + beacon_arm, beacon_top_world.z),
            Float3(goal_center_world.x + beacon_arm * 0.75, goal_center_world.y + beacon_arm * 0.75, beacon_top_world.z),
            Float3(goal_center_world.x + floor_arm, goal_center_world.y, floor_marker_z),
            Float3(goal_center_world.x, goal_center_world.y + floor_arm, floor_marker_z),
        ]
        beacon_colors = [
            (1.0, 0.08, 0.02, 1.0),
            (1.0, 0.22, 0.08, 1.0),
            (1.0, 0.22, 0.08, 1.0),
            (1.0, 0.36, 0.10, 1.0),
            (1.0, 0.10, 0.02, 1.0),
            (1.0, 0.10, 0.02, 1.0),
        ]
        beacon_sizes = [14.0, 12.0, 12.0, 10.0, 10.0, 10.0]

        point_list_1 = arena_p1 + halo_p1 + goal_p1 + beacon_p1
        point_list_2 = arena_p2 + halo_p2 + goal_p2 + beacon_p2
        colors = arena_colors + halo_colors + goal_colors + beacon_colors
        sizes = arena_sizes + halo_sizes + goal_sizes + beacon_sizes
        self.draw.draw_lines(point_list_1, point_list_2, colors, sizes)

    def _draw_traj(self):
        drone_pos = self.drone_states[..., :3]
        drone_vel = self.drone.get_velocities()[..., :3]
        point_list1, point_list2, colors, sizes = draw_traj(
            drone_pos[self.central_env_idx, :], drone_vel[self.central_env_idx, :], dt=0.02, size=4.0
        )
        point_list1 = [
            _carb_float3_add(p, self.central_env_pos) for p in point_list1
        ]
        point_list2 = [
            _carb_float3_add(p, self.central_env_pos) for p in point_list2
        ]
        self.draw.draw_lines(point_list1, point_list2, colors, sizes)   
    
    def _draw_detection(self):
        self.draw.clear_points()

        # drone detection
        drone_pos = self.drone_states[..., :3]
        drone_ori = self.drone_states[..., 3:7]
        drone_xaxis = quat_axis(drone_ori, 0)
        drone_yaxis = quat_axis(drone_ori, 1)
        drone_zaxis = quat_axis(drone_ori, 2)
        drone_point_list, drone_colors, drone_sizes = draw_detection(
            pos=drone_pos[self.central_env_idx, :],
            xaxis=drone_xaxis[self.central_env_idx, 0, :],
            yaxis=drone_yaxis[self.central_env_idx, 0, :],
            zaxis=drone_zaxis[self.central_env_idx, 0, :],
            drange=self.drone_detect_radius,
        )

        # target detection
        target_pos, target_ori = self.get_env_poses(self.target.get_world_poses())
        target_xaxis = quat_axis(target_ori, 0)
        target_yaxis = quat_axis(target_ori, 1)
        target_zaxis = quat_axis(target_ori, 2)
        target_point_list, target_colors, target_sizes = draw_detection(
            pos=target_pos[self.central_env_idx, :],
            xaxis=target_xaxis[self.central_env_idx, 0, :],
            yaxis=target_yaxis[self.central_env_idx, 0, :],
            zaxis=target_zaxis[self.central_env_idx, 0, :],
            drange=self.target_detect_radius,
        )
        
        point_list = drone_point_list + target_point_list
        colors = drone_colors + target_colors
        sizes = drone_sizes + target_sizes
        point_list = [
            _carb_float3_add(p, self.central_env_pos) for p in point_list
        ]
        self.draw.draw_points(point_list, colors, sizes)

    def _draw_catch(self):
        self.draw.clear_points()
        # drone detection
        drone_pos = self.drone_states[..., :3]
        drone_ori = self.drone_states[..., 3:7]
        drone_xaxis = quat_axis(drone_ori, 0)
        drone_yaxis = quat_axis(drone_ori, 1)
        drone_zaxis = quat_axis(drone_ori, 2)
        # catch
        point_list, colors, sizes = draw_catch(
            pos=drone_pos[self.central_env_idx, :],
            xaxis=drone_xaxis[self.central_env_idx, 0, :],
            yaxis=drone_yaxis[self.central_env_idx, 0, :],
            zaxis=drone_zaxis[self.central_env_idx, 0, :],
            drange=self.catch_radius,
        )
        # predicted target
        if self.use_TP_net:
            for step in range(self.target_pos_predicted.shape[1]):
                point_list.append(Float3(self.target_pos_predicted[self.central_env_idx, step].cpu().numpy().tolist()))
                colors.append((1.0, 1.0, 0.0, 0.3))
                sizes.append(20.0)
        point_list = [
            _carb_float3_add(p, self.central_env_pos) for p in point_list
        ]
        # catch, green
        catch_mask = self.capture[self.central_env_idx].unsqueeze(1).expand(-1, 400).reshape(-1)
        for idx in range(len(catch_mask)):
            if catch_mask[idx]:
                colors[idx] = (0.0, 1.0, 0.0, 0.3)
        self.draw.draw_points(point_list, colors, sizes)
