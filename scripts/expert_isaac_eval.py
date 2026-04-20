"""
expert_isaac_eval.py
====================
在 Isaac Sim 中运行层次化专家策略，录制视频并评估泛化能力。

动作空间澄清
-----------
追逐无人机的 action_transform = "PIDrate"
  PIDRateController 接收: [omega_x, omega_y, omega_z, thrust]
    omega  ∈ [-1,1]  → body-frame 角速度 (rad/s, scaled)
    thrust ∈ [-1,1]  → (thrust+1)/2 * max_thrust_ratio * max_thrust
                       thrust=0 ≈ hover, thrust=1 = max, thrust=-1 = fall

专家策略输出 t + omega，再映射到训练时同一条 PIDRate 控制链。

用法
----
  bash scripts/expert_isaac_eval.sh tp_net 1.5 2 100
  bash scripts/expert_isaac_eval.sh oracle_pos 1.5 0 512
  bash scripts/expert_isaac_eval.sh oracle_next 1.5 0 512
"""

import argparse
import collections
import datetime
import itertools
import logging
import math
import os
import sys

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

# ──────────────────────────────────────────────────────────────
#  Parse extra args (before Hydra consumes argv)
# ──────────────────────────────────────────────────────────────
_extra_parser = argparse.ArgumentParser(add_help=False)
_extra_parser.add_argument("--pred_mode",   default="tp_net",
                            choices=["noise", "tp_net", "oracle_pos", "oracle_next"])
_extra_parser.add_argument("--strategy_variant", default="baseline",
                            choices=["baseline", "expert2"])
_extra_parser.add_argument("--forward_dir_mode", default="default",
                            choices=["default", "motion_biased", "motion_only"])
_extra_parser.add_argument("--enable_goal_mode", type=lambda x: x.lower() != "false",
                            default=None)
_extra_parser.add_argument("--enable_close_mode", type=lambda x: x.lower() != "false",
                            default=None)
_extra_parser.add_argument("--enable_rush_mode", type=lambda x: x.lower() != "false",
                            default=None)
_extra_parser.add_argument("--expert2_front_layout", default="symmetric",
                            choices=["symmetric", "staggered"],)
_extra_parser.add_argument("--expert_intercept_pred_step", type=int, default=5)
_extra_parser.add_argument("--expert_intercept_use_direct_pred", type=lambda x: x.lower() != "false",
                            default=False)
_extra_parser.add_argument("--tp_weight",   default="")
_extra_parser.add_argument("--n_video",     type=int, default=2)
_extra_parser.add_argument("--n_generic",   type=int, default=100)
_extra_parser.add_argument("--v_prey_test", type=float, default=1.5)
_extra_parser.add_argument("--v_prey_schedule", default="",
                            help="Comma-separated target-speed schedule cycled across generic waves.")
_extra_parser.add_argument("--v_drone_test", type=float, default=1.5)
_extra_parser.add_argument("--video_dir",   default="eval_videos/expert")
_extra_parser.add_argument("--video_seed_base", type=int, default=0)
_extra_parser.add_argument("--video_seed_list", default="",
                            help="Comma-separated explicit seeds for video recording. Overrides n_video/base stepping.")
_extra_parser.add_argument("--generic_seed_base", type=int, default=999)
_extra_parser.add_argument("--generic_batch_envs", type=int, default=1,
                            help="Parallel env count for generic no-video evaluation.")
_extra_parser.add_argument("--random_init", type=lambda x: x.lower() != "false",
                            default=True)
_extra_parser.add_argument("--episode_length", type=int, default=1200,
                            help="Environment episode length / timeout steps.")
_extra_parser.add_argument("--max_steps",   type=int, default=1200,
                            help="Hard runner cap; should normally match episode_length.")
_extra_parser.add_argument("--collect_success_dataset", type=lambda x: x.lower() == "true",
                            default=False)
_extra_parser.add_argument("--dataset_dir", default="expert_datasets")
_extra_parser.add_argument("--dataset_name", default="")
_extra_parser.add_argument("--min_success_steps", type=int, default=1)
_extra_parser.add_argument("--dataset_dtype", default="float16",
                            choices=["float16", "float32"])
_extra_parser.add_argument("--collect_tp_dataset", type=lambda x: x.lower() == "true",
                            default=False)
_extra_parser.add_argument("--tp_dataset_dir", default="tp_datasets")
_extra_parser.add_argument("--tp_dataset_name", default="")
_extra_parser.add_argument("--tp_dataset_dtype", default="float16",
                            choices=["float16", "float32"])
_extra_args, _remaining_argv = _extra_parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining_argv

logging.basicConfig(level=logging.INFO, format="%(asctime)s[%(levelname)s] %(message)s")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from expert_strategy_test import EnvCfg as SharedExpertCfg
from expert_strategy_test import ExpertPolicy as SharedExpertPolicy
from expert_strategy_test import TP_net as SharedTPNet
from expert_strategy_test import apply_strategy_defaults as apply_shared_strategy_defaults


# ──────────────────────────────────────────────────────────────
#  Utility helpers
# ──────────────────────────────────────────────────────────────
def safe_normalize(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    norm = v.norm(dim=-1, keepdim=True).clamp(min=eps)
    return v / norm


def clip_vector_norm(v: torch.Tensor, max_norm: float) -> torch.Tensor:
    n = v.norm(dim=-1, keepdim=True)
    return v * (n.clamp(max=max_norm) / n.clamp(min=1e-6))


def clip_vector_norm_per_row(v: torch.Tensor, max_norm) -> torch.Tensor:
    if not isinstance(max_norm, torch.Tensor):
        max_norm = torch.full(
            v.shape[:-1], float(max_norm), device=v.device, dtype=v.dtype
        )
    n = v.norm(dim=-1)
    scale = (max_norm / n.clamp(min=1e-6)).clamp(max=1.0)
    return v * scale.unsqueeze(-1)


def safe_atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.atanh(x.clamp(-1.0 + eps, 1.0 - eps))


def parse_float_schedule(value: str, default: float):
    values = []
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values or [float(default)]


def project_to_plane(v: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    return v - (v * normal).sum(dim=-1, keepdim=True) * normal


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


def compute_true_target_next(
    base_env,
    target_pos_w: torch.Tensor,
    target_vel_3: torch.Tensor,
):
    """Replicate HideAndSeek target update to get the next-step ground-truth target state."""
    forces_target = base_env._get_dummy_policy_prey()
    desired_acc = base_env.target_accel_limit * forces_target / (
        torch.norm(forces_target, dim=-1, keepdim=True) + 1e-5
    )
    accel_alpha = min(1.0, float(base_env.dt) / max(base_env.target_command_tau, float(base_env.dt)))
    next_target_acc_cmd = torch.lerp(base_env.target_acc_cmd, desired_acc, accel_alpha)
    next_target_vel = target_vel_3 + (
        next_target_acc_cmd - base_env.target_velocity_damping * target_vel_3
    ) * float(base_env.dt)
    next_target_vel = clip_vector_norm(next_target_vel, float(base_env.current_target_speed))
    next_target_pos = target_pos_w + next_target_vel * float(base_env.dt)
    return next_target_pos, next_target_vel


# ──────────────────────────────────────────────────────────────
#  TP-Net predictor (LSTM-based target trajectory prediction)
# ──────────────────────────────────────────────────────────────
class TPNetPredictor:
    """Wraps the trained LSTM trajectory-prediction network."""

    class _Model(nn.Module):
        def __init__(self, hidden=128, n_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(6, hidden, n_layers, batch_first=True)
            self.head  = nn.Sequential(
                nn.Linear(hidden, 64), nn.ReLU(),
                nn.Linear(64, 3),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1])

    def __init__(self, weight_path: str, device: str,
                 arena_size: float, max_height: float,
                 max_episode_length: int,
                 history_len: int = 10):
        self.device      = device
        self.arena_size  = arena_size
        self.max_height  = max_height
        self.hlen        = history_len
        self.model       = self._Model().to(device)
        self.model.load_state_dict(torch.load(weight_path, map_location=device))
        self.model.eval()
        self.buf: collections.deque = collections.deque(maxlen=history_len)

    def reset(self):
        self.buf.clear()

    @torch.no_grad()
    def predict(self, pos: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
        """pos, vel: [3]. Returns predicted next position [3]."""
        feat = torch.cat([
            pos / self.arena_size,
            vel / max(self.max_height, 1.0),
        ]).cpu()
        self.buf.append(feat)
        if len(self.buf) < self.hlen:
            return (pos + vel * 0.01).to(self.device)
        seq   = torch.stack(list(self.buf)).unsqueeze(0).to(self.device)
        delta = self.model(seq)[0]
        return (pos + delta * self.arena_size).clamp(
            -self.arena_size, self.arena_size
        )


# ──────────────────────────────────────────────────────────────
#  Expert Policy (Blocker + Flanker hierarchical strategy)
# ──────────────────────────────────────────────────────────────
class ExpertPolicy:
    """
    Hierarchical cooperative pursuit:
    - Blocker: intercepts the target on its predicted path
    - Flankers: surround the target from multiple angles
    """

    def __init__(
        self,
        pred_mode:    str,
        tp_predictor,
        v_max:        float = 1.0,
        catch_radius: float = 0.4,
        goal_center:  torch.Tensor = None,
        goal_radius:  float = 0.5,
        goal_height:  float = 0.4,
    ):
        self.pred_mode    = pred_mode
        self.tp           = tp_predictor
        self.v_max        = v_max
        self.catch_radius = catch_radius
        self.goal_center  = goal_center
        self.goal_radius  = goal_radius
        self.goal_height  = goal_height
        self.noise_std    = 0.05 * v_max

    def reset(self):
        if self.tp is not None:
            self.tp.reset()

    def _predict_target(self, step: int,
                        tp: torch.Tensor, tv: torch.Tensor) -> torch.Tensor:
        if self.pred_mode == "tp_net" and self.tp is not None:
            return self.tp.predict(tp[0], tv[0])
        if self.pred_mode == "oracle_pos":
            return tp[0].clone()
        if self.pred_mode == "oracle_next":
            return tp[0] + tv[0] * 0.1
        noise = torch.randn(3, device=tp.device) * self.noise_std
        return tp[0] + tv[0] * 0.1 + noise

    def get_vel_cmd(
        self,
        step:           int,
        drone_pos:      torch.Tensor,  # [n, 3]
        drone_vel:      torch.Tensor,  # [n, 3]
        target_pos:     torch.Tensor,  # [1, 3]
        target_vel:     torch.Tensor,  # [1, 3]
        target_next_pos: torch.Tensor, # [1, 3]  (noise mode hint)
    ) -> torch.Tensor:
        """Returns world-frame velocity commands [n, 3], norm ≤ v_max."""
        device  = drone_pos.device
        n       = drone_pos.shape[0]
        tp_pred = self._predict_target(step, target_pos, target_vel)

        # ── role assignment ────────────────────────────────────────
        # drone closest to the predicted intercept → Blocker
        tp_3 = tp_pred.unsqueeze(0)                       # [1, 3]
        dists = (drone_pos - tp_3).norm(dim=-1)           # [n]
        blocker_idx = int(dists.argmin().item())

        vel_cmds = []
        for i in range(n):
            if i == blocker_idx:
                # Blocker: rush toward predicted target position
                to_wp = tp_3[0] - drone_pos[i]
                spd   = self.v_max
            else:
                # Flanker: approach from angular offset to encircle
                flank_angle = (i * 2.0 * math.pi / max(1, n - 1)
                               if n > 1 else 0.0)
                offset = torch.tensor([
                    math.cos(flank_angle) * self.catch_radius * 1.5,
                    math.sin(flank_angle) * self.catch_radius * 1.5,
                    0.0,
                ], device=device)
                wp    = target_pos[0] + offset
                to_wp = wp - drone_pos[i]
                dist  = to_wp.norm()
                spd   = self.v_max * min(1.0, dist / (self.catch_radius * 3))

            to_wp_norm = safe_normalize(to_wp.unsqueeze(0))[0]
            v          = to_wp_norm * spd
            v          = clip_vector_norm(v.unsqueeze(0), self.v_max)[0]
            # altitude maintenance: keep at same height as target
            target_z   = float(target_pos[0, 2])
            v[2]       = (target_z - drone_pos[i, 2]) * 2.0          # z-PD
            v          = clip_vector_norm(v.unsqueeze(0), self.v_max)[0]
            vel_cmds.append(v)

        return torch.stack(vel_cmds)   # [n, 3]


class BatchedTPNetPredictor:
    """Batched TP-Net predictor for generic large-batch evaluation."""

    def __init__(self, weight_path: str, cfg: SharedExpertCfg, device: str):
        self.cfg = cfg
        self.device = device
        self.net = SharedTPNet(
            input_dim=1 + 3 + 3 + 3 * cfg.max_agents,
            output_dim=3 * cfg.future_predcition_step,
            future_predcition_step=cfg.future_predcition_step,
            window_step=cfg.window_step,
        ).to(device)
        state = torch.load(weight_path, map_location=device)
        self.net.load_state_dict(state)
        self.net.eval()
        self.position_scale = torch.tensor(
            [cfg.arena_size, cfg.arena_size, cfg.max_height], device=device
        )
        self.history = None

    def reset(self, batch_size: int):
        feat_dim = 1 + 3 + 3 + 3 * self.cfg.max_agents
        self.history = torch.zeros(
            batch_size,
            self.cfg.history_step,
            feat_dim,
            device=self.device,
        )
        self._initialized = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

    def _make_frame_batch(
        self,
        step: int,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor,
        drone_pos: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_agents, _ = drone_pos.shape
        step_norm = torch.full(
            (batch_size, 1),
            step / max(self.cfg.max_episode_length, 1),
            device=self.device,
            dtype=target_pos.dtype,
        )
        tp_norm = target_pos.squeeze(1) / self.position_scale
        tv_norm = target_vel.squeeze(1)
        dp_norm = drone_pos / self.position_scale.view(1, 1, 3)
        if n_agents < self.cfg.max_agents:
            pad = torch.full(
                (batch_size, self.cfg.max_agents - n_agents, 3),
                -5.0,
                device=self.device,
                dtype=dp_norm.dtype,
            )
            dp_norm = torch.cat([dp_norm, pad], dim=1)
        frame = torch.cat(
            [
                step_norm,
                tp_norm,
                tv_norm,
                dp_norm.reshape(batch_size, -1),
            ],
            dim=-1,
        )
        return frame

    @torch.no_grad()
    def predict_next_pos_batch(
        self,
        step: int,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor,
        drone_pos: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = target_pos.shape[0]
        if self.history is None or self.history.shape[0] != batch_size:
            self.reset(batch_size)

        frame = self._make_frame_batch(step, target_pos, target_vel, drone_pos)
        first_mask = ~self._initialized
        if bool(first_mask.any()):
            self.history[first_mask] = frame[first_mask].unsqueeze(1).expand(
                -1, self.cfg.history_step, -1
            )
            self._initialized[first_mask] = True
        if bool((~first_mask).any()):
            active = ~first_mask
            self.history[active] = torch.roll(self.history[active], shifts=-1, dims=1)
            self.history[active, -1] = frame[active]

        out = self.net(self.history).reshape(batch_size, self.cfg.future_predcition_step, 3)
        pred_idx = max(
            0,
            min(
                int(getattr(self.cfg, "expert_intercept_pred_step", 5)) - 1,
                int(self.cfg.future_predcition_step) - 1,
            ),
        )
        pred = out[:, pred_idx].clone()
        pred[:, :2] = pred[:, :2] * self.cfg.arena_size
        pred[:, 2] = (pred[:, 2] + 1.0) / 2.0 * self.cfg.max_height
        return pred


class BatchedExpertPolicy:
    """
    Fast batched expert used by no-render generic evaluation.

    The physics already runs batched in Isaac; this class removes the
    per-environment Python loop from expert decision-making.
    """

    def __init__(
        self,
        cfg: SharedExpertCfg,
        pred_mode: str = "noise",
        tp_weight_path: str = "",
        tp_device: str = "cpu",
    ):
        self.cfg = cfg
        self.pred_mode = pred_mode
        self.device = torch.device(tp_device)
        self.permutations = torch.tensor(
            list(itertools.permutations(range(cfg.num_agents))),
            device=self.device,
            dtype=torch.long,
        )
        self.prev_assignment = None
        if pred_mode == "tp_net":
            assert os.path.isfile(tp_weight_path), f"TP weight not found: {tp_weight_path}"
            self.tp_predictor = BatchedTPNetPredictor(tp_weight_path, cfg, str(self.device))
        else:
            self.tp_predictor = None

    def reset(self, batch_size: int):
        self.prev_assignment = None
        if self.tp_predictor is not None:
            self.tp_predictor.reset(batch_size)

    def predict_target_batch(
        self,
        step: int,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor,
        drone_pos: torch.Tensor,
        target_next_pos: torch.Tensor,
    ) -> torch.Tensor:
        if self.pred_mode == "tp_net":
            return self.tp_predictor.predict_next_pos_batch(
                step, target_pos, target_vel, drone_pos
            )
        if self.pred_mode == "oracle_pos":
            return target_pos.squeeze(1).clone()
        if self.pred_mode == "oracle_next":
            return target_next_pos.squeeze(1).clone()
        batch_size = target_pos.shape[0]
        noise_mag = torch.rand(batch_size, 1, device=target_pos.device, dtype=target_pos.dtype) * 0.5
        direction = torch.randn(batch_size, 3, device=target_pos.device, dtype=target_pos.dtype)
        direction = safe_normalize(direction)
        return target_next_pos.squeeze(1) + noise_mag * direction

    def _build_formation_targets_batch(
        self,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor,
        target_pos_pred: torch.Tensor,
    ):
        goal_pos = self.cfg.goal_region_center.to(target_pos.device, target_pos.dtype).unsqueeze(0)
        goal_vec = goal_pos - target_pos
        goal_dist = goal_vec.norm(dim=-1)
        goal_dir = safe_normalize(goal_vec)
        lookahead_time = torch.where(
            goal_dist > 1.4,
            torch.full_like(goal_dist, 0.18),
            torch.full_like(goal_dist, 0.10),
        )
        if bool(getattr(self.cfg, "expert_intercept_use_direct_pred", False)):
            intercept_seed = target_pos_pred
        else:
            intercept_seed = target_pos_pred + target_vel * lookahead_time.unsqueeze(-1)

        motion_hint = target_vel.clone()
        motion_norm = motion_hint.norm(dim=-1)
        need_intercept = motion_norm < 1e-4
        motion_hint = torch.where(
            need_intercept.unsqueeze(-1),
            intercept_seed - target_pos,
            motion_hint,
        )
        motion_norm = motion_hint.norm(dim=-1)
        motion_hint = torch.where(
            (motion_norm < 1e-4).unsqueeze(-1),
            goal_dir,
            motion_hint,
        )
        motion_dir = safe_normalize(motion_hint)

        trap_mode = goal_dist < 1.55
        goal_emergency = goal_dist < 1.10
        if self.cfg.strategy_variant == "expert2" and not bool(self.cfg.enable_goal_mode):
            goal_emergency = torch.zeros_like(goal_emergency, dtype=torch.bool)

        if self.cfg.strategy_variant == "expert2" and not bool(self.cfg.enable_goal_mode):
            forward_dir = motion_dir
        elif self.cfg.forward_dir_mode == "motion_only":
            forward_dir = motion_dir
        else:
            if self.cfg.forward_dir_mode == "motion_biased":
                if self.cfg.strategy_variant == "expert2":
                    goal_weight = torch.where(
                        goal_emergency,
                        torch.full_like(goal_dist, 0.80),
                        torch.full_like(goal_dist, 0.45),
                    )
                else:
                    goal_weight = torch.where(
                        goal_emergency,
                        torch.full_like(goal_dist, 0.80),
                        torch.where(
                            trap_mode,
                            torch.full_like(goal_dist, 0.60),
                            torch.full_like(goal_dist, 0.45),
                        ),
                    )
            else:
                if self.cfg.strategy_variant == "expert2":
                    goal_weight = torch.where(
                        goal_emergency,
                        torch.full_like(goal_dist, 0.92),
                        torch.full_like(goal_dist, 0.70),
                    )
                else:
                    goal_weight = torch.where(
                        goal_emergency,
                        torch.full_like(goal_dist, 0.92),
                        torch.where(
                            trap_mode,
                            torch.full_like(goal_dist, 0.80),
                            torch.full_like(goal_dist, 0.70),
                        ),
                    )
            forward_dir = safe_normalize(
                goal_weight.unsqueeze(-1) * goal_dir + (1 - goal_weight).unsqueeze(-1) * motion_dir
            )

        up = torch.tensor([0.0, 0.0, 1.0], device=target_pos.device, dtype=target_pos.dtype).expand_as(forward_dir)
        lateral = torch.linalg.cross(up, forward_dir)
        lateral = torch.where(
            (lateral.norm(dim=-1, keepdim=True) < 1e-5),
            torch.tensor([1.0, 0.0, 0.0], device=target_pos.device, dtype=target_pos.dtype).expand_as(lateral),
            lateral,
        )
        lateral = safe_normalize(lateral)
        if self.cfg.strategy_variant == "expert2":
            rear_offset = torch.where(
                goal_emergency,
                torch.zeros_like(goal_dist),
                torch.full_like(goal_dist, -float(self.cfg.expert2_rear_back)),
            )
            rear = intercept_seed + forward_dir * rear_offset.unsqueeze(-1)
            if self.cfg.expert2_front_layout == "staggered":
                left = intercept_seed
                right = (
                    intercept_seed
                    + forward_dir * float(self.cfg.expert2_front_lead_forward)
                    + lateral * float(self.cfg.expert2_front_lead_side)
                )
            else:
                left = intercept_seed + lateral * float(self.cfg.expert2_side_width)
                right = intercept_seed - lateral * float(self.cfg.expert2_side_width)
            anchors = torch.stack([rear, left, right], dim=1)
            trap_mode = torch.zeros_like(goal_dist, dtype=torch.bool)
            return anchors, forward_dir, trap_mode

        front_cap = torch.clamp(goal_dist - self.cfg.goal_region_radius - 0.12, min=0.45)
        base_front = torch.minimum(
            front_cap,
            torch.where(
                trap_mode,
                torch.full_like(front_cap, 0.62),
                torch.full_like(front_cap, 1.00),
            ),
        )
        side_front = torch.where(
            trap_mode,
            torch.minimum(
                front_cap,
                torch.where(
                    goal_emergency,
                    torch.full_like(front_cap, 0.32),
                    torch.full_like(front_cap, 0.18),
                ),
            ),
            torch.full_like(front_cap, -0.10),
        )
        side_width = torch.where(
            trap_mode,
            torch.where(
                goal_emergency,
                torch.full_like(front_cap, 0.28),
                torch.full_like(front_cap, 0.34),
            ),
            torch.full_like(front_cap, 0.64),
        )

        center = intercept_seed + forward_dir * base_front.unsqueeze(-1)
        left = intercept_seed + forward_dir * side_front.unsqueeze(-1) + lateral * side_width.unsqueeze(-1)
        right = intercept_seed + forward_dir * side_front.unsqueeze(-1) - lateral * side_width.unsqueeze(-1)
        anchors = torch.stack([center, left, right], dim=1)
        return anchors, forward_dir, trap_mode

    def _assign_anchors_batch(
        self,
        drone_pos: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_agents, _ = drone_pos.shape
        dist = torch.cdist(drone_pos, anchors)
        perm_count = self.permutations.shape[0]
        perm = self.permutations.unsqueeze(0).expand(batch_size, -1, -1)
        dist_expanded = dist.unsqueeze(1).expand(-1, perm_count, -1, -1)
        gather_idx = perm.unsqueeze(-1)
        assigned = torch.gather(dist_expanded, 3, gather_idx).squeeze(-1)
        cost = assigned.sum(dim=-1)
        if self.prev_assignment is not None and self.prev_assignment.shape[0] == batch_size:
            changed = (perm != self.prev_assignment.unsqueeze(1)).float().sum(dim=-1)
            cost = cost + 0.20 * changed
        best_idx = cost.argmin(dim=-1)
        batch_idx = torch.arange(batch_size, device=drone_pos.device)
        best_perm = perm[batch_idx, best_idx]
        self.prev_assignment = best_perm.clone()
        return best_perm

    def _vel_to_t_omega_batch(
        self,
        desired_vel: torch.Tensor,
        current_vel: torch.Tensor,
        current_quat: torch.Tensor,
        heading_hint_world: torch.Tensor,
    ):
        desired_vel = clip_vector_norm_per_row(desired_vel, self.cfg.v_drone)
        desired_speed = desired_vel.norm(dim=-1)
        zero_mask = desired_speed < 1e-6

        vel_error = desired_vel - current_vel
        acc_cmd = 3.2 * vel_error
        # Clamp z first, then couple xy limit to max tilt angle
        acc_cmd[:, 2] = acc_cmd[:, 2].clamp(-3.5, 3.5)
        # Coupled xy limit: ||a_xy|| ≤ (g + a_z) * tan(θ_max)
        # Ensures thrust direction never exceeds θ_max tilt
        max_tilt_tan = math.tan(math.radians(25))  # θ_max = 25°
        a_xy_max = ((self.cfg.gravity + acc_cmd[:, 2]) * max_tilt_tan).clamp(min=0.1)
        acc_cmd[:, :2] = clip_vector_norm_per_row(acc_cmd[:, :2], a_xy_max)

        e3 = torch.tensor([0.0, 0.0, 1.0], device=desired_vel.device, dtype=desired_vel.dtype).unsqueeze(0)
        thrust_world = acc_cmd + self.cfg.gravity * e3
        thrust_norm = thrust_world.norm(dim=-1).clamp(min=1e-6)
        z_des = thrust_world / thrust_norm.unsqueeze(-1)
        t = self.cfg.hover_thrust_ratio * (thrust_norm / self.cfg.gravity)
        t = t.clamp(self.cfg.min_thrust_ratio, self.cfg.max_thrust_ratio)

        basis_x = torch.tensor([1.0, 0.0, 0.0], device=desired_vel.device, dtype=desired_vel.dtype).unsqueeze(0).expand_as(z_des)
        basis_y = torch.tensor([0.0, 1.0, 0.0], device=desired_vel.device, dtype=desired_vel.dtype).unsqueeze(0).expand_as(z_des)
        basis_z = torch.tensor([0.0, 0.0, 1.0], device=desired_vel.device, dtype=desired_vel.dtype).unsqueeze(0).expand_as(z_des)

        x_cur = quat_rotate(current_quat, basis_x)
        y_cur = quat_rotate(current_quat, basis_y)
        z_cur = quat_rotate(current_quat, basis_z)

        x_ref = project_to_plane(x_cur, z_des)
        x_ref = torch.where(
            (x_ref.norm(dim=-1, keepdim=True) < 1e-5),
            basis_x,
            x_ref,
        )
        hint = project_to_plane(heading_hint_world, z_des)
        valid_hint = hint.norm(dim=-1, keepdim=True) > 1e-5
        x_ref = torch.where(valid_hint, 0.75 * x_ref + 0.25 * hint, x_ref)

        x_des = safe_normalize(x_ref)
        y_des = torch.linalg.cross(z_des, x_des)
        y_des = torch.where(
            (y_des.norm(dim=-1, keepdim=True) < 1e-5),
            basis_y,
            y_des,
        )
        y_des = safe_normalize(y_des)
        x_des = safe_normalize(torch.linalg.cross(y_des, z_des))

        e_rot = 0.5 * (
            torch.linalg.cross(x_cur, x_des)
            + torch.linalg.cross(y_cur, y_des)
            + torch.linalg.cross(z_cur, z_des)
        )
        omega_world = 2.0 * e_rot
        omega_body = quat_rotate_inverse(current_quat, omega_world)
        omega_body = clip_vector_norm_per_row(
            omega_body,
            torch.full(
                (desired_vel.shape[0],),
                0.65 * self.cfg.max_body_rate_rad_s,
                device=desired_vel.device,
                dtype=desired_vel.dtype,
            ),
        )
        omega_body = torch.nan_to_num(omega_body)
        t = torch.nan_to_num(
            t,
            nan=float(self.cfg.hover_thrust_ratio),
        )
        if bool(zero_mask.any()):
            omega_body[zero_mask] = 0.0
            t[zero_mask] = float(self.cfg.hover_thrust_ratio)
        return t, omega_body, desired_vel

    def t_omega_to_pidrate_raw(
        self,
        t: torch.Tensor,
        omega: torch.Tensor,
    ) -> torch.Tensor:
        action = self.t_omega_to_pidrate_action(t, omega)
        return safe_atanh(action)

    def t_omega_to_pidrate_action(
        self,
        t: torch.Tensor,
        omega: torch.Tensor,
    ) -> torch.Tensor:
        rate_norm = (omega / max(self.cfg.max_body_rate_rad_s, 1e-6)).clamp(-0.999, 0.999)
        thrust_norm = (2.0 * t - 1.0).clamp(-0.999, 0.999).unsqueeze(-1)
        return torch.cat([rate_norm, thrust_norm], dim=-1)

    @torch.no_grad()
    def get_actions_batch(
        self,
        step: int,
        drone_pos: torch.Tensor,
        drone_vel: torch.Tensor,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor,
        target_next_pos: torch.Tensor,
        drone_quat: torch.Tensor,
        return_debug: bool = False,
    ):
        batch_size, n_agents, _ = drone_pos.shape
        target_pos_flat = target_pos.squeeze(1)
        target_vel_flat = target_vel.squeeze(1)
        goal_pos = self.cfg.goal_region_center.to(target_pos.device, target_pos.dtype).unsqueeze(0)

        target_pos_pred = self.predict_target_batch(
            step, target_pos, target_vel, drone_pos, target_next_pos
        )
        anchors, forward_dir, trap_mode = self._build_formation_targets_batch(
            target_pos_flat, target_vel_flat, target_pos_pred
        )
        assignment = self._assign_anchors_batch(drone_pos, anchors)

        dist_to_target_all = torch.norm(
            drone_pos - target_pos_flat.unsqueeze(1), dim=-1
        )  # [batch, n_agents]
        min_target_dist = dist_to_target_all.min(dim=-1).values  # [batch]
        goal_dist = torch.norm(goal_pos - target_pos_flat, dim=-1)
        rush_idx = dist_to_target_all.argmin(dim=-1)  # [batch] index of closest drone
        if bool(self.cfg.enable_rush_mode):
            rush_mode = min_target_dist < self.cfg.catch_radius * 1.8
        else:
            rush_mode = torch.zeros_like(min_target_dist, dtype=torch.bool)

        t_list = []
        omega_list = []
        vel_cmd_list = []
        waypoint_list = []
        batch_idx = torch.arange(batch_size, device=drone_pos.device)

        for agent_idx in range(n_agents):
            is_rush_agent = rush_mode & (rush_idx == agent_idx)  # [batch] bool

            anchor_idx = assignment[:, agent_idx]
            waypoint = anchors[batch_idx, anchor_idx]
            pos_i = drone_pos[:, agent_idx]
            vel_i = drone_vel[:, agent_idx]
            quat_i = drone_quat[:, agent_idx]
            to_wp = waypoint - pos_i
            dist_wp = to_wp.norm(dim=-1)

            is_expert2 = self.cfg.strategy_variant == "expert2"
            is_chaser = is_expert2 & (anchor_idx == 0)
            is_front_interceptor = is_expert2 & (anchor_idx != 0)

            # ── Normal formation logic ──
            if self.cfg.strategy_variant == "expert2":
                forward_bias = torch.where(
                    is_chaser,
                    torch.full_like(dist_wp, -0.08),
                    torch.full_like(dist_wp, 0.10),
                ).to(to_wp.dtype)
                feedforward = target_vel_flat + forward_bias.unsqueeze(-1) * forward_dir
                kp = torch.where(
                    is_chaser,
                    torch.full_like(dist_wp, 1.70),
                    torch.full_like(dist_wp, 1.80),
                ).to(to_wp.dtype)
            else:
                forward_bias = torch.where(anchor_idx == 0, 0.22, -0.05).to(to_wp.dtype)
                feedforward = target_vel_flat + forward_bias.unsqueeze(-1) * forward_dir
                kp = torch.where(anchor_idx == 0, 1.90, 1.65).to(to_wp.dtype)
                kp = kp + trap_mode.to(to_wp.dtype) * 0.20
            v_des = kp.unsqueeze(-1) * to_wp + 0.85 * feedforward - 0.26 * vel_i

            if bool(self.cfg.enable_close_mode):
                if self.cfg.strategy_variant == "expert2":
                    target_close_threshold = torch.where(
                        is_chaser,
                        torch.full_like(dist_wp, float(self.cfg.expert2_chaser_close_target_threshold)),
                        torch.full_like(dist_wp, float(self.cfg.expert2_front_close_target_threshold)),
                    )
                    close_mode = (dist_wp < 0.45) | (dist_to_target_all[:, agent_idx] < target_close_threshold)
                else:
                    close_mode = (dist_wp < 0.45) | (min_target_dist < 0.95)
            else:
                close_mode = torch.zeros_like(dist_wp, dtype=torch.bool)
            desired_height = target_pos_pred[:, 2] + torch.where(
                close_mode,
                torch.full_like(dist_wp, 0.10),
                torch.full_like(dist_wp, 0.05),
            )
            desired_height = desired_height.clamp(1.2, self.cfg.max_height - 0.4)
            v_des[:, 2] = 0.90 * (desired_height - pos_i[:, 2]) - 0.45 * vel_i[:, 2]

            if self.cfg.strategy_variant != "expert2":
                frontal_wp = target_pos_pred + 0.95 * forward_dir
                v_frontal = 1.60 * (frontal_wp - pos_i) + 0.70 * feedforward - 0.35 * vel_i
                v_frontal[:, 2] = 0.95 * (desired_height - pos_i[:, 2]) - 0.45 * vel_i[:, 2]
                frontal_mask = (goal_dist < 1.4) & (anchor_idx == 0) & (dist_wp > 0.45)
                v_des = torch.where(frontal_mask.unsqueeze(-1), v_frontal, v_des)

            side_axis = project_to_plane(waypoint - target_pos_flat, forward_dir)
            side_axis = torch.where(
                (side_axis.norm(dim=-1, keepdim=True) > 1e-5),
                safe_normalize(side_axis),
                torch.zeros_like(side_axis),
            )
            if self.cfg.strategy_variant == "expert2":
                close_wp_chaser = target_pos_pred - float(self.cfg.expert2_chaser_close_back) * forward_dir
                close_wp_interceptor = (
                    target_pos_pred
                    + float(self.cfg.expert2_front_close_forward) * forward_dir
                    + side_axis * float(self.cfg.expert2_front_close_side)
                )
                close_wp = torch.where(
                    is_chaser.unsqueeze(-1),
                    close_wp_chaser,
                    close_wp_interceptor,
                )
                inward_gain = torch.where(
                    is_chaser,
                    torch.full_like(dist_wp, float(self.cfg.expert2_chaser_inward_gain)),
                    torch.full_like(dist_wp, float(self.cfg.expert2_front_inward_gain)),
                ).to(to_wp.dtype)
            else:
                close_wp_block = target_pos_pred + 0.03 * forward_dir
                close_wp_flank = (
                    target_pos_pred
                    - 0.03 * forward_dir
                    + side_axis
                    * torch.where(
                        trap_mode,
                        torch.full_like(goal_dist, 0.12),
                        torch.full_like(goal_dist, 0.18),
                    ).unsqueeze(-1)
                )
                close_wp = torch.where(
                    (anchor_idx == 0).unsqueeze(-1),
                    close_wp_block,
                    close_wp_flank,
                )
                inward_gain = torch.where(anchor_idx == 0, 0.18, 0.10).to(to_wp.dtype)
            rush_close_wp = target_pos_pred.clone()
            close_wp = torch.where(is_rush_agent.unsqueeze(-1), rush_close_wp, close_wp)
            inward_gain = torch.where(is_rush_agent, 0.35, inward_gain)
            inward = safe_normalize(target_pos_pred - pos_i)
            v_close = (
                2.05 * (close_wp - pos_i)
                + 0.92 * target_vel_flat
                + inward_gain.unsqueeze(-1) * inward * self.cfg.v_drone
                - 0.34 * vel_i
            )
            v_close[:, 2] = 0.90 * (target_pos_pred[:, 2] - pos_i[:, 2]) - 0.55 * vel_i[:, 2]
            v_des = torch.where(close_mode.unsqueeze(-1), v_close, v_des)

            # Speed cap: all drones use full v_drone (removed 0.96 flanker cap)
            v_des = clip_vector_norm_per_row(v_des, self.cfg.v_drone)

            # ── Drone-Drone collision avoidance (repulsive velocity correction) ──
            safe_dist = torch.full_like(dist_wp, 4.0 * self.cfg.collision_radius)
            repulse_gain = torch.full_like(dist_wp, 0.5)
            if self.cfg.strategy_variant == "expert2":
                safe_dist = torch.where(
                    close_mode,
                    torch.full_like(dist_wp, float(self.cfg.expert2_close_safe_dist)),
                    safe_dist,
                )
                repulse_gain = torch.where(
                    close_mode,
                    torch.full_like(dist_wp, float(self.cfg.expert2_close_repulse_gain)),
                    repulse_gain,
                )
            for other_idx in range(n_agents):
                if other_idx == agent_idx:
                    continue
                sep = pos_i - drone_pos[:, other_idx]  # [batch, 3]
                d = sep.norm(dim=-1)  # [batch]
                too_close = d < safe_dist
                if not bool(too_close.any()):
                    continue
                repulse_dir = safe_normalize(sep)
                repulse_mag = self.cfg.v_drone * repulse_gain * (1.0 - d / safe_dist.clamp(min=1e-6))
                repulse_mag = repulse_mag.clamp(min=0.0)
                if self.cfg.strategy_variant == "expert2":
                    rel_vel = vel_i - drone_vel[:, other_idx]
                    closing_speed = (-(rel_vel * repulse_dir).sum(dim=-1)).clamp(min=0.0)
                    closing_boost = 1.0 + float(self.cfg.expert2_close_repulse_closing_boost) * torch.clamp(
                        closing_speed / max(self.cfg.v_drone, 1e-6), min=0.0, max=1.0
                    )
                    repulse_mag = torch.where(close_mode, repulse_mag * closing_boost, repulse_mag)
                v_des = v_des + (too_close.float() * repulse_mag).unsqueeze(-1) * repulse_dir
                v_des = clip_vector_norm_per_row(v_des, self.cfg.v_drone)

            # ── Universal floor guard: prevent ground contact ──
            # Escalate to a guaranteed climb inside the low-altitude band.
            floor_safe_z = self.cfg.low_altitude_guard_z
            cur_z = pos_i[:, 2]  # [batch]
            low_mask = cur_z < floor_safe_z
            if bool(low_mask.any()):
                floor_correction = 3.0 * (floor_safe_z - cur_z)
                v_des[:, 2] = torch.where(
                    low_mask,
                    torch.max(v_des[:, 2], floor_correction),
                    v_des[:, 2],
                )
                v_des = clip_vector_norm_per_row(v_des, self.cfg.v_drone)

            # ── heading hint: rush drone looks at target, others look at waypoint ──
            heading_hint = torch.where(is_rush_agent.unsqueeze(-1), target_pos_pred - pos_i, to_wp)

            ti, oi, vi = self._vel_to_t_omega_batch(
                v_des,
                vel_i,
                quat_i,
                heading_hint,
            )
            t_list.append(ti)
            omega_list.append(oi)
            vel_cmd_list.append(vi)
            waypoint_list.append(waypoint)

        outputs = (
            torch.stack(t_list, dim=1),
            torch.stack(omega_list, dim=1),
            torch.stack(vel_cmd_list, dim=1),
        )
        if not return_debug:
            return outputs

        debug = {
            "assignment": assignment.detach().clone(),
            "waypoint": torch.stack(waypoint_list, dim=1).detach().clone(),
            "target_pos_pred": target_pos_pred.detach().clone(),
            "forward_dir": forward_dir.detach().clone(),
            "trap_mode": trap_mode.detach().clone(),
        }
        return outputs + (debug,)


# ──────────────────────────────────────────────────────────────
#  Velocity → PIDrate action converter
#
#  PIDRateController action format (transforms.py line 433-444):
#    action = [omega_x, omega_y, omega_z, thrust]
#    omega in [-1,1] → body-frame angular rate (scaled by 180*target_clip deg/s)
#    thrust in [-1,1]:
#      (thrust+1)/2 * max_thrust_ratio * max_thrust_total
#      thrust ≈ 0  → hover  (≈ 50% max thrust)
#      thrust = 1  → full thrust up
#      thrust = -1 → min thrust (drone falls)
# ──────────────────────────────────────────────────────────────
def vel_to_pidrate(
    vel_cmd:   torch.Tensor,   # [n, 3]  desired world-frame velocity
    drone_vel: torch.Tensor,   # [n, 3]  current world-frame velocity
    v_max:     float = 1.0,
) -> torch.Tensor:
    """
    Convert expert velocity command → PIDrate action [omega(3), thrust(1)].

    omega:  cross product of current and desired velocity directions
            gives the body rotation needed to align with the command.
            Scaled to [-1,1] with a proportional gain.

    thrust: proportional to desired speed, offset so that thrust≈0 at hover.
            Vertical velocity component adds an extra correction.
    """
    n = vel_cmd.shape[0]
    device = vel_cmd.device

    # ── angular velocity: direction alignment ─────────────────
    v_des_hat = safe_normalize(vel_cmd)                   # [n, 3]
    v_cur_hat = safe_normalize(drone_vel)                 # [n, 3]

    # cross product: rotation axis to align current → desired
    omega_raw = torch.cross(v_cur_hat, v_des_hat, dim=-1)  # [n, 3]
    # scale: gain chosen so that 90° misalignment → omega ≈ 0.7
    omega_gain = 0.7
    omega = (omega_raw * omega_gain).clamp(-1.0, 1.0)     # [n, 3]

    # ── thrust: speed-proportional + hover baseline ───────────
    # t ∈ [0,1]: normalized speed ratio
    t = vel_cmd.norm(dim=-1) / (v_max + 1e-6)             # [n]
    # vertical component adds correction (positive z_vel → more thrust)
    z_correction = (vel_cmd[:, 2] / (v_max + 1e-6)) * 0.3
    # center at hover: thrust=0 is hover in PIDrate space
    # we scale t from [0,1] to [0, 0.8] then center at 0 with z correction
    thrust = (t * 0.8 + z_correction).clamp(-0.9, 1.0)   # [n]

    return torch.cat([omega, thrust.unsqueeze(-1)], dim=-1)  # [n, 4]


# ──────────────────────────────────────────────────────────────
#  Video save helper
# ──────────────────────────────────────────────────────────────
def save_video(frames, fps: int, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    try:
        import imageio
        imageio.mimwrite(path, frames, fps=fps, quality=8)
        logging.info(f"Video saved: {path}")
        return path
    except Exception as exc:
        npy = path.replace(".mp4", ".npy")
        np.save(npy, np.stack(frames))
        logging.warning(f"imageio failed ({exc}); frames saved to {npy}")
        return npy


def _tuple_key_to_str(key) -> str:
    if isinstance(key, tuple):
        return "/".join(str(k) for k in key)
    return str(key)


def _storage_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _obs_to_storage_dict(obs_td, dtype: torch.dtype):
    obs_cpu = obs_td.detach().cpu()
    stored = {}
    for key in obs_cpu.keys(True, True):
        value = obs_cpu.get(key)
        if torch.is_floating_point(value):
            value = value.to(dtype=dtype)
        stored[_tuple_key_to_str(key)] = value.contiguous()
    return stored


def _extract_env_aligned_aux_from_obs(obs_td, future_steps: int = 5):
    """Build auxiliary labels directly from env observation tensors."""
    state_self = obs_td["state_self"].squeeze(-2)       # [N, A, 23]
    cooperation = obs_td["cooperation"].squeeze(-2)     # [N, A, C]

    assigned_wp = state_self[..., 13:16]
    close_wp = state_self[..., 16:19]
    close_trigger = state_self[..., 19:20] > 0.5
    active_wp = torch.where(close_trigger, close_wp, assigned_wp)

    role_onehot = state_self[..., 20:23]
    assignment = role_onehot.argmax(dim=-1).to(torch.long)

    pred_start = 3 + max(0, int(future_steps) - 1) * 3
    pred_end = pred_start + 3
    target_pos_pred = cooperation[..., pred_start:pred_end][:, 0, :]
    forward_dir = cooperation[..., 18:21][:, 0, :]
    trap_mode = torch.zeros(
        state_self.shape[0], dtype=torch.bool, device=state_self.device
    )

    return {
        "assignment": assignment,
        "waypoint": active_wp,
        "target_pos_pred": target_pos_pred,
        "forward_dir": forward_dir,
        "trap_mode": trap_mode,
    }


def _build_tp_dataset_chunk(
    tp_input_steps,
    tp_groundtruth_steps,
    tp_done_steps,
    env_done_steps,
    *,
    future_step: int,
    window_step: int,
    storage_dtype: torch.dtype,
):
    if not tp_input_steps or not tp_groundtruth_steps:
        return None

    tp_inputs = torch.stack(tp_input_steps, dim=1)  # [N, T, H, D]
    tp_groundtruth = torch.stack(tp_groundtruth_steps, dim=1)  # [N, T, 3]
    tp_done = torch.stack(tp_done_steps, dim=1).reshape(tp_inputs.shape[0], -1).bool()
    env_done = torch.stack(env_done_steps, dim=1).reshape(tp_inputs.shape[0], -1).bool()

    total_steps = int(tp_groundtruth.shape[1])
    if total_steps <= int(future_step):
        return {
            "TP_input": torch.empty(
                0,
                tp_inputs.shape[2],
                tp_inputs.shape[3],
                dtype=storage_dtype,
            ),
            "TP_future": torch.empty(0, future_step, tp_groundtruth.shape[-1], dtype=storage_dtype),
            "num_samples": 0,
            "episode_steps": total_steps,
        }

    windows = (
        tp_groundtruth.unfold(dimension=1, size=future_step + 1, step=window_step)
        .transpose(2, 3)[:, :, 1:]
    )
    window_count = int(windows.shape[1])

    timeout_valid = tp_done[:, :window_count]
    continuity_windows = env_done.unfold(
        dimension=1, size=future_step, step=window_step
    )[:, :window_count]
    continuity_valid = ~continuity_windows.any(dim=-1)
    valid_windows = timeout_valid & continuity_valid

    flat_mask = valid_windows.reshape(-1)
    flat_inputs = tp_inputs[:, :window_count].reshape(-1, *tp_inputs.shape[2:])
    flat_windows = windows.reshape(-1, future_step, tp_groundtruth.shape[-1])

    selected_inputs = flat_inputs[flat_mask].detach().cpu()
    selected_windows = flat_windows[flat_mask].detach().cpu()
    if torch.is_floating_point(selected_inputs):
        selected_inputs = selected_inputs.to(storage_dtype)
    if torch.is_floating_point(selected_windows):
        selected_windows = selected_windows.to(storage_dtype)

    return {
        "TP_input": selected_inputs.contiguous(),
        "TP_future": selected_windows.contiguous(),
        "num_samples": int(selected_inputs.shape[0]),
        "episode_steps": total_steps,
    }


def _stats_scalar(stats, key: str, env_idx: int, default: float = float("nan")) -> float:
    try:
        value = stats[key][env_idx]
        if isinstance(value, torch.Tensor):
            return float(value.reshape(-1)[0].item())
        return float(value)
    except Exception:
        return default


def _hover_pidrate_action(expert, n_agents: int, device: torch.device) -> torch.Tensor:
    t_cmd = torch.full(
        (n_agents,),
        float(expert.cfg.hover_thrust_ratio),
        device=device,
    )
    omega_cmd = torch.zeros(n_agents, 3, device=device)
    return _expert_to_pidrate_action(expert, t_cmd, omega_cmd)


def _expert_to_pidrate_action(
    expert,
    t_cmd: torch.Tensor,
    omega_cmd: torch.Tensor,
) -> torch.Tensor:
    try:
        return expert.t_omega_to_pidrate_action(
            t_cmd,
            omega_cmd,
        )
    except TypeError:
        return expert.t_omega_to_pidrate_action(
            t_cmd,
            omega_cmd,
            max_body_rate_rad_s=float(expert.cfg.max_body_rate_rad_s),
        )


# ──────────────────────────────────────────────────────────────
#  One episode runner
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def run_episode(env, base_env, expert,
                record_video: bool = False,
                max_steps: int = 1300):
    """
    Run one episode of the expert policy inside Isaac Sim.

    Action space: PIDRateController receives [omega(3), thrust(1)].
      - omega:  body-frame angular rate command, [-1,1]
      - thrust: [-1,1], where 0 ≈ hover

    Expert outputs thrust ratio `t` plus body-rate command `omega`.
    We convert that to the same normalized PIDRate action space used by
    the current tanh actor, so the pursuers use the same 6DoF dynamics
    path and the same action semantics as RL training.

    dt = 0.01 s (sim_base.yaml).  max_steps=1300 → 13 s per episode.
    """
    frames     = []
    tensordict = env.reset()
    step_count = 0
    done_flag  = False
    ep_info    = {}
    device     = base_env.device
    n_agents   = base_env.drone.n
    dt         = float(base_env.dt)   # 0.01 s
    v_max      = float(base_env.cfg.task.v_drone)
    min_target_dist_seen = float("inf")
    min_goal_dist_seen = float("inf")
    expert_pred_error_pos_sum = 0.0
    expert_pred_error_next_sum = 0.0
    expert_pred_error_steps = 0

    def _scalar(stats, key, default=float("nan")):
        try:
            return float(stats[key][0].item())
        except Exception:
            return default

    while not done_flag and step_count < max_steps:
        # ── read physical state from Isaac Sim ────────────────
        drone_pos_w, drone_quat_w = base_env.get_env_poses(base_env.drone.get_world_poses())
        drone_vel_3    = base_env.drone.get_velocities()[..., :3]   # [n_envs, n, 3]
        target_pos_w, _= base_env.get_env_poses(base_env.target.get_world_poses())
        target_vel_full= base_env.target.get_velocities()            # [n_envs, 1, 6]
        target_vel_3   = target_vel_full[..., :3]                    # [n_envs, 1, 3]

        # env 0 only (num_envs=1 in eval mode)
        dp = drone_pos_w[0]    # [n_agents, 3]
        dq = drone_quat_w[0]   # [n_agents, 4]
        dv = drone_vel_3[0]    # [n_agents, 3]
        tp = target_pos_w[0].reshape(-1, 3)[:1]   # [1, 3]
        tv = target_vel_3[0].reshape(-1, 3)[:1]   # [1, 3]
        target_dist_now = torch.norm(dp - tp[0], dim=-1).min().item()
        goal_dist_now = torch.norm(tp[0, :2] - base_env.goal_region_center[:2]).item()
        min_target_dist_seen = min(min_target_dist_seen, target_dist_now)
        min_goal_dist_seen = min(min_goal_dist_seen, goal_dist_now)

        # ── expert: thrust ratio + body-rate command ─────────
        target_next_pos, _ = compute_true_target_next(base_env, tp, tv)
        target_next_pos = target_next_pos.reshape(-1, 3)[:1]
        try:
            if hasattr(expert, "get_actions_batch"):
                t_cmd_b, omega_cmd_b, vel_cmd_b, expert_debug = expert.get_actions_batch(
                    step=step_count,
                    drone_pos=dp.unsqueeze(0),
                    drone_vel=dv.unsqueeze(0),
                    target_pos=tp.unsqueeze(0),
                    target_vel=tv.unsqueeze(0),
                    target_next_pos=target_next_pos.unsqueeze(0),
                    drone_quat=dq.unsqueeze(0),
                    return_debug=True,
                )
                t_cmd = t_cmd_b.squeeze(0)
                omega_cmd = omega_cmd_b.squeeze(0)
                vel_cmd = vel_cmd_b.squeeze(0)
                pred_pos_debug = expert_debug["target_pos_pred"][0]
            else:
                t_cmd, omega_cmd, vel_cmd, expert_debug = expert.get_actions(
                    step=step_count,
                    drone_pos=dp,
                    drone_vel=dv,
                    target_pos=tp,
                    target_vel=tv,
                    target_next_pos=target_next_pos,
                    drone_quat=dq,
                    return_debug=True,
                )
                pred_pos_debug = expert_debug["target_pos_pred"]
            expert_pred_error_pos_sum += torch.norm(
                pred_pos_debug - tp[0]
            ).item()
            expert_pred_error_next_sum += torch.norm(
                pred_pos_debug - target_next_pos[0]
            ).item()
            expert_pred_error_steps += 1
        except Exception as e:
            logging.warning(f"Expert error at step {step_count}: {e}")
            t_cmd = torch.full((n_agents,), expert.cfg.hover_thrust_ratio, device=device)
            omega_cmd = torch.zeros(n_agents, 3, device=device)
            vel_cmd = torch.zeros(n_agents, 3, device=device)

        # ── convert (t, omega) → normalized PIDrate action ────
        pidrate_action = _expert_to_pidrate_action(expert, t_cmd, omega_cmd)

        # expand to [n_envs, n_agents, 4] (only env 0 matters)
        action_batch = pidrate_action.unsqueeze(0).expand(
            base_env.num_envs, -1, -1
        ).clone()
        tensordict["agents", "action"] = action_batch

        # ── step ──────────────────────────────────────────────
        tensordict = env.step(tensordict)
        td_next    = tensordict.get("next")

        # ── render ────────────────────────────────────────────
        if record_video and step_count % 2 == 0:
            frame = base_env.render(mode="rgb_array")
            if frame is not None:
                frames.append(frame)

        # ── check termination ─────────────────────────────────
        done = td_next.get("done")[0].item()
        if done:
            stats   = td_next.get("stats")
            success = bool(stats["success"][0].item() > 0)
            goal    = bool(stats["goal_reached"][0].item() > 0)
            landed  = bool(stats["any_landed"][0].item() > 0)
            ep_info = {
                "steps":   step_count,
                "success": success,
                "goal":    goal,
                "landed":  landed,
                "timeout": not success and not goal and not landed,
                "min_target_dist_seen": min_target_dist_seen,
                "min_goal_dist_seen": min_goal_dist_seen,
                "final_min_target_dist": target_dist_now,
                "return": _scalar(stats, "return"),
                "first_capture_step": _scalar(stats, "first_capture_step"),
                "d_i_min": _scalar(stats, "d_i_min"),
                "d_i_mean": _scalar(stats, "d_i_mean"),
                "n_agents_ahead_mean": _scalar(stats, "n_agents_ahead_mean"),
                "phi_block": _scalar(stats, "phi_block"),
                "phi_pressure": _scalar(stats, "phi_pressure"),
                "phi_spread": _scalar(stats, "phi_spread"),
                "phi_team": _scalar(stats, "phi_team"),
                "goal_progress_reward": _scalar(stats, "goal_progress_reward"),
                "capture_progress_reward": _scalar(stats, "capture_progress_reward"),
                "collision": _scalar(stats, "collision"),
                "target_predicted_error": _scalar(stats, "target_predicted_error"),
                "expert_pred_error_pos_mean": (
                    expert_pred_error_pos_sum / max(expert_pred_error_steps, 1)
                ),
                "expert_pred_error_next_mean": (
                    expert_pred_error_next_sum / max(expert_pred_error_steps, 1)
                ),
            }
            done_flag = True

        tensordict = td_next
        step_count += 1

    if not done_flag:
        ep_info = {
            "steps": step_count,
            "success": False,
            "goal": False,
            "landed": False,
            "timeout": True,
            "min_target_dist_seen": min_target_dist_seen,
            "min_goal_dist_seen": min_goal_dist_seen,
            "final_min_target_dist": float("nan"),
            "return": float("nan"),
            "first_capture_step": float("nan"),
            "d_i_min": float("nan"),
            "d_i_mean": float("nan"),
            "n_agents_ahead_mean": float("nan"),
            "phi_block": float("nan"),
            "phi_pressure": float("nan"),
            "phi_spread": float("nan"),
            "phi_team": float("nan"),
            "goal_progress_reward": float("nan"),
            "capture_progress_reward": float("nan"),
            "collision": float("nan"),
            "target_predicted_error": float("nan"),
            "expert_pred_error_pos_mean": (
                expert_pred_error_pos_sum / max(expert_pred_error_steps, 1)
            ),
            "expert_pred_error_next_mean": (
                expert_pred_error_next_sum / max(expert_pred_error_steps, 1)
            ),
        }

    return frames, ep_info


@torch.no_grad()
def run_episode_batch(
    env,
    base_env,
    expert,
    max_steps: int = 1300,
    collect_success_dataset: bool = False,
    storage_dtype: torch.dtype = torch.float16,
    min_success_steps: int = 1,
    collect_tp_dataset: bool = False,
    tp_storage_dtype: torch.dtype = torch.float16,
):
    """
    Run one batched wave of episodes in parallel.

    Uses a batched expert policy so that expert decisions are computed
    for all env slots together instead of one Python loop per env.
    """
    tensordict = env.reset()
    device = base_env.device
    n_envs = int(base_env.num_envs)
    n_agents = int(base_env.drone.n)

    done_mask = torch.zeros(n_envs, dtype=torch.bool, device=device)
    results = [None for _ in range(n_envs)]
    min_target_dist_seen = [float("inf")] * n_envs
    min_goal_dist_seen = [float("inf")] * n_envs
    expert_pred_error_pos_sum = torch.zeros(n_envs, device=device)
    expert_pred_error_next_sum = torch.zeros(n_envs, device=device)
    expert_pred_error_count = torch.zeros(n_envs, device=device)
    obs_step_buffer = [] if collect_success_dataset else None
    prev_action_step_buffer = [] if collect_success_dataset else None
    action_step_buffer = [] if collect_success_dataset else None
    aux_step_buffer = [] if collect_success_dataset else None
    tp_input_step_buffer = [] if collect_tp_dataset else None
    tp_groundtruth_step_buffer = [] if collect_tp_dataset else None
    tp_done_step_buffer = [] if collect_tp_dataset else None
    env_done_step_buffer = [] if collect_tp_dataset else None
    expert.reset(n_envs)
    hover_action = _expert_to_pidrate_action(
        expert,
        torch.full(
            (n_envs, n_agents),
            float(expert.cfg.hover_thrust_ratio),
            device=device,
        ).reshape(-1),
        torch.zeros(n_envs * n_agents, 3, device=device),
        ).reshape(n_envs, n_agents, 4)
    prev_action_batch = torch.zeros(n_envs, n_agents, 4, device=device, dtype=torch.float32)

    for step_count in range(max_steps):
        drone_pos_w, drone_quat_w = base_env.get_env_poses(base_env.drone.get_world_poses())
        drone_vel_3 = base_env.drone.get_velocities()[..., :3]
        target_pos_w, _ = base_env.get_env_poses(base_env.target.get_world_poses())
        target_vel_3 = base_env.target.get_velocities()[..., :3]
        target_dist_all = torch.norm(
            drone_pos_w - target_pos_w[..., :1, :], dim=-1
        ).min(dim=-1).values
        goal_dist_all = torch.norm(
            target_pos_w[:, 0, :2] - base_env.goal_region_center[:2], dim=-1
        )
        for env_idx in range(n_envs):
            min_target_dist_seen[env_idx] = min(
                min_target_dist_seen[env_idx], float(target_dist_all[env_idx].item())
            )
            min_goal_dist_seen[env_idx] = min(
                min_goal_dist_seen[env_idx], float(goal_dist_all[env_idx].item())
            )

        if collect_success_dataset:
            obs_step_buffer.append(
                _obs_to_storage_dict(
                    tensordict[("agents", "observation")],
                    storage_dtype,
                )
            )
            prev_action_step_buffer.append(
                prev_action_batch.detach().cpu().to(storage_dtype).contiguous()
            )
            env_aux = _extract_env_aligned_aux_from_obs(
                tensordict[("agents", "observation")],
                future_steps=int(getattr(base_env, "future_predcition_step", 5)),
            )
        else:
            env_aux = None

        target_next_pos, _ = compute_true_target_next(base_env, target_pos_w, target_vel_3)
        try:
            expert_out = expert.get_actions_batch(
                step=step_count,
                drone_pos=drone_pos_w,
                drone_vel=drone_vel_3,
                target_pos=target_pos_w,
                target_vel=target_vel_3,
                target_next_pos=target_next_pos,
                drone_quat=drone_quat_w,
                return_debug=True,
            )
            t_cmd, omega_cmd, vel_cmd_expert, expert_debug = expert_out
            action_batch = _expert_to_pidrate_action(
                expert,
                t_cmd.reshape(-1),
                omega_cmd.reshape(-1, 3),
            ).reshape(n_envs, n_agents, 4)
            active_mask = ~done_mask
            if bool(active_mask.any()):
                pred_pos_err = torch.norm(
                    expert_debug["target_pos_pred"] - target_pos_w.squeeze(1), dim=-1
                )
                pred_next_err = torch.norm(
                    expert_debug["target_pos_pred"] - target_next_pos.squeeze(1), dim=-1
                )
                expert_pred_error_pos_sum[active_mask] += pred_pos_err[active_mask]
                expert_pred_error_next_sum[active_mask] += pred_next_err[active_mask]
                expert_pred_error_count[active_mask] += 1
        except Exception as exc:
            logging.warning(f"Batched expert error at step {step_count}: {exc}")
            action_batch = hover_action.clone()
            vel_cmd_expert = torch.zeros(n_envs, n_agents, 3, device=device)
            expert_debug = {
                "assignment": torch.zeros(n_envs, n_agents, dtype=torch.long, device=device),
                "waypoint": torch.zeros(n_envs, n_agents, 3, device=device),
                "target_pos_pred": torch.zeros(n_envs, 3, device=device),
                "forward_dir": torch.zeros(n_envs, 3, device=device),
                "trap_mode": torch.zeros(n_envs, dtype=torch.bool, device=device),
            }

        if bool(done_mask.any()):
            action_batch[done_mask] = hover_action[done_mask]
            if collect_success_dataset:
                vel_cmd_expert[done_mask] = 0.0
                expert_debug["assignment"][done_mask] = 0
                expert_debug["waypoint"][done_mask] = 0.0
                expert_debug["target_pos_pred"][done_mask] = 0.0
                expert_debug["forward_dir"][done_mask] = 0.0
                expert_debug["trap_mode"][done_mask] = False

        if collect_success_dataset:
            action_step_buffer.append(
                action_batch.detach().cpu().to(storage_dtype).contiguous()
            )
            aux_step_buffer.append(
                {
                    "vel_cmd": vel_cmd_expert.detach().cpu().to(storage_dtype).contiguous(),
                    "assignment": env_aux["assignment"].detach().cpu().to(torch.int16).contiguous(),
                    "waypoint": env_aux["waypoint"].detach().cpu().to(storage_dtype).contiguous(),
                    "target_pos_pred": env_aux["target_pos_pred"].detach().cpu().to(storage_dtype).contiguous(),
                    "forward_dir": env_aux["forward_dir"].detach().cpu().to(storage_dtype).contiguous(),
                    "trap_mode": env_aux["trap_mode"].detach().cpu().to(torch.bool).contiguous(),
                }
            )

        tensordict["agents", "action"] = action_batch
        prev_action_batch = action_batch.detach().clone()
        tensordict = env.step(tensordict)
        td_next = tensordict.get("next")
        done_vec = td_next.get("done").reshape(n_envs).bool()
        stats = td_next.get("stats")

        if collect_tp_dataset:
            tp_td = td_next["agents"]["TP"]
            tp_input_step_buffer.append(
                tp_td["TP_input"].detach().cpu().to(tp_storage_dtype).contiguous()
            )
            tp_groundtruth_step_buffer.append(
                tp_td["TP_groundtruth"].detach().cpu().to(tp_storage_dtype).contiguous()
            )
            tp_done_step_buffer.append(
                tp_td["TP_done"].detach().cpu().to(torch.bool).contiguous()
            )
            env_done_step_buffer.append(
                done_vec.detach().cpu().to(torch.bool).contiguous()
            )

        for env_idx in range(n_envs):
            if done_mask[env_idx] or not done_vec[env_idx]:
                continue
            success = bool(stats["success"][env_idx].item() > 0)
            goal = bool(stats["goal_reached"][env_idx].item() > 0)
            landed = bool(stats["any_landed"][env_idx].item() > 0)
            results[env_idx] = {
                "steps": step_count,
                "success": success,
                "goal": goal,
                "landed": landed,
                "timeout": not success and not goal and not landed,
                "min_target_dist_seen": min_target_dist_seen[env_idx],
                "min_goal_dist_seen": min_goal_dist_seen[env_idx],
                "return": _stats_scalar(stats, "return", env_idx),
                "first_capture_step": _stats_scalar(stats, "first_capture_step", env_idx),
                "d_i_min": _stats_scalar(stats, "d_i_min", env_idx),
                "d_i_mean": _stats_scalar(stats, "d_i_mean", env_idx),
                "n_agents_ahead_mean": _stats_scalar(stats, "n_agents_ahead_mean", env_idx),
                "phi_block": _stats_scalar(stats, "phi_block", env_idx),
                "phi_pressure": _stats_scalar(stats, "phi_pressure", env_idx),
                "phi_spread": _stats_scalar(stats, "phi_spread", env_idx),
                "phi_team": _stats_scalar(stats, "phi_team", env_idx),
                "goal_progress_reward": _stats_scalar(stats, "goal_progress_reward", env_idx),
                "capture_progress_reward": _stats_scalar(stats, "capture_progress_reward", env_idx),
                "collision": _stats_scalar(stats, "collision", env_idx),
                "collision_drone": _stats_scalar(stats, "collision_drone", env_idx),
                "collision_wall": _stats_scalar(stats, "collision_wall", env_idx),
                "collision_floor": _stats_scalar(stats, "collision_floor", env_idx),
                "pursuer_collisions_count": _stats_scalar(stats, "pursuer_collisions_count", env_idx),
                "target_predicted_error": _stats_scalar(stats, "target_predicted_error", env_idx),
                "expert_pred_error_pos_mean": float(
                    (expert_pred_error_pos_sum[env_idx] / expert_pred_error_count[env_idx].clamp(min=1)).item()
                ),
                "expert_pred_error_next_mean": float(
                    (expert_pred_error_next_sum[env_idx] / expert_pred_error_count[env_idx].clamp(min=1)).item()
                ),
            }
            done_mask[env_idx] = True

        if bool(done_mask.all()):
            tensordict = td_next
            break

        tensordict = td_next

    for env_idx in range(n_envs):
        if results[env_idx] is not None:
            continue
        results[env_idx] = {
            "steps": max_steps - 1,
            "success": False,
            "goal": False,
            "landed": False,
            "timeout": True,
            "min_target_dist_seen": min_target_dist_seen[env_idx],
            "min_goal_dist_seen": min_goal_dist_seen[env_idx],
            "return": float("nan"),
            "first_capture_step": float("nan"),
            "d_i_min": float("nan"),
            "d_i_mean": float("nan"),
            "n_agents_ahead_mean": float("nan"),
            "phi_block": float("nan"),
            "phi_pressure": float("nan"),
            "phi_spread": float("nan"),
            "phi_team": float("nan"),
            "goal_progress_reward": float("nan"),
            "capture_progress_reward": float("nan"),
            "collision": float("nan"),
            "target_predicted_error": float("nan"),
            "expert_pred_error_pos_mean": float(
                (expert_pred_error_pos_sum[env_idx] / expert_pred_error_count[env_idx].clamp(min=1)).item()
            ),
            "expert_pred_error_next_mean": float(
                (expert_pred_error_next_sum[env_idx] / expert_pred_error_count[env_idx].clamp(min=1)).item()
            ),
        }

    dataset_chunk = None
    if collect_success_dataset:
        assert (
            obs_step_buffer is not None
            and prev_action_step_buffer is not None
            and action_step_buffer is not None
            and aux_step_buffer is not None
        )
        stacked_obs = {
            key: torch.stack([step_obs[key] for step_obs in obs_step_buffer], dim=0)
            for key in obs_step_buffer[0].keys()
        } if obs_step_buffer else {}
        stacked_aux = {
            key: torch.stack([step_aux[key] for step_aux in aux_step_buffer], dim=0)
            for key in aux_step_buffer[0].keys()
        } if aux_step_buffer else {}
        stacked_action = (
            torch.stack(action_step_buffer, dim=0)
            if action_step_buffer else
            torch.empty(0, n_envs, n_agents, 4, dtype=storage_dtype)
        )
        stacked_prev_action = (
            torch.stack(prev_action_step_buffer, dim=0)
            if prev_action_step_buffer else
            torch.empty(0, n_envs, n_agents, 4, dtype=storage_dtype)
        )

        obs_flat = {key: [] for key in stacked_obs.keys()}
        aux_flat = {key: [] for key in stacked_aux.keys()}
        prev_action_flat = []
        action_flat = []
        episode_lengths = []
        episode_meta = []
        dropped_too_short = 0

        for env_idx, ep_info in enumerate(results):
            steps = int(ep_info["steps"]) + 1
            if not ep_info["success"]:
                continue
            if steps < int(min_success_steps):
                dropped_too_short += 1
                continue
            for key, value in stacked_obs.items():
                obs_flat[key].append(value[:steps, env_idx].contiguous())
            for key, value in stacked_aux.items():
                aux_flat[key].append(value[:steps, env_idx].contiguous())
            prev_action_flat.append(stacked_prev_action[:steps, env_idx].contiguous())
            action_flat.append(stacked_action[:steps, env_idx].contiguous())
            episode_lengths.append(steps)
            episode_meta.append({
                **ep_info,
                "env_idx": env_idx,
            })

        if episode_lengths:
            obs_flat = {
                key: torch.cat(parts, dim=0).contiguous()
                for key, parts in obs_flat.items()
            }
            aux_flat = {
                key: torch.cat(parts, dim=0).contiguous()
                for key, parts in aux_flat.items()
            }
            action_flat = torch.cat(action_flat, dim=0).contiguous()
            prev_action_flat = torch.cat(prev_action_flat, dim=0).contiguous()
        else:
            obs_flat = {key: torch.empty(0, *value.shape[2:], dtype=value.dtype) for key, value in stacked_obs.items()}
            aux_flat = {key: torch.empty(0, *value.shape[2:], dtype=value.dtype) for key, value in stacked_aux.items()}
            action_flat = torch.empty(0, n_agents, 4, dtype=storage_dtype)
            prev_action_flat = torch.empty(0, n_agents, 4, dtype=storage_dtype)

        dataset_chunk = {
            "obs": obs_flat,
            "expert_aux": aux_flat,
            "prev_action": prev_action_flat,
            "action_raw": action_flat,
            "action_label_type": "pidrate_normalized",
            "episode_lengths": torch.as_tensor(episode_lengths, dtype=torch.int32),
            "episode_meta": episode_meta,
            "num_success_episodes": len(episode_lengths),
            "num_success_steps": int(action_flat.shape[0]),
            "dropped_too_short": int(dropped_too_short),
            "storage_dtype": str(storage_dtype),
        }

    tp_dataset_chunk = None
    if collect_tp_dataset:
        assert (
            tp_input_step_buffer is not None
            and tp_groundtruth_step_buffer is not None
            and tp_done_step_buffer is not None
            and env_done_step_buffer is not None
        )
        tp_dataset_chunk = _build_tp_dataset_chunk(
            tp_input_step_buffer,
            tp_groundtruth_step_buffer,
            tp_done_step_buffer,
            env_done_step_buffer,
            future_step=int(base_env.future_predcition_step),
            window_step=int(base_env.window_step),
            storage_dtype=tp_storage_dtype,
        )

    return results, dataset_chunk, tp_dataset_chunk


# ──────────────────────────────────────────────────────────────
#  Main (Hydra entry point)
# ──────────────────────────────────────────────────────────────
from omni_drones import CONFIG_PATH, init_simulation_app

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # ── unpack extra args ──────────────────────────────────────
    args        = _extra_args
    pred_mode   = args.pred_mode
    strategy_variant = args.strategy_variant
    forward_dir_mode = args.forward_dir_mode
    enable_goal_mode = args.enable_goal_mode
    enable_close_mode = args.enable_close_mode
    enable_rush_mode = args.enable_rush_mode
    expert2_front_layout = args.expert2_front_layout
    expert_intercept_pred_step = int(args.expert_intercept_pred_step)
    expert_intercept_use_direct_pred = bool(args.expert_intercept_use_direct_pred)
    tp_weight   = args.tp_weight
    n_video     = args.n_video
    n_generic   = args.n_generic
    v_prey_test = args.v_prey_test
    v_prey_schedule = parse_float_schedule(args.v_prey_schedule, v_prey_test)
    v_drone_test = args.v_drone_test
    video_dir   = args.video_dir
    video_seed_base = args.video_seed_base
    video_seed_list = [
        int(x.strip()) for x in str(args.video_seed_list or "").split(",") if x.strip()
    ]
    if video_seed_list:
        n_video = len(video_seed_list)
    generic_seed_base = int(args.generic_seed_base)
    generic_batch_envs = max(1, int(args.generic_batch_envs))
    random_init = args.random_init
    episode_length = args.episode_length
    max_steps   = min(args.max_steps, episode_length)
    collect_success_dataset = bool(args.collect_success_dataset)
    dataset_dir = args.dataset_dir
    dataset_name = str(args.dataset_name or "").strip()
    min_success_steps = max(1, int(args.min_success_steps))
    dataset_dtype = _storage_dtype(args.dataset_dtype)
    collect_tp_dataset = bool(args.collect_tp_dataset)
    tp_dataset_dir = args.tp_dataset_dir
    tp_dataset_name = str(args.tp_dataset_name or "").strip()
    tp_dataset_dtype = _storage_dtype(args.tp_dataset_dtype)

    # headless=True + enable_render(True) → offscreen rendering, no display needed
    # do NOT force headless=False (crashes without X server)

    # ── env count ─────────────────────────────────────────────
    # Video recording stays single-env. Generic no-video testing can be batched.
    eval_num_envs = 1 if n_video > 0 else max(1, min(generic_batch_envs, n_generic if n_generic > 0 else 1))
    cfg.task.num_envs     = eval_num_envs
    cfg.task.env.num_envs = eval_num_envs
    if hasattr(cfg, "env"):
        cfg.env.num_envs = eval_num_envs
        cfg.env.max_episode_length = int(episode_length)
    cfg.task.max_episode_length = int(episode_length)
    cfg.task.env.max_episode_length = int(episode_length)
    cfg.task.v_drone = float(v_drone_test)

    # ── random init ───────────────────────────────────────────
    cfg.task.use_eval = 0 if random_init else 1

    # ── start Isaac Sim ───────────────────────────────────────
    simulation_app = init_simulation_app(cfg)

    from omni_drones.envs.isaac_env import IsaacEnv
    from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose
    from omni_drones.utils.torchrl.transforms import ravel_composite
    from omni_drones.controllers import PIDRateController as _PIDRateController
    from omni_drones.utils.torchrl.transforms import PIDRateController

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env  = env_class(cfg, headless=cfg.headless)

    dt        = float(base_env.dt)   # 0.01 s (from sim_base.yaml)
    fps       = max(1, int(0.5 / dt))

    # ── build env with PIDRateController transform ─────────────
    # This matches the training environment exactly:
    #   Policy/Expert outputs [omega(3), thrust(1)] in [-1,1]
    #   → PIDRateController converts to motor RPMs
    transforms = [InitTracker()]
    controller = _PIDRateController(dt, 9.81, base_env.drone.params).to(base_env.device)
    transforms.append(PIDRateController(controller, actor_has_tanh=True))
    env = TransformedEnv(base_env, Compose(*transforms))
    env.set_seed(0)

    device = base_env.device

    if not tp_weight and pred_mode == "tp_net":
        import glob
        candidates = sorted(glob.glob("checkpoints/**/tp_only_*.pt", recursive=True))
        tp_weight = candidates[-1] if candidates else ""
    if pred_mode == "tp_net" and (not tp_weight or not os.path.isfile(tp_weight)):
        print(f"[Expert] tp_net weight not found: {tp_weight!r}, falling back to noise mode")
        pred_mode = "noise"
    if pred_mode == "tp_net" and bool(getattr(cfg.algo, "use_TP_net", False)):
        tp_state = torch.load(tp_weight, map_location=device)
        if isinstance(tp_state, dict) and "TP" in tp_state:
            tp_state = tp_state["TP"]
        base_env.TP.load_state_dict(tp_state)
        base_env.TP.eval()
        print(f"[Expert] Env TP_net loaded for observations: {tp_weight}")

    hover_thrust_ratio = float(
        (base_env.drone.gravity[0, 0] / controller.max_thrusts.sum()).item()
    )
    expert_cfg = SharedExpertCfg()
    expert_cfg.dt = dt
    expert_cfg.num_agents = int(base_env.drone.n)
    expert_cfg.arena_size = float(base_env.arena_size)
    expert_cfg.max_height = float(base_env.max_height)
    expert_cfg.max_episode_length = int(base_env.max_episode_length)
    expert_cfg.v_drone = float(cfg.task.v_drone)
    expert_cfg.catch_radius = float(cfg.task.catch_radius)
    expert_cfg.goal_region_center = base_env.goal_region_center.to(device)
    expert_cfg.goal_region_radius = float(cfg.task.goal_region_radius)
    expert_cfg.goal_region_height = float(cfg.task.goal_region_height)
    expert_cfg.hover_thrust_ratio = hover_thrust_ratio
    expert_cfg.min_thrust_ratio = float(controller.min_thrust_ratio)
    expert_cfg.max_thrust_ratio = float(controller.max_thrust_ratio)
    expert_cfg.target_clip = float(controller.target_clip)
    expert_cfg.max_body_rate_rad_s = math.radians(180.0 * expert_cfg.target_clip)
    expert_cfg.forward_dir_mode = str(forward_dir_mode)
    apply_shared_strategy_defaults(
        expert_cfg,
        strategy_variant=str(strategy_variant),
        enable_goal_mode=enable_goal_mode,
        enable_close_mode=enable_close_mode,
        enable_rush_mode=enable_rush_mode,
        expert2_front_layout=str(expert2_front_layout) if expert2_front_layout is not None else None,
    )
    strategy_variant = str(expert_cfg.strategy_variant)
    enable_goal_mode = bool(expert_cfg.enable_goal_mode)
    enable_close_mode = bool(expert_cfg.enable_close_mode)
    enable_rush_mode = bool(expert_cfg.enable_rush_mode)
    expert2_front_layout = str(expert_cfg.expert2_front_layout)
    if hasattr(base_env, "history_step"):
        expert_cfg.history_step = int(base_env.history_step)
    if hasattr(base_env, "future_predcition_step"):
        expert_cfg.future_predcition_step = int(base_env.future_predcition_step)
    if hasattr(base_env, "window_step"):
        expert_cfg.window_step = int(base_env.window_step)
    expert_cfg.expert_intercept_pred_step = int(expert_intercept_pred_step)
    expert_cfg.expert_intercept_use_direct_pred = bool(expert_intercept_use_direct_pred)

    def _build_expert():
        return SharedExpertPolicy(
            expert_cfg,
            pred_mode=pred_mode,
            tp_weight_path=tp_weight,
            tp_device=str(device),
        )

    def _build_batched_expert():
        return BatchedExpertPolicy(
            expert_cfg,
            pred_mode=pred_mode,
            tp_weight_path=tp_weight,
            tp_device=str(device),
        )

    expert = _build_expert()
    batched_expert = _build_batched_expert()
    if pred_mode == "tp_net":
        print(f"[Expert] TP_net loaded: {tp_weight}")
    elif pred_mode.startswith("oracle_"):
        print(f"[Expert] Oracle predictor enabled: {pred_mode}")
    print(
        f"[Expert] strategy_variant={strategy_variant} "
        f"forward_dir_mode={forward_dir_mode} "
        f"goal_mode={enable_goal_mode} close={enable_close_mode} "
        f"rush={enable_rush_mode} front_layout={expert2_front_layout} "
        f"intercept_pred_step={expert_intercept_pred_step} "
        f"direct_pred={expert_intercept_use_direct_pred}"
    )

    # ── rendering policy ──────────────────────────────────────
    # Generic evaluation with n_video=0 does not need the render pipeline.
    # Keeping rendering enabled here makes Isaac carry the graphics stack
    # even for pure statistics runs, which slows large batched evaluation.
    base_env.enable_render(n_video > 0)
    base_env.eval()
    env.eval()

    os.makedirs(video_dir, exist_ok=True)
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if collect_success_dataset:
        if not os.path.isabs(dataset_dir):
            dataset_dir = os.path.join(os.getcwd(), dataset_dir)
        dataset_dir = os.path.join(dataset_dir, dataset_name or f"{cfg.task.name}_expert_{time_str}")
        os.makedirs(dataset_dir, exist_ok=True)
    if collect_tp_dataset:
        if not os.path.isabs(tp_dataset_dir):
            tp_dataset_dir = os.path.join(os.getcwd(), tp_dataset_dir)
        tp_dataset_dir = os.path.join(tp_dataset_dir, tp_dataset_name or f"{cfg.task.name}_tp_{time_str}")
        os.makedirs(tp_dataset_dir, exist_ok=True)

    # Keep the harder target speed while using random initial positions.
    if hasattr(base_env, "_set_curriculum_stage"):
        base_env._set_curriculum_stage(
            len(base_env.curriculum_stages) - 1,
            reset_metrics=False,
            announce=False,
        )
    base_env.current_target_speed = float(v_prey_test)
    base_env.target_velocity_scale = float(v_prey_test)

    print(f"\n[Expert eval] dt={dt:.3f}s  episode_length={episode_length}  max_steps={max_steps}")
    print(
        f"             pred_mode={pred_mode}  strategy_variant={strategy_variant} "
        f"forward_dir_mode={forward_dir_mode}  goal_mode={enable_goal_mode} "
        f"close={enable_close_mode} rush={enable_rush_mode} "
        f"front_layout={expert2_front_layout}  intercept_pred_step={expert_intercept_pred_step} "
        f"direct_pred={expert_intercept_use_direct_pred} "
        f"v_drone={cfg.task.v_drone}  v_prey={v_prey_test}"
    )
    if len(v_prey_schedule) > 1:
        print(f"             v_prey_schedule={','.join(f'{v:.3g}' for v in v_prey_schedule)}")
    print(f"             num_envs={base_env.num_envs}  generic_batch_envs={generic_batch_envs}")
    print(f"             hover_thrust_ratio={hover_thrust_ratio:.3f}  max_body_rate={expert_cfg.max_body_rate_rad_s:.3f} rad/s")
    if collect_success_dataset:
        print(f"             dataset_dir={dataset_dir}  min_success_steps={min_success_steps}  dataset_dtype={args.dataset_dtype}")
    if collect_tp_dataset:
        print(f"             tp_dataset_dir={tp_dataset_dir}  tp_dataset_dtype={args.tp_dataset_dtype}")

    # ═══════════════════════════════════════════════════════════
    #  Part 1: Record n_video videos (random init, normal v_prey)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  Part 1: Recording {n_video} videos")
    print(f"  pred_mode={pred_mode}, v_prey={cfg.task.v_prey}")
    if video_seed_list:
        print(f"  explicit video seeds={video_seed_list}")
    print(f"{'='*60}")

    video_expert = batched_expert
    video_paths = []
    seeds_to_record = video_seed_list or [video_seed_base + vid_idx * 100 for vid_idx in range(n_video)]
    for vid_idx, video_seed in enumerate(seeds_to_record):
        env.set_seed(video_seed)
        video_expert.reset(1)
        frames, ep_info = run_episode(
            env, base_env, video_expert,
            record_video=True,
            max_steps=max_steps,
        )
        outcome = ("CAPTURE" if ep_info["success"] else
                   "GOAL_ZONE" if ep_info["goal"] else
                   "LANDED" if ep_info["landed"] else "TIMEOUT")
        vpath = os.path.join(
            video_dir, f"expert_{outcome}_seed{video_seed}_{time_str}.mp4"
        )
        if frames:
            saved = save_video(frames, fps, vpath)
            video_paths.append(saved)
        print(f"  Video {vid_idx+1}/{n_video}: {outcome} | "
              f"seed={video_seed} | steps={ep_info['steps']} | saved={vpath}")

    # ═══════════════════════════════════════════════════════════
    #  Part 2: Generalization test — v_prey = v_prey_test
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  Part 2: Generalization test — v_prey_test={v_prey_test}")
    print(f"  Running {n_generic} episodes...")
    print(f"{'='*60}")

    # Override target speed
    base_env._set_curriculum_stage(
        len(base_env.curriculum_stages) - 1,
        reset_metrics=False,
        announce=False,
    )
    base_env.current_target_speed  = v_prey_test
    base_env.target_velocity_scale = v_prey_test
    print(f"  Target velocity scale set to: {base_env.target_velocity_scale:.2f} m/s")

    def _mean(items, key):
        vals = [r[key] for r in items if not math.isnan(r.get(key, float("nan")))]
        return float(np.mean(vals)) if vals else float("nan")

    gen_results = []
    saved_dataset_chunks = 0
    saved_dataset_episodes = 0
    saved_dataset_steps = 0
    saved_tp_chunks = 0
    saved_tp_samples = 0
    if int(base_env.num_envs) == 1:
        if collect_success_dataset:
            logging.warning("collect_success_dataset is intended for batched generic eval; skipping dataset save in single-env mode.")
        if collect_tp_dataset:
            logging.warning("collect_tp_dataset is intended for batched generic eval; skipping TP dataset save in single-env mode.")
        for ep_idx in range(n_generic):
            wave_v_prey = v_prey_schedule[ep_idx % len(v_prey_schedule)]
            base_env.current_target_speed = float(wave_v_prey)
            base_env.target_velocity_scale = float(wave_v_prey)
            env.set_seed(ep_idx * 7 + generic_seed_base)
            expert.reset()
            _, ep_info = run_episode(
                env, base_env, expert,
                record_video=False,
                max_steps=max_steps,
            )
            gen_results.append(ep_info)
            outcome = ("✅ CAPTURE" if ep_info["success"] else
                       "❌ GOAL_ZONE" if ep_info["goal"] else
                       "⚠️  LANDED" if ep_info["landed"] else "⏱  TIMEOUT")
            print(
                f"  EP {ep_idx+1:02d} | {outcome} | steps={ep_info['steps']} "
                f"| mindist={ep_info['min_target_dist_seen']:.3f} "
                f"| dmin={ep_info['d_i_min']:.3f} | ahead={ep_info['n_agents_ahead_mean']:.2f} "
                f"| team={ep_info['phi_team']:.3f} | cap_prog={ep_info['capture_progress_reward']:.4f}"
            )
    else:
        ep_idx = 0
        wave = 0
        while ep_idx < n_generic:
            wave += 1
            wave_v_prey = v_prey_schedule[(wave - 1) % len(v_prey_schedule)]
            base_env.current_target_speed = float(wave_v_prey)
            base_env.target_velocity_scale = float(wave_v_prey)
            env.set_seed(wave * 97 + generic_seed_base)
            wave_start_idx = ep_idx
            wave_results, dataset_chunk, tp_dataset_chunk = run_episode_batch(
                env,
                base_env,
                batched_expert,
                max_steps=max_steps,
                collect_success_dataset=collect_success_dataset,
                storage_dtype=dataset_dtype,
                min_success_steps=min_success_steps,
                collect_tp_dataset=collect_tp_dataset,
                tp_storage_dtype=tp_dataset_dtype,
            )
            if collect_success_dataset and dataset_chunk is not None and dataset_chunk["num_success_episodes"] > 0:
                for meta in dataset_chunk["episode_meta"]:
                    meta["wave"] = wave
                    meta["global_episode_idx"] = wave_start_idx + int(meta["env_idx"])
                dataset_chunk.update({
                    "wave": wave,
                    "pred_mode": pred_mode,
                    "v_prey_test": float(wave_v_prey),
                    "v_prey_schedule": [float(v) for v in v_prey_schedule],
                    "v_drone_test": float(v_drone_test),
                    "episode_length": int(episode_length),
                    "batch_envs": int(base_env.num_envs),
                    "dt": float(dt),
                })
                chunk_path = os.path.join(dataset_dir, f"expert_success_wave_{wave:05d}.pt")
                torch.save(dataset_chunk, chunk_path)
                saved_dataset_chunks += 1
                saved_dataset_episodes += int(dataset_chunk["num_success_episodes"])
                saved_dataset_steps += int(dataset_chunk["num_success_steps"])
                print(
                    f"  [dataset] wave={wave:03d} saved={chunk_path} "
                    f"| v_prey={wave_v_prey:.2f} "
                    f"| success_eps={dataset_chunk['num_success_episodes']} "
                    f"| success_steps={dataset_chunk['num_success_steps']} "
                    f"| dropped_short={dataset_chunk['dropped_too_short']}"
                )
            if collect_tp_dataset and tp_dataset_chunk is not None and tp_dataset_chunk["num_samples"] > 0:
                tp_dataset_chunk.update({
                    "wave": wave,
                    "pred_mode": pred_mode,
                    "v_prey_test": float(wave_v_prey),
                    "v_prey_schedule": [float(v) for v in v_prey_schedule],
                    "v_drone_test": float(v_drone_test),
                    "episode_length": int(episode_length),
                    "batch_envs": int(base_env.num_envs),
                    "dt": float(dt),
                    "history_step": int(base_env.history_step),
                    "future_predcition_step": int(base_env.future_predcition_step),
                    "window_step": int(base_env.window_step),
                    "arena_size": float(base_env.arena_size),
                    "max_height": float(base_env.max_height),
                    "input_dim": int(tp_dataset_chunk["TP_input"].shape[-1]),
                    "storage_dtype": str(tp_dataset_dtype),
                })
                tp_chunk_path = os.path.join(tp_dataset_dir, f"tp_wave_{wave:05d}.pt")
                torch.save(tp_dataset_chunk, tp_chunk_path)
                saved_tp_chunks += 1
                saved_tp_samples += int(tp_dataset_chunk["num_samples"])
                print(
                    f"  [tp_dataset] wave={wave:03d} saved={tp_chunk_path} "
                    f"| v_prey={wave_v_prey:.2f} "
                    f"| samples={tp_dataset_chunk['num_samples']} "
                    f"| episode_steps={tp_dataset_chunk['episode_steps']}"
                )
            for ep_info in wave_results:
                if ep_idx >= n_generic:
                    break
                gen_results.append(ep_info)
                outcome = ("✅ CAPTURE" if ep_info["success"] else
                           "❌ GOAL_ZONE" if ep_info["goal"] else
                           "⚠️  LANDED" if ep_info["landed"] else "⏱  TIMEOUT")
                print(
                    f"  EP {ep_idx+1:02d} | {outcome} | steps={ep_info['steps']} "
                    f"| mindist={ep_info['min_target_dist_seen']:.3f} "
                    f"| dmin={ep_info['d_i_min']:.3f} | ahead={ep_info['n_agents_ahead_mean']:.2f} "
                    f"| team={ep_info['phi_team']:.3f} | cap_prog={ep_info['capture_progress_reward']:.4f}"
                )
                ep_idx += 1

    # ── summary ───────────────────────────────────────────────
    n_total  = len(gen_results)
    if n_total > 0:
        n_cap    = sum(1 for r in gen_results if r["success"])
        n_goal   = sum(1 for r in gen_results if r["goal"])
        n_landed = sum(1 for r in gen_results if r["landed"])
        n_to     = sum(1 for r in gen_results if r["timeout"])
        n_any_collision = sum(1 for r in gen_results if r.get("collision", 0.0) > 0.0)
        n_drone_collision = sum(1 for r in gen_results if r.get("pursuer_collisions_count", 0.0) > 0.0)
        cap_steps= [r["steps"] for r in gen_results if r["success"]]

        schedule_label = (
            f"schedule={','.join(f'{v:.3g}' for v in v_prey_schedule)}"
            if len(v_prey_schedule) > 1
            else f"v_prey={v_prey_test}"
        )
        print(f"\n{'='*60}")
        print(f"  Generalization Summary [{schedule_label}]")
        print(f"{'='*60}")
        print(f"  Episodes      : {n_total}")
        print(f"  ✅ Capture    : {n_cap/n_total:.0%}  ({n_cap})")
        print(f"  ❌ Goal zone  : {n_goal/n_total:.0%}  ({n_goal})")
        print(f"  ⚠️  Landed    : {n_landed/n_total:.0%}  ({n_landed})")
        print(f"  ⏱  Timeout    : {n_to/n_total:.0%}  ({n_to})")
        print(f"  💥 Any coll   : {n_any_collision/n_total:.0%}  ({n_any_collision})")
        print(f"  🤝 Drone coll : {n_drone_collision/n_total:.0%}  ({n_drone_collision})")
        print(
            f"  Expert pred   : pos_err={_mean(gen_results, 'expert_pred_error_pos_mean'):.4f}, "
            f"next_err={_mean(gen_results, 'expert_pred_error_next_mean'):.4f}"
        )
        print(
            f"  Collision avg : any={_mean(gen_results, 'collision'):.3f}, "
            f"drone={_mean(gen_results, 'collision_drone'):.3f}, "
            f"wall={_mean(gen_results, 'collision_wall'):.3f}, "
            f"floor={_mean(gen_results, 'collision_floor'):.3f}, "
            f"pairs={_mean(gen_results, 'pursuer_collisions_count'):.3f}"
        )
        if cap_steps:
            print(f"  Capture steps : mean={np.mean(cap_steps):.1f}, "
                  f"std={np.std(cap_steps):.1f}, "
                  f"min={min(cap_steps)}, max={max(cap_steps)}")
        timeout_eps = [r for r in gen_results if r["timeout"]]
        capture_eps = [r for r in gen_results if r["success"]]
        if timeout_eps:
            print(
                f"  Timeout avg   : mindist={_mean(timeout_eps, 'min_target_dist_seen'):.3f}, "
                f"dmin={_mean(timeout_eps, 'd_i_min'):.3f}, "
                f"ahead={_mean(timeout_eps, 'n_agents_ahead_mean'):.2f}, "
                f"team={_mean(timeout_eps, 'phi_team'):.3f}, "
                f"spread={_mean(timeout_eps, 'phi_spread'):.3f}, "
                f"cap_prog={_mean(timeout_eps, 'capture_progress_reward'):.4f}, "
                f"pred_pos={_mean(timeout_eps, 'expert_pred_error_pos_mean'):.4f}, "
                f"pred_next={_mean(timeout_eps, 'expert_pred_error_next_mean'):.4f}"
            )
        if capture_eps:
            print(
                f"  Capture avg   : mindist={_mean(capture_eps, 'min_target_dist_seen'):.3f}, "
                f"dmin={_mean(capture_eps, 'd_i_min'):.3f}, "
                f"ahead={_mean(capture_eps, 'n_agents_ahead_mean'):.2f}, "
                f"team={_mean(capture_eps, 'phi_team'):.3f}, "
                f"spread={_mean(capture_eps, 'phi_spread'):.3f}, "
                f"cap_prog={_mean(capture_eps, 'capture_progress_reward'):.4f}, "
                f"pred_pos={_mean(capture_eps, 'expert_pred_error_pos_mean'):.4f}, "
                f"pred_next={_mean(capture_eps, 'expert_pred_error_next_mean'):.4f}"
            )
        print(f"{'='*60}")
        if collect_success_dataset:
            print(
                f"  Expert dataset: chunks={saved_dataset_chunks} "
                f"| success_episodes={saved_dataset_episodes} "
                f"| success_steps={saved_dataset_steps} "
                f"| dir={dataset_dir}"
            )
        if collect_tp_dataset:
            print(
                f"  TP dataset    : chunks={saved_tp_chunks} "
                f"| samples={saved_tp_samples} "
                f"| dir={tp_dataset_dir}"
            )
    print(f"\n  Videos saved:")
    for p in video_paths:
        print(f"    {p}")

    simulation_app.close()


if __name__ == "__main__":
    main()
