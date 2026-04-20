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
"""

import argparse
import collections
import datetime
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
                            choices=["noise", "tp_net"])
_extra_parser.add_argument("--tp_weight",   default="")
_extra_parser.add_argument("--n_video",     type=int, default=2)
_extra_parser.add_argument("--n_generic",   type=int, default=100)
_extra_parser.add_argument("--v_prey_test", type=float, default=1.5)
_extra_parser.add_argument("--v_drone_test", type=float, default=1.5)
_extra_parser.add_argument("--video_dir",   default="eval_videos/expert")
_extra_parser.add_argument("--video_seed_base", type=int, default=0)
_extra_parser.add_argument("--random_init", type=lambda x: x.lower() != "false",
                            default=True)
_extra_parser.add_argument("--episode_length", type=int, default=1000,
                            help="Environment episode length / timeout steps.")
_extra_parser.add_argument("--max_steps",   type=int, default=1000,
                            help="Hard runner cap; should normally match episode_length.")
_extra_args, _remaining_argv = _extra_parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining_argv

logging.basicConfig(level=logging.INFO, format="%(asctime)s[%(levelname)s] %(message)s")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from expert_strategy_test import EnvCfg as SharedExpertCfg
from expert_strategy_test import ExpertPolicy as SharedExpertPolicy


# ──────────────────────────────────────────────────────────────
#  Utility helpers
# ──────────────────────────────────────────────────────────────
def safe_normalize(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    norm = v.norm(dim=-1, keepdim=True).clamp(min=eps)
    return v / norm


def clip_vector_norm(v: torch.Tensor, max_norm: float) -> torch.Tensor:
    n = v.norm(dim=-1, keepdim=True)
    return v * (n.clamp(max=max_norm) / n.clamp(min=1e-6))


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
    We convert that to the raw action expected by the training-time
    PIDRate transform, so the pursuers use the same 6DoF dynamics path
    as RL training.

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
        tp = target_pos_w[0]   # [1, 3]
        tv = target_vel_3[0]   # [1, 3]
        target_dist_now = torch.norm(dp - tp[0], dim=-1).min().item()
        goal_dist_now = torch.norm(tp[0, :2] - base_env.goal_region_center[:2]).item()
        min_target_dist_seen = min(min_target_dist_seen, target_dist_now)
        min_goal_dist_seen = min(min_goal_dist_seen, goal_dist_now)

        # ── expert: thrust ratio + body-rate command ─────────
        target_next_pos = tp + tv * dt
        try:
            t_cmd, omega_cmd, vel_cmd = expert.get_actions(
                step=step_count,
                drone_pos=dp,
                drone_vel=dv,
                target_pos=tp,
                target_vel=tv,
                target_next_pos=target_next_pos,
                drone_quat=dq,
            )
        except Exception as e:
            logging.warning(f"Expert error at step {step_count}: {e}")
            t_cmd = torch.full((n_agents,), expert.cfg.hover_thrust_ratio, device=device)
            omega_cmd = torch.zeros(n_agents, 3, device=device)
            vel_cmd = torch.zeros(n_agents, 3, device=device)

        # ── convert (t, omega) → raw PIDrate action ───────────
        pidrate_action = expert.t_omega_to_pidrate_raw(
            t_cmd,
            omega_cmd,
            max_body_rate_rad_s=expert.cfg.max_body_rate_rad_s,
        )

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
        }

    return frames, ep_info


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
    tp_weight   = args.tp_weight
    n_video     = args.n_video
    n_generic   = args.n_generic
    v_prey_test = args.v_prey_test
    v_drone_test = args.v_drone_test
    video_dir   = args.video_dir
    video_seed_base = args.video_seed_base
    random_init = args.random_init
    episode_length = args.episode_length
    max_steps   = min(args.max_steps, episode_length)

    # headless=True + enable_render(True) → offscreen rendering, no display needed
    # do NOT force headless=False (crashes without X server)

    # ── single env (visualization) ────────────────────────────
    cfg.task.num_envs     = 1
    cfg.task.env.num_envs = 1
    if hasattr(cfg, "env"):
        cfg.env.num_envs = 1
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
    transforms.append(PIDRateController(controller))
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
    if hasattr(base_env, "history_step"):
        expert_cfg.history_step = int(base_env.history_step)
    if hasattr(base_env, "future_predcition_step"):
        expert_cfg.future_predcition_step = int(base_env.future_predcition_step)
    if hasattr(base_env, "window_step"):
        expert_cfg.window_step = int(base_env.window_step)

    expert = SharedExpertPolicy(
        expert_cfg,
        pred_mode=pred_mode,
        tp_weight_path=tp_weight,
        tp_device=str(device),
    )
    if pred_mode == "tp_net":
        print(f"[Expert] TP_net loaded: {tp_weight}")

    # ── enable offscreen rendering ────────────────────────────
    base_env.enable_render(True)
    base_env.eval()
    env.eval()

    os.makedirs(video_dir, exist_ok=True)
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

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
    print(f"             pred_mode={pred_mode}  v_drone={cfg.task.v_drone}  v_prey={v_prey_test}")
    print(f"             hover_thrust_ratio={hover_thrust_ratio:.3f}  max_body_rate={expert_cfg.max_body_rate_rad_s:.3f} rad/s")

    # ═══════════════════════════════════════════════════════════
    #  Part 1: Record n_video videos (random init, normal v_prey)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  Part 1: Recording {n_video} videos")
    print(f"  pred_mode={pred_mode}, v_prey={cfg.task.v_prey}")
    print(f"{'='*60}")

    video_paths = []
    for vid_idx in range(n_video):
        video_seed = video_seed_base + vid_idx * 100
        env.set_seed(video_seed)
        expert.reset()
        frames, ep_info = run_episode(
            env, base_env, expert,
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
    for ep_idx in range(n_generic):
        env.set_seed(ep_idx * 7 + 999)
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

    # ── summary ───────────────────────────────────────────────
    n_total  = len(gen_results)
    n_cap    = sum(1 for r in gen_results if r["success"])
    n_goal   = sum(1 for r in gen_results if r["goal"])
    n_landed = sum(1 for r in gen_results if r["landed"])
    n_to     = sum(1 for r in gen_results if r["timeout"])
    cap_steps= [r["steps"] for r in gen_results if r["success"]]

    print(f"\n{'='*60}")
    print(f"  Generalization Summary [v_prey={v_prey_test}]")
    print(f"{'='*60}")
    print(f"  Episodes      : {n_total}")
    print(f"  ✅ Capture    : {n_cap/n_total:.0%}  ({n_cap})")
    print(f"  ❌ Goal zone  : {n_goal/n_total:.0%}  ({n_goal})")
    print(f"  ⚠️  Landed    : {n_landed/n_total:.0%}  ({n_landed})")
    print(f"  ⏱  Timeout    : {n_to/n_total:.0%}  ({n_to})")
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
            f"cap_prog={_mean(timeout_eps, 'capture_progress_reward'):.4f}"
        )
    if capture_eps:
        print(
            f"  Capture avg   : mindist={_mean(capture_eps, 'min_target_dist_seen'):.3f}, "
            f"dmin={_mean(capture_eps, 'd_i_min'):.3f}, "
            f"ahead={_mean(capture_eps, 'n_agents_ahead_mean'):.2f}, "
            f"team={_mean(capture_eps, 'phi_team'):.3f}, "
            f"spread={_mean(capture_eps, 'phi_spread'):.3f}, "
            f"cap_prog={_mean(capture_eps, 'capture_progress_reward'):.4f}"
        )
    print(f"{'='*60}")
    print(f"\n  Videos saved:")
    for p in video_paths:
        print(f"    {p}")

    simulation_app.close()


if __name__ == "__main__":
    main()
