import argparse
import datetime
import glob
import logging
import os
import random
import sys
from typing import Dict, List, Optional

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter
from torchrl.envs.transforms import InitTracker, Compose, TransformedEnv

from omni_drones import CONFIG_PATH, init_simulation_app
from omni_drones.learning import MAPPOPolicy
from omni_drones.utils.torchrl import AgentSpec
from omni_drones.utils.torchrl.transforms import (
    FromDiscreteAction,
    FromMultiDiscreteAction,
    History,
    ravel_composite,
)


_extra_parser = argparse.ArgumentParser(add_help=False)
_extra_parser.add_argument("--model_dir", required=True)
_extra_parser.add_argument("--pred_mode", default="tp_net", choices=["noise", "tp_net"])
_extra_parser.add_argument("--tp_weight", default="")
_extra_parser.add_argument("--waves", type=int, default=8)
_extra_parser.add_argument("--batch_envs", type=int, default=128)
_extra_parser.add_argument("--episode_length", type=int, default=1000)
_extra_parser.add_argument("--v_prey_test", type=float, default=1.5)
_extra_parser.add_argument("--v_drone_test", type=float, default=1.5)
_extra_parser.add_argument("--bc_lr", type=float, default=1e-4)
_extra_parser.add_argument("--action_mse_coef", type=float, default=1.0)
_extra_parser.add_argument("--log_prob_coef", type=float, default=0.0)
_extra_parser.add_argument("--aux_vel_cmd_coef", type=float, default=0.0)
_extra_parser.add_argument("--aux_waypoint_coef", type=float, default=0.02)
_extra_parser.add_argument("--aux_target_pos_coef", type=float, default=0.0)
_extra_parser.add_argument("--aux_forward_dir_coef", type=float, default=0.0)
_extra_parser.add_argument("--aux_assignment_coef", type=float, default=0.05)
_extra_parser.add_argument("--aux_trap_coef", type=float, default=0.01)
_extra_parser.add_argument("--entropy_bonus_coef", type=float, default=0.0)
_extra_parser.add_argument("--accum_batch_size", type=int, default=2048)
_extra_parser.add_argument("--expert_mix_prob_init", type=float, default=0.25)
_extra_parser.add_argument("--expert_mix_prob_final", type=float, default=0.05)
_extra_parser.add_argument("--min_bc_altitude", type=float, default=0.0)
_extra_parser.add_argument("--replay_dataset_dir", default="")
_extra_parser.add_argument("--replay_updates_per_wave", type=int, default=0)
_extra_parser.add_argument("--replay_batch_size", type=int, default=4096)
_extra_parser.add_argument("--replay_front_weight_alpha", type=float, default=0.0)
_extra_parser.add_argument("--online_goal_weight_alpha", type=float, default=0.0)
_extra_parser.add_argument("--online_goal_weight_radius", type=float, default=1.5)
_extra_parser.add_argument("--online_disagreement_weight_alpha", type=float, default=0.0)
_extra_parser.add_argument("--online_success_only", type=lambda x: x.lower() != "false", default=False)
_extra_parser.add_argument("--goal_rescue_radius", type=float, default=0.0)
_extra_parser.add_argument("--altitude_rescue_threshold", type=float, default=0.0)
_extra_parser.add_argument("--rescue_horizon", type=int, default=0)
_extra_parser.add_argument("--eval_every", type=int, default=2)
_extra_parser.add_argument("--n_eval", type=int, default=128)
_extra_parser.add_argument("--eval_batch_envs", type=int, default=128)
_extra_parser.add_argument("--save_dir", default="checkpoints")
_extra_parser.add_argument("--save_tag", default="actor_dagger")
_extra_parser.add_argument("--device", default="")
_extra_parser.add_argument("--seed_base", type=int, default=20260404)
_extra_parser.add_argument("--enable_actor_rnn", type=lambda x: x.lower() != "false", default=False)
_extra_parser.add_argument("--actor_rnn_hidden_size", type=int, default=128)
_extra_parser.add_argument("--actor_rnn_train_seq_len", type=int, default=8)
_extra_parser.add_argument("--enable_prev_action_conditioning", type=lambda x: x.lower() != "false", default=False)
_extra_parser.add_argument("--prev_action_condition_hidden_dim", type=int, default=128)
_extra_parser.add_argument("--train_prev_action_only", type=lambda x: x.lower() != "false", default=False)
_extra_args, _remaining_argv = _extra_parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining_argv

from expert_isaac_eval import BatchedExpertPolicy, SharedExpertCfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def checkpoint_has_bc_aux(state_dict) -> bool:
    if bool(state_dict.get("actor_has_bc_aux", False)):
        return True
    actor_params = state_dict.get("actor_params", None)
    if actor_params is None:
        return False
    keys = actor_params.to_tensordict().keys(True, True)
    return any("aux_heads" in "/".join(map(str, key)) for key in keys)


def checkpoint_actor_rnn_hidden_size(state_dict) -> Optional[int]:
    if int(state_dict.get("actor_rnn_hidden_size", 0)) > 0:
        return int(state_dict["actor_rnn_hidden_size"])
    actor_params = state_dict.get("actor_params", None)
    if actor_params is None:
        return None
    actor_td = actor_params.to_tensordict()
    for key in actor_td.keys(True, True):
        key_str = "/".join(map(str, key))
        if "rnn" not in key_str or not key_str.endswith("cell.weight_ih"):
            continue
        weight = actor_td.get(key)
        if weight.ndim == 2 and weight.shape[0] % 3 == 0:
            return int(weight.shape[0] // 3)
    return None


def enable_actor_rnn_cfg(cfg, hidden_size: int, train_seq_len: int):
    cfg.algo.actor.rnn = OmegaConf.create(
        {
            "cls": "gru",
            "kwargs": {"hidden_size": int(hidden_size)},
            "train_seq_len": int(train_seq_len),
        }
    )


def checkpoint_prev_action_condition_hidden_size(state_dict) -> Optional[int]:
    if bool(state_dict.get("actor_has_prev_action_conditioning", False)):
        hidden_dim = int(state_dict.get("actor_prev_action_condition_hidden_dim", 0))
        return hidden_dim if hidden_dim > 0 else 128
    actor_params = state_dict.get("actor_params", None)
    if actor_params is None:
        return None
    actor_td = actor_params.to_tensordict()
    for key in actor_td.keys(True, True):
        key_str = "/".join(map(str, key))
        if "prev_action_conditioner" not in key_str or not key_str.endswith("0.weight"):
            continue
        weight = actor_td.get(key)
        if weight.ndim == 2:
            return int(weight.shape[0])
    return None


def enable_prev_action_conditioning_cfg(cfg, hidden_dim: int):
    cfg.algo.actor.prev_action_conditioning = OmegaConf.create(
        {
            "enabled": True,
            "hidden_dim": int(hidden_dim),
        }
    )


def find_latest_tp_weight(project_root: str):
    candidates = sorted(
        glob.glob(os.path.join(project_root, "checkpoints", "**", "tp_only_*.pt"), recursive=True)
    )
    return candidates[-1] if candidates else None


def maybe_tuple_key(key_str: str):
    parts = tuple(p for p in str(key_str).split("/") if p)
    return parts[0] if len(parts) == 1 else parts


def build_obs_tensordict_from_storage(obs_storage: dict, indices: torch.Tensor, device: str, n_agents: int):
    obs_data = {}
    for key_str, value in obs_storage.items():
        obs_data[maybe_tuple_key(key_str)] = value[indices].to(device=device, dtype=torch.float32)
    return TensorDict(obs_data, batch_size=[indices.numel(), n_agents], device=device)


def build_step_weights(episode_lengths: torch.Tensor, front_weight_alpha: float = 0.0) -> torch.Tensor:
    if episode_lengths.numel() == 0:
        return torch.empty(0, dtype=torch.float32)
    weights = []
    for length in episode_lengths.tolist():
        length = int(length)
        w = torch.ones(length, dtype=torch.float32)
        if front_weight_alpha > 0.0:
            if length == 1:
                front = torch.full((1,), 1.0 + front_weight_alpha, dtype=torch.float32)
            else:
                progress = torch.linspace(0.0, 1.0, steps=length, dtype=torch.float32)
                front = 1.0 + front_weight_alpha * (1.0 - progress)
            w *= front
        weights.append(w)
    flat = torch.cat(weights, dim=0)
    return flat / flat.mean().clamp_min(1e-8)


def sample_replay_batch(chunk_files, device, n_agents: int, replay_batch_size: int, front_weight_alpha: float = 0.0):
    chunk_path = random.choice(chunk_files)
    chunk = torch.load(chunk_path, map_location="cpu")
    action_raw = chunk["action_raw"]
    num_steps = int(action_raw.shape[0])
    if num_steps == 0:
        return None
    indices = torch.randint(0, num_steps, (min(replay_batch_size, num_steps),), dtype=torch.long)
    obs_td = build_obs_tensordict_from_storage(chunk["obs"], indices, device, n_agents)
    action_batch = action_raw[indices].to(device=device, dtype=torch.float32)
    step_weights = build_step_weights(
        chunk["episode_lengths"].to(dtype=torch.long),
        front_weight_alpha=float(front_weight_alpha),
    )
    weight_batch = step_weights[indices].to(device=device, dtype=torch.float32)
    batch_td = TensorDict(
        {
            ("agents", "observation"): obs_td,
            ("agents", "prev_action"): torch.zeros_like(action_batch),
            ("agents", "action"): action_batch,
            "bc_weight": weight_batch,
        },
        batch_size=[indices.numel()],
        device=device,
    )
    aux_storage = chunk.get("expert_aux", {})
    for key, value in aux_storage.items():
        aux_value = value[indices]
        if aux_value.dtype.is_floating_point:
            aux_value = aux_value.to(device=device, dtype=torch.float32)
        else:
            aux_value = aux_value.to(device=device)
        batch_td["expert_aux", key] = aux_value
    return batch_td


def make_env(cfg):
    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    transforms = [InitTracker()]
    if cfg.task.get("flatten_obs", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation")))
    if cfg.task.get("flatten_state", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "state")))
    if (
        cfg.task.get("flatten_intrinsics", True)
        and ("agents", "intrinsics") in base_env.observation_spec.keys(True)
    ):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "intrinsics"), start_dim=-1))
    if cfg.task.get("history", False):
        transforms.append(History([("agents", "observation")], steps=4))

    action_transform = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transforms.append(FromMultiDiscreteAction(nbins=nbins))
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transforms.append(FromDiscreteAction(nbins=nbins))
        elif action_transform == "velocity":
            from omni_drones.controllers import LeePositionController
            from omni_drones.utils.torchrl.transforms import VelController

            controller = LeePositionController(9.81, base_env.drone.params).to(base_env.device)
            transforms.append(VelController(controller))
        elif action_transform == "attitude":
            from omni_drones.controllers import AttitudeController as Controller
            from omni_drones.utils.torchrl.transforms import AttitudeController

            controller = Controller(9.81, base_env.drone.params).to(base_env.device)
            transforms.append(AttitudeController(controller))
        elif action_transform == "rate":
            from omni_drones.controllers import RateController as _RateController
            from omni_drones.utils.torchrl.transforms import RateController

            controller = _RateController(9.81, base_env.drone.params).to(base_env.device)
            transforms.append(RateController(controller))
        elif action_transform == "PIDrate":
            from omni_drones.controllers import PIDRateController as _PIDRateController
            from omni_drones.utils.torchrl.transforms import PIDRateController

            controller = _PIDRateController(cfg.sim.dt, 9.81, base_env.drone.params).to(base_env.device)
            transforms.append(PIDRateController(controller))
        elif not action_transform.lower() == "none":
            raise NotImplementedError(f"Unknown action transform: {action_transform}")
    env = TransformedEnv(base_env, Compose(*transforms))
    return env, base_env, controller


def build_expert_cfg(base_env, controller, cfg) -> SharedExpertCfg:
    device = base_env.device
    hover_thrust_ratio = float(
        (base_env.drone.gravity[0, 0] / controller.max_thrusts.sum()).item()
    )
    expert_cfg = SharedExpertCfg()
    expert_cfg.dt = float(base_env.dt)
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
    expert_cfg.max_body_rate_rad_s = np.deg2rad(180.0 * expert_cfg.target_clip)
    if hasattr(base_env, "history_step"):
        expert_cfg.history_step = int(base_env.history_step)
    if hasattr(base_env, "future_predcition_step"):
        expert_cfg.future_predcition_step = int(base_env.future_predcition_step)
    if hasattr(base_env, "window_step"):
        expert_cfg.window_step = int(base_env.window_step)
    return expert_cfg


def tensor_mean(items: List[Dict[str, float]], key: str) -> float:
    vals = [item[key] for item in items if key in item]
    return float(np.mean(vals)) if vals else float("nan")


def make_bc_batch(obs_td, prev_action, expert_action, expert_debug, expert_vel_cmd, device, bc_weight=None):
    obs_clone = TensorDict({}, batch_size=obs_td.batch_size, device=device)
    for key in obs_td.keys(True, True):
        value = obs_td.get(key).detach().clone()
        if torch.is_floating_point(value):
            value = value.to(device=device, dtype=torch.float32)
        else:
            value = value.to(device=device)
        obs_clone.set(key, value)

    batch_td = TensorDict(
        {
            ("agents", "observation"): obs_clone,
            ("agents", "prev_action"): prev_action.detach().clone().to(device=device, dtype=torch.float32),
            ("agents", "action"): expert_action.detach().clone().to(device=device, dtype=torch.float32),
        },
        batch_size=[expert_action.shape[0]],
        device=device,
    )
    batch_td["expert_aux", "vel_cmd"] = expert_vel_cmd.detach().clone().to(device=device, dtype=torch.float32)
    batch_td["expert_aux", "assignment"] = expert_debug["assignment"].detach().clone().to(device=device, dtype=torch.long)
    batch_td["expert_aux", "waypoint"] = expert_debug["waypoint"].detach().clone().to(device=device, dtype=torch.float32)
    if "target_pos_pred" in expert_debug:
        batch_td["expert_aux", "target_pos_pred"] = expert_debug["target_pos_pred"].detach().clone().to(device=device, dtype=torch.float32)
    if "forward_dir" in expert_debug:
        batch_td["expert_aux", "forward_dir"] = expert_debug["forward_dir"].detach().clone().to(device=device, dtype=torch.float32)
    batch_td["expert_aux", "trap_mode"] = expert_debug["trap_mode"].detach().clone().to(device=device, dtype=torch.float32)
    if bc_weight is None:
        bc_weight = torch.ones(expert_action.shape[0], device=device, dtype=torch.float32)
    batch_td["bc_weight"] = bc_weight.detach().clone().to(device=device, dtype=torch.float32)
    return batch_td


def split_episode_into_seq_chunks(
    step_batches: List[TensorDict],
    seq_len: int,
    keep_tail: bool = False,
):
    if not step_batches:
        return []
    episode_td = torch.cat(step_batches, dim=0)
    episode_len = int(episode_td.batch_size[0])
    chunks = []
    for start in range(0, episode_len, seq_len):
        end = min(start + seq_len, episode_len)
        chunk_len = end - start
        if chunk_len < seq_len and not keep_tail:
            continue
        chunk_td = episode_td[start:end].clone()
        if chunk_len < seq_len:
            pad_td = chunk_td[-1:].expand(seq_len - chunk_len).clone()
            if "bc_weight" in pad_td.keys():
                pad_td["bc_weight"].zero_()
            chunk_td = torch.cat([chunk_td, pad_td], dim=0)
        is_init = torch.zeros(1, seq_len, 1, dtype=torch.bool)
        is_init[:, 0] = True
        chunk_td = chunk_td.unsqueeze(0)
        chunk_td["is_init"] = is_init
        chunks.append(chunk_td)
    return chunks


def carry_actor_rnn_state(policy, src_td: TensorDict, dst_td: TensorDict):
    rnn_key = f"{policy.agent_spec.name}.actor_rnn_state"
    if rnn_key in src_td.keys():
        dst_td[rnn_key] = src_td[rnn_key].detach()


def maybe_train_prev_action_branch_only(policy, lr: float):
    trainable_params = []
    frozen = 0
    for name, param in policy.actor_params.named_parameters():
        trainable = "prev_action_conditioner" in name
        param.requires_grad_(trainable)
        if trainable:
            trainable_params.append(param)
        else:
            frozen += int(param.numel())
    if not trainable_params:
        raise RuntimeError("train_prev_action_only requested but no prev_action_conditioner params were found.")
    policy.actor_opt = torch.optim.Adam(trainable_params, lr=float(lr))
    return {
        "trainable_count": int(sum(p.numel() for p in trainable_params)),
        "frozen_count": int(frozen),
    }


def flush_bc_buffer(policy, buffer, args):
    if not buffer:
        return None
    batch = torch.cat(buffer, dim=0)
    info = policy.update_actor_bc(
        batch,
        entropy_bonus_coef=float(args.entropy_bonus_coef),
        action_mse_coef=float(args.action_mse_coef),
        log_prob_coef=float(args.log_prob_coef),
        aux_vel_cmd_coef=float(args.aux_vel_cmd_coef),
        aux_waypoint_coef=float(args.aux_waypoint_coef),
        aux_target_pos_coef=float(args.aux_target_pos_coef),
        aux_forward_dir_coef=float(args.aux_forward_dir_coef),
        aux_assignment_coef=float(args.aux_assignment_coef),
        aux_trap_coef=float(args.aux_trap_coef),
    )
    buffer.clear()
    return info


@torch.no_grad()
def summarize_results(results: List[Dict[str, float]]):
    n_total = len(results)
    n_cap = sum(1 for r in results if r["success"])
    n_goal = sum(1 for r in results if r["goal"])
    n_landed = sum(1 for r in results if r["landed"])
    n_timeout = n_total - n_cap - n_goal - n_landed
    cap_steps = [r["steps"] for r in results if r["success"]]
    return {
        "episodes": n_total,
        "capture_rate": n_cap / max(n_total, 1),
        "goal_rate": n_goal / max(n_total, 1),
        "landed_rate": n_landed / max(n_total, 1),
        "timeout_rate": n_timeout / max(n_total, 1),
        "capture_steps_mean": float(np.mean(cap_steps)) if cap_steps else float("nan"),
    }


def run_dagger_wave(env, base_env, policy, expert, args, mix_prob: float):
    device = base_env.device
    n_envs = int(base_env.num_envs)
    n_agents = int(base_env.drone.n)
    tensordict = env.reset()
    tensordict[("agents", "prev_action")] = torch.zeros(
        n_envs, n_agents, 4, device=device, dtype=torch.float32
    )
    done_mask = torch.zeros(n_envs, dtype=torch.bool, device=device)
    results = [None for _ in range(n_envs)]
    min_target_dist_seen = [float("inf")] * n_envs
    min_goal_dist_seen = [float("inf")] * n_envs
    expert.reset(n_envs)
    hover_action = expert.t_omega_to_pidrate_raw(
        torch.full(
            (n_envs * n_agents,),
            float(expert.cfg.hover_thrust_ratio),
            device=device,
        ),
        torch.zeros(n_envs * n_agents, 3, device=device),
    ).reshape(n_envs, n_agents, 4)

    bc_buffer = []
    bc_infos = []
    buffered = 0
    use_sequence_buffers = bool(args.online_success_only) or hasattr(policy, "minibatch_seq_len")
    episode_buffers = [[] for _ in range(n_envs)] if use_sequence_buffers else None
    rescue_countdown = torch.zeros(n_envs, dtype=torch.long, device=device)
    total_active_env_steps = 0
    total_bc_env_steps = 0
    total_rescue_env_steps = 0

    for step_count in range(int(args.episode_length)):
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

        target_next_pos = target_pos_w + target_vel_3 * float(base_env.dt)
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
        expert_action = expert.t_omega_to_pidrate_raw(
            t_cmd.reshape(-1),
            omega_cmd.reshape(-1, 3),
        ).reshape(n_envs, n_agents, 4)

        active_mask = ~done_mask
        total_active_env_steps += int(active_mask.sum().item())
        min_altitude = drone_pos_w[..., 2].min(dim=1).values

        actor_out = policy(tensordict, deterministic=True)
        actor_action = actor_out[("agents", "action")].detach().clone()

        rescue_trigger = torch.zeros(n_envs, dtype=torch.bool, device=device)
        if int(args.rescue_horizon) > 0:
            if float(args.goal_rescue_radius) > 0.0:
                rescue_trigger = rescue_trigger | (
                    active_mask & (goal_dist_all <= float(args.goal_rescue_radius))
                )
            if float(args.altitude_rescue_threshold) > 0.0:
                rescue_trigger = rescue_trigger | (
                    active_mask & (min_altitude <= float(args.altitude_rescue_threshold))
                )
            rescue_countdown[rescue_trigger] = int(args.rescue_horizon)
        rescue_mask = active_mask & (rescue_countdown > 0)
        total_rescue_env_steps += int(rescue_mask.sum().item())

        bc_mask = active_mask
        if float(args.min_bc_altitude) > 0.0:
            bc_mask = bc_mask & (min_altitude >= float(args.min_bc_altitude))
        if bool(bc_mask.any()):
            obs_batch = tensordict.get(("agents", "observation"))[bc_mask].clone()
            prev_action_batch = tensordict.get(("agents", "prev_action"))[bc_mask].clone()
            bc_weight = None
            if float(args.online_goal_weight_alpha) > 0.0:
                goal_dist_sel = goal_dist_all[bc_mask]
                radius = max(float(args.online_goal_weight_radius), 1e-6)
                urgency = ((radius - goal_dist_sel) / radius).clamp(0.0, 1.0)
                bc_weight = 1.0 + float(args.online_goal_weight_alpha) * urgency
            if float(args.online_disagreement_weight_alpha) > 0.0:
                disagreement = (actor_action - expert_action).pow(2).mean(dim=(-1, -2))
                disagreement = disagreement[bc_mask]
                disagreement_weight = 1.0 + float(args.online_disagreement_weight_alpha) * disagreement
                bc_weight = disagreement_weight if bc_weight is None else (bc_weight * disagreement_weight)
            if bool(rescue_mask.any()):
                rescue_weight = torch.where(
                    rescue_mask[bc_mask],
                    torch.full(
                        (int(bc_mask.sum().item()),),
                        1.5,
                        device=device,
                        dtype=torch.float32,
                    ),
                    torch.ones(int(bc_mask.sum().item()), device=device, dtype=torch.float32),
                )
                bc_weight = rescue_weight if bc_weight is None else (bc_weight * rescue_weight)
            if episode_buffers is not None:
                bc_env_ids = torch.nonzero(bc_mask, as_tuple=False).squeeze(-1).tolist()
                for local_idx, env_idx in enumerate(bc_env_ids):
                    weight_i = None
                    if bc_weight is not None:
                        weight_i = bc_weight[local_idx : local_idx + 1].detach().clone().to(
                            device="cpu", dtype=torch.float32
                        )
                    single_batch = make_bc_batch(
                        obs_batch[local_idx : local_idx + 1],
                        prev_action_batch[local_idx : local_idx + 1],
                        expert_action[env_idx : env_idx + 1],
                        {
                            "assignment": expert_debug["assignment"][env_idx : env_idx + 1],
                            "waypoint": expert_debug["waypoint"][env_idx : env_idx + 1],
                            "target_pos_pred": expert_debug["target_pos_pred"][env_idx : env_idx + 1],
                            "forward_dir": expert_debug["forward_dir"][env_idx : env_idx + 1],
                            "trap_mode": expert_debug["trap_mode"][env_idx : env_idx + 1],
                        },
                        vel_cmd_expert[env_idx : env_idx + 1],
                        "cpu",
                        bc_weight=weight_i,
                    )
                    episode_buffers[env_idx].append(single_batch)
            else:
                bc_batch = make_bc_batch(
                    obs_batch,
                    prev_action_batch,
                    expert_action[bc_mask],
                    {
                        "assignment": expert_debug["assignment"][bc_mask],
                        "waypoint": expert_debug["waypoint"][bc_mask],
                        "target_pos_pred": expert_debug["target_pos_pred"][bc_mask],
                        "forward_dir": expert_debug["forward_dir"][bc_mask],
                        "trap_mode": expert_debug["trap_mode"][bc_mask],
                    },
                    vel_cmd_expert[bc_mask],
                    device,
                    bc_weight=bc_weight,
                )
                bc_buffer.append(bc_batch)
                buffered += int(bc_mask.sum().item())
                total_bc_env_steps += int(bc_mask.sum().item())

        if episode_buffers is None and buffered >= int(args.accum_batch_size):
            info = flush_bc_buffer(policy, bc_buffer, args)
            buffered = 0
            if info is not None:
                bc_infos.append(info)
        action_batch = actor_action

        if mix_prob > 0.0 and bool(active_mask.any()):
            mix_env = (torch.rand(n_envs, device=device) < mix_prob) & active_mask
            action_batch[mix_env] = expert_action[mix_env]
        if bool(rescue_mask.any()):
            action_batch[rescue_mask] = expert_action[rescue_mask]
        if bool(done_mask.any()):
            action_batch[done_mask] = hover_action[done_mask]

        tensordict[("agents", "action")] = action_batch
        tensordict = env.step(tensordict)
        td_next = tensordict.get("next")
        td_next[("agents", "prev_action")] = action_batch.detach()
        carry_actor_rnn_state(policy, actor_out, td_next)
        done_vec = td_next.get("done").reshape(n_envs).bool()
        stats = td_next.get("stats")

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
            }
            if episode_buffers is not None:
                keep_episode = (not bool(args.online_success_only)) or success
                if keep_episode and episode_buffers[env_idx]:
                    if hasattr(policy, "minibatch_seq_len"):
                        seq_chunks = split_episode_into_seq_chunks(
                            episode_buffers[env_idx],
                            int(policy.minibatch_seq_len),
                            keep_tail=False,
                        )
                        total_bc_env_steps += sum(
                            int(chunk.batch_size[1]) for chunk in seq_chunks
                        )
                        bc_buffer.extend(seq_chunks)
                        buffered += len(seq_chunks) * int(policy.minibatch_seq_len)
                    else:
                        total_bc_env_steps += len(episode_buffers[env_idx])
                        bc_buffer.extend(episode_buffers[env_idx])
                        buffered += len(episode_buffers[env_idx])
                episode_buffers[env_idx] = []
            done_mask[env_idx] = True

        tensordict = td_next
        if bool(active_mask.any()) and int(args.rescue_horizon) > 0:
            rescue_countdown[active_mask] = torch.clamp_min(rescue_countdown[active_mask] - 1, 0)
        if episode_buffers is not None and buffered >= int(args.accum_batch_size):
            info = flush_bc_buffer(policy, [batch.to(device) for batch in bc_buffer], args)
            bc_buffer.clear()
            buffered = 0
            if info is not None:
                bc_infos.append(info)
        if bool(done_mask.all()):
            break

    if episode_buffers is not None:
        if bc_buffer:
            info = flush_bc_buffer(policy, [batch.to(device) for batch in bc_buffer], args)
            bc_buffer.clear()
            buffered = 0
            if info is not None:
                bc_infos.append(info)
    else:
        info = flush_bc_buffer(policy, bc_buffer, args)
        if info is not None:
            bc_infos.append(info)

    for env_idx in range(n_envs):
        if results[env_idx] is None:
            results[env_idx] = {
                "steps": int(args.episode_length) - 1,
                "success": False,
                "goal": False,
                "landed": False,
                "timeout": True,
                "min_target_dist_seen": min_target_dist_seen[env_idx],
                "min_goal_dist_seen": min_goal_dist_seen[env_idx],
            }

    summary = summarize_results(results)
    summary["bc_kept_rate"] = total_bc_env_steps / max(total_active_env_steps, 1)
    summary["rescue_rate"] = total_rescue_env_steps / max(total_active_env_steps, 1)
    return summary, bc_infos


@torch.no_grad()
def run_policy_eval(env, base_env, policy, max_steps: int, n_eval: int):
    results = []
    eval_done = 0
    wave = 0
    while eval_done < n_eval:
        wave += 1
        env.set_seed(9000 + wave * 97)
        tensordict = env.reset()
        device = base_env.device
        n_envs = int(base_env.num_envs)
        n_agents = int(base_env.drone.n)
        tensordict[("agents", "prev_action")] = torch.zeros(
            n_envs, n_agents, 4, device=device, dtype=torch.float32
        )
        done_mask = torch.zeros(n_envs, dtype=torch.bool, device=device)
        zero_action = torch.zeros(n_envs, n_agents, 4, device=device)
        min_target_dist_seen = [float("inf")] * n_envs
        min_goal_dist_seen = [float("inf")] * n_envs
        wave_results = [None for _ in range(n_envs)]

        for step_count in range(max_steps):
            drone_pos_w, _ = base_env.get_env_poses(base_env.drone.get_world_poses())
            target_pos_w, _ = base_env.get_env_poses(base_env.target.get_world_poses())
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

            action_td = policy(tensordict, deterministic=True)
            if bool(done_mask.any()):
                action_td["agents", "action"][done_mask] = zero_action[done_mask]
            tensordict = env.step(action_td)
            td_next = tensordict.get("next")
            td_next[("agents", "prev_action")] = action_td[("agents", "action")].detach()
            carry_actor_rnn_state(policy, action_td, td_next)
            done_vec = td_next.get("done").reshape(n_envs).bool()
            stats = td_next.get("stats")

            for env_idx in range(n_envs):
                if done_mask[env_idx] or not done_vec[env_idx]:
                    continue
                success = bool(stats["success"][env_idx].item() > 0)
                goal = bool(stats["goal_reached"][env_idx].item() > 0)
                landed = bool(stats["any_landed"][env_idx].item() > 0)
                wave_results[env_idx] = {
                    "steps": step_count,
                    "success": success,
                    "goal": goal,
                    "landed": landed,
                    "timeout": not success and not goal and not landed,
                    "min_target_dist_seen": min_target_dist_seen[env_idx],
                    "min_goal_dist_seen": min_goal_dist_seen[env_idx],
                }
                done_mask[env_idx] = True
            if bool(done_mask.all()):
                break
            tensordict = td_next

        for env_idx in range(n_envs):
            if wave_results[env_idx] is None:
                wave_results[env_idx] = {
                    "steps": max_steps - 1,
                    "success": False,
                    "goal": False,
                    "landed": False,
                    "timeout": True,
                    "min_target_dist_seen": min_target_dist_seen[env_idx],
                    "min_goal_dist_seen": min_goal_dist_seen[env_idx],
                }
            if eval_done < n_eval:
                results.append(wave_results[env_idx])
                eval_done += 1
    return summarize_results(results)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg):
    args = _extra_args
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    set_seed(cfg.seed)
    cfg.headless = True
    if args.device:
        cfg.sim.device = args.device
    cfg.task.num_envs = int(args.batch_envs)
    cfg.task.env.num_envs = int(args.batch_envs)
    if hasattr(cfg, "env"):
        cfg.env.num_envs = int(args.batch_envs)
        cfg.env.max_episode_length = int(args.episode_length)
    cfg.task.max_episode_length = int(args.episode_length)
    cfg.task.env.max_episode_length = int(args.episode_length)
    cfg.task.v_drone = float(args.v_drone_test)
    cfg.task.use_eval = 0

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_state = torch.load(args.model_dir, map_location="cpu", weights_only=False)
    checkpoint_rnn_hidden = checkpoint_actor_rnn_hidden_size(checkpoint_state)
    checkpoint_prev_action_hidden = checkpoint_prev_action_condition_hidden_size(checkpoint_state)
    if bool(args.enable_actor_rnn):
        enable_actor_rnn_cfg(
            cfg,
            hidden_size=int(args.actor_rnn_hidden_size),
            train_seq_len=int(args.actor_rnn_train_seq_len),
        )
    elif checkpoint_rnn_hidden is not None:
        enable_actor_rnn_cfg(
            cfg,
            hidden_size=int(checkpoint_rnn_hidden),
            train_seq_len=int(args.actor_rnn_train_seq_len),
        )
    if bool(args.enable_prev_action_conditioning):
        enable_prev_action_conditioning_cfg(
            cfg,
            hidden_dim=int(args.prev_action_condition_hidden_dim),
        )
    elif checkpoint_prev_action_hidden is not None:
        enable_prev_action_conditioning_cfg(
            cfg,
            hidden_dim=int(checkpoint_prev_action_hidden),
        )
    use_bc_aux = checkpoint_has_bc_aux(checkpoint_state) or any(
        float(v) != 0.0
        for v in (
            args.aux_vel_cmd_coef,
            args.aux_waypoint_coef,
            args.aux_target_pos_coef,
            args.aux_forward_dir_coef,
            args.aux_assignment_coef,
            args.aux_trap_coef,
        )
    )
    cfg.algo.actor.bc_aux.enabled = bool(use_bc_aux)

    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_root = args.save_dir
    if not os.path.isabs(save_root):
        save_root = os.path.join(project_root, save_root)
    save_dir = os.path.join(save_root, f"{args.save_tag}_{time_str}")
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(project_root, "runs", f"{args.save_tag}_{time_str}"))

    simulation_app = init_simulation_app(cfg)
    env, base_env, controller = make_env(cfg)
    env.set_seed(cfg.seed)
    base_env.enable_render(False)

    if hasattr(base_env, "_set_curriculum_stage"):
        base_env._set_curriculum_stage(
            len(base_env.curriculum_stages) - 1,
            reset_metrics=False,
            announce=False,
        )
    base_env.current_target_speed = float(args.v_prey_test)
    base_env.target_velocity_scale = float(args.v_prey_test)

    tp_weight = args.tp_weight or find_latest_tp_weight(project_root)
    if args.pred_mode == "tp_net" and (not tp_weight or not os.path.isfile(tp_weight)):
        raise FileNotFoundError(f"TP weight not found: {tp_weight}")

    expert_cfg = build_expert_cfg(base_env, controller, cfg)
    expert = BatchedExpertPolicy(
        expert_cfg,
        pred_mode=args.pred_mode,
        tp_weight_path=tp_weight,
        tp_device=str(base_env.device),
    )

    agent_spec: AgentSpec = env.agent_spec["drone"]
    policy = MAPPOPolicy(cfg.algo, agent_spec=agent_spec, device=cfg.sim.device, TP_net=base_env.TP)
    policy.load_state_dict(checkpoint_state)
    for group in policy.actor_opt.param_groups:
        group["lr"] = float(args.bc_lr)
    if bool(args.train_prev_action_only):
        freeze_info = maybe_train_prev_action_branch_only(policy, lr=float(args.bc_lr))
        logging.info(
            "DAgger prev-action-only: trainable=%d frozen=%d",
            int(freeze_info["trainable_count"]),
            int(freeze_info["frozen_count"]),
        )
    if hasattr(policy, "critic"):
        policy.critic.eval()
    if hasattr(policy, "TP_net"):
        policy.TP_net.eval()

    logging.info("DAgger init model: %s", args.model_dir)
    logging.info("DAgger TP weight: %s", tp_weight)
    logging.info(
        "DAgger setup: waves=%d batch_envs=%d episode_length=%d v_drone=%.2f v_prey=%.2f "
        "mix_init=%.3f mix_final=%.3f accum_batch_size=%d rescue_horizon=%d goal_rescue=%.2f alt_rescue=%.2f "
        "success_only=%s",
        int(args.waves),
        int(args.batch_envs),
        int(args.episode_length),
        float(args.v_drone_test),
        float(args.v_prey_test),
        float(args.expert_mix_prob_init),
        float(args.expert_mix_prob_final),
        int(args.accum_batch_size),
        int(args.rescue_horizon),
        float(args.goal_rescue_radius),
        float(args.altitude_rescue_threshold),
        bool(args.online_success_only),
    )
    if cfg.algo.actor.get("rnn", None):
        logging.info(
            "DAgger actor-rnn: cls=%s hidden=%d train_seq_len=%d",
            str(cfg.algo.actor.rnn.cls),
            int(cfg.algo.actor.rnn.kwargs.hidden_size),
            int(cfg.algo.actor.rnn.train_seq_len),
        )
    if cfg.algo.actor.get("prev_action_conditioning", None):
        logging.info(
            "DAgger prev-action conditioning: hidden=%d",
            int(cfg.algo.actor.prev_action_conditioning.hidden_dim),
        )
    logging.info(
        "DAgger loss: action_mse=%.4f log_prob=%.4f aux_vel=%.4f aux_waypoint=%.4f "
        "aux_target_pos=%.4f aux_forward_dir=%.4f aux_assignment=%.4f aux_trap=%.4f "
        "goal_weight=%.3f disagree_weight=%.3f",
        float(args.action_mse_coef),
        float(args.log_prob_coef),
        float(args.aux_vel_cmd_coef),
        float(args.aux_waypoint_coef),
        float(args.aux_target_pos_coef),
        float(args.aux_forward_dir_coef),
        float(args.aux_assignment_coef),
        float(args.aux_trap_coef),
        float(args.online_goal_weight_alpha),
        float(args.online_disagreement_weight_alpha),
    )
    replay_chunk_files = []
    if args.replay_dataset_dir:
        replay_dataset_dir = args.replay_dataset_dir
        if not os.path.isabs(replay_dataset_dir):
            replay_dataset_dir = os.path.join(project_root, replay_dataset_dir)
        replay_chunk_files = sorted(glob.glob(os.path.join(replay_dataset_dir, "expert_success_wave_*.pt")))
        if not replay_chunk_files:
            raise FileNotFoundError(f"No replay chunks found in {replay_dataset_dir}")
        logging.info(
            "DAgger replay: dataset=%s chunks=%d updates_per_wave=%d batch_size=%d front_weight_alpha=%.3f",
            replay_dataset_dir,
            len(replay_chunk_files),
            int(args.replay_updates_per_wave),
            int(args.replay_batch_size),
            float(args.replay_front_weight_alpha),
        )

    best_capture = -1.0
    best_path = None

    for wave in range(1, int(args.waves) + 1):
        mix_prob = float(args.expert_mix_prob_init)
        if int(args.waves) > 1:
            alpha = (wave - 1) / max(int(args.waves) - 1, 1)
            mix_prob = (1.0 - alpha) * float(args.expert_mix_prob_init) + alpha * float(args.expert_mix_prob_final)

        env.set_seed(int(args.seed_base) + wave * 97)
        wave_summary, bc_infos = run_dagger_wave(env, base_env, policy, expert, args, mix_prob)
        wave_bc = {
            "bc_loss": tensor_mean(bc_infos, "bc_loss"),
            "bc_action_mse": tensor_mean(bc_infos, "bc_action_mse"),
            "bc_aux_waypoint_loss": tensor_mean(bc_infos, "bc_aux_waypoint_loss"),
            "bc_aux_target_pos_loss": tensor_mean(bc_infos, "bc_aux_target_pos_loss"),
            "bc_aux_forward_dir_loss": tensor_mean(bc_infos, "bc_aux_forward_dir_loss"),
            "bc_aux_assignment_loss": tensor_mean(bc_infos, "bc_aux_assignment_loss"),
            "bc_aux_trap_loss": tensor_mean(bc_infos, "bc_aux_trap_loss"),
            "bc_aux_assignment_acc": tensor_mean(bc_infos, "bc_aux_assignment_acc"),
            "bc_aux_trap_acc": tensor_mean(bc_infos, "bc_aux_trap_acc"),
            "bc_pred_action_norm": tensor_mean(bc_infos, "bc_pred_action_norm"),
            "bc_target_action_norm": tensor_mean(bc_infos, "bc_target_action_norm"),
        }

        replay_infos = []
        if replay_chunk_files and int(args.replay_updates_per_wave) > 0:
            for _ in range(int(args.replay_updates_per_wave)):
                replay_batch = sample_replay_batch(
                    replay_chunk_files,
                    cfg.sim.device,
                    int(agent_spec.n),
                    int(args.replay_batch_size),
                    float(args.replay_front_weight_alpha),
                )
                if replay_batch is None:
                    continue
                replay_info = policy.update_actor_bc(
                    replay_batch,
                    entropy_bonus_coef=float(args.entropy_bonus_coef),
                    action_mse_coef=float(args.action_mse_coef),
                    log_prob_coef=float(args.log_prob_coef),
                    aux_vel_cmd_coef=float(args.aux_vel_cmd_coef),
                    aux_waypoint_coef=float(args.aux_waypoint_coef),
                    aux_target_pos_coef=float(args.aux_target_pos_coef),
                    aux_forward_dir_coef=float(args.aux_forward_dir_coef),
                    aux_assignment_coef=float(args.aux_assignment_coef),
                    aux_trap_coef=float(args.aux_trap_coef),
                )
                replay_infos.append(replay_info)
        replay_bc = {
            "bc_loss": tensor_mean(replay_infos, "bc_loss"),
            "bc_action_mse": tensor_mean(replay_infos, "bc_action_mse"),
            "bc_aux_waypoint_loss": tensor_mean(replay_infos, "bc_aux_waypoint_loss"),
            "bc_aux_target_pos_loss": tensor_mean(replay_infos, "bc_aux_target_pos_loss"),
            "bc_aux_forward_dir_loss": tensor_mean(replay_infos, "bc_aux_forward_dir_loss"),
            "bc_aux_assignment_loss": tensor_mean(replay_infos, "bc_aux_assignment_loss"),
            "bc_aux_trap_loss": tensor_mean(replay_infos, "bc_aux_trap_loss"),
            "bc_pred_action_norm": tensor_mean(replay_infos, "bc_pred_action_norm"),
            "bc_target_action_norm": tensor_mean(replay_infos, "bc_target_action_norm"),
        }

        for key, value in wave_summary.items():
            writer.add_scalar(f"dagger/wave/{key}", value, wave)
        for key, value in wave_bc.items():
            writer.add_scalar(f"dagger/train/{key}", value, wave)
        for key, value in replay_bc.items():
            writer.add_scalar(f"dagger/replay/{key}", value, wave)
        writer.add_scalar("dagger/wave/expert_mix_prob", mix_prob, wave)

        logging.info(
            "DAgger wave %d/%d | cap=%.1f%% goal=%.1f%% landed=%.1f%% timeout=%.1f%% "
            "cap_steps=%.1f | mix=%.3f | bc_loss=%.6f mse=%.6f",
            wave,
            int(args.waves),
            100.0 * wave_summary["capture_rate"],
            100.0 * wave_summary["goal_rate"],
            100.0 * wave_summary["landed_rate"],
            100.0 * wave_summary["timeout_rate"],
            wave_summary["capture_steps_mean"],
            mix_prob,
            wave_bc["bc_loss"],
            wave_bc["bc_action_mse"],
        )
        if replay_infos:
            logging.info(
                "DAgger replay %d/%d | loss=%.6f mse=%.6f pred_norm=%.6f target_norm=%.6f",
                wave,
                int(args.waves),
                replay_bc["bc_loss"],
                replay_bc["bc_action_mse"],
                replay_bc["bc_pred_action_norm"],
                replay_bc["bc_target_action_norm"],
            )

        ckpt_path = os.path.join(save_dir, f"dagger_wave_{wave:03d}.pt")
        torch.save(policy.state_dict(), ckpt_path)

        need_eval = int(args.eval_every) > 0 and (
            (wave % int(args.eval_every) == 0) or (wave == int(args.waves))
        )
        if need_eval:
            eval_summary = run_policy_eval(
                env,
                base_env,
                policy,
                max_steps=int(args.episode_length),
                n_eval=int(args.n_eval),
            )
            for key, value in eval_summary.items():
                writer.add_scalar(f"dagger/eval/{key}", value, wave)
            logging.info(
                "DAgger eval @wave %d | cap=%.1f%% goal=%.1f%% landed=%.1f%% timeout=%.1f%% cap_steps=%.1f",
                wave,
                100.0 * eval_summary["capture_rate"],
                100.0 * eval_summary["goal_rate"],
                100.0 * eval_summary["landed_rate"],
                100.0 * eval_summary["timeout_rate"],
                eval_summary["capture_steps_mean"],
            )
            if eval_summary["capture_rate"] > best_capture:
                best_capture = eval_summary["capture_rate"]
                best_path = os.path.join(save_dir, "dagger_best.pt")
                torch.save(policy.state_dict(), best_path)

    final_path = os.path.join(save_dir, "dagger_final.pt")
    torch.save(policy.state_dict(), final_path)
    logging.info("Saved final DAgger checkpoint: %s", final_path)
    if best_path is not None:
        logging.info("Best eval checkpoint: %s (capture=%.1f%%)", best_path, 100.0 * best_capture)

    writer.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
