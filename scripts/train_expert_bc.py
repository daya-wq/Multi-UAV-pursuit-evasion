import argparse
import csv
import datetime
import glob
import json
import logging
import os
import random
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter
from torchrl.envs.transforms import InitTracker, Compose, TransformedEnv

from omni_drones import CONFIG_PATH, init_simulation_app
from omni_drones.utils.torchrl import AgentSpec
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
    History,
)
from omni_drones.learning import MAPPOPolicy


_extra_parser = argparse.ArgumentParser(add_help=False)
_extra_parser.add_argument("--dataset_dir", required=True)
_extra_parser.add_argument("--epochs", type=int, default=10)
_extra_parser.add_argument("--batch_size", type=int, default=4096)
_extra_parser.add_argument("--save_every", type=int, default=1)
_extra_parser.add_argument("--save_dir", default="checkpoints")
_extra_parser.add_argument("--save_tag", default="expert_bc")
_extra_parser.add_argument("--bc_lr", type=float, default=-1.0)
_extra_parser.add_argument("--entropy_bonus_coef", type=float, default=0.0)
_extra_parser.add_argument("--action_mse_coef", type=float, default=1.0)
_extra_parser.add_argument("--log_prob_coef", type=float, default=0.0)
_extra_parser.add_argument("--aux_vel_cmd_coef", type=float, default=0.0)
_extra_parser.add_argument("--aux_waypoint_coef", type=float, default=0.0)
_extra_parser.add_argument("--aux_assignment_coef", type=float, default=0.0)
_extra_parser.add_argument("--aux_trap_coef", type=float, default=0.0)
_extra_parser.add_argument("--equalize_episode_weight", type=lambda x: x.lower() != "false", default=False)
_extra_parser.add_argument("--front_weight_alpha", type=float, default=0.0)
_extra_parser.add_argument("--close_weight_mult", type=float, default=1.0)
_extra_parser.add_argument("--near_dist_reweight", type=lambda x: x.lower() != "false", default=False)
_extra_parser.add_argument("--near_dist_bins", type=int, default=5)
_extra_parser.add_argument("--action_dim_weights", type=str, default="1,1,1,1")
_extra_parser.add_argument("--min_episode_len", type=int, default=-1)
_extra_parser.add_argument("--max_episode_len", type=int, default=-1)
_extra_parser.add_argument("--keep_prefix_steps", type=int, default=-1)
_extra_parser.add_argument("--keep_suffix_steps", type=int, default=0)
_extra_parser.add_argument("--shuffle_chunks", type=lambda x: x.lower() != "false", default=True)
_extra_parser.add_argument("--max_chunks", type=int, default=-1)
_extra_parser.add_argument("--device", default="")
_extra_parser.add_argument("--eval_every", type=int, default=1)
_extra_parser.add_argument("--n_eval", type=int, default=1024)
_extra_parser.add_argument("--eval_batch_envs", type=int, default=1024)
_extra_parser.add_argument("--episode_length", type=int, default=1200)
_extra_parser.add_argument("--v_prey_test", type=float, default=1.5)
_extra_parser.add_argument("--v_drone_test", type=float, default=1.5)
_extra_parser.add_argument("--eval_success_threshold", type=float, default=0.60)
_extra_args, _remaining_argv = _extra_parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining_argv


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_tuple_key(key_str: str):
    parts = tuple(p for p in key_str.split("/") if p)
    return parts[0] if len(parts) == 1 else parts


def build_obs_tensordict(obs_storage: dict, indices: torch.Tensor, device: str, n_agents: int):
    obs_data = {}
    for key_str, value in obs_storage.items():
        obs_data[maybe_tuple_key(key_str)] = value[indices].to(device=device, dtype=torch.float32)
    return TensorDict(obs_data, batch_size=[indices.numel(), n_agents], device=device)


def get_chunk_action_targets(chunk: dict, *, actor_has_tanh: bool) -> torch.Tensor:
    action = chunk["action_raw"]
    label_type = str(chunk.get("action_label_type", "legacy_raw"))
    if label_type == "pidrate_normalized":
        return action
    if actor_has_tanh:
        return torch.tanh(action)
    return action


def build_step_weights(
    episode_lengths: torch.Tensor,
    *,
    equalize_episode_weight: bool = False,
    front_weight_alpha: float = 0.0,
) -> torch.Tensor:
    if episode_lengths.numel() == 0:
        return torch.empty(0, dtype=torch.float32)

    weights = []
    for length in episode_lengths.tolist():
        L = int(length)
        w = torch.ones(L, dtype=torch.float32)
        if equalize_episode_weight:
            w /= max(L, 1)
        if front_weight_alpha > 0.0:
            if L == 1:
                front = torch.ones(1, dtype=torch.float32) * (1.0 + front_weight_alpha)
            else:
                progress = torch.linspace(0.0, 1.0, steps=L, dtype=torch.float32)
                front = 1.0 + front_weight_alpha * (1.0 - progress)
            w *= front
        weights.append(w)

    flat = torch.cat(weights, dim=0)
    return flat / flat.mean().clamp_min(1e-8)


def compute_obs_aware_weights(
    obs_td,
    base_weight: torch.Tensor,
    *,
    close_weight_mult: float = 1.0,
    near_dist_reweight: bool = False,
) -> torch.Tensor:
    """Compute per-sample adaptive weights from observation data.

    Args:
        obs_td: TensorDict with 'state_self' key, shape [batch, agents, 1, 23].
        base_weight: [batch] base per-step weights from episode structure.
        close_weight_mult: Multiplier for close_trigger=1 samples.
        near_dist_reweight: If True, apply inverse-distance weighting based
            on waypoint distance.

    Returns:
        [batch] adjusted weights on the same device as base_weight.
    """
    state_self = obs_td["state_self"]  # [batch, agents, 1, 23] or [batch, agents, 23]
    if state_self.dim() == 4:
        state_self = state_self.squeeze(-2)  # [batch, agents, 23]

    device = base_weight.device
    weight = base_weight.clone()

    # --- Close trigger boost ---
    if close_weight_mult > 1.0 + 1e-6:
        close_trigger = state_self[..., 19].to(device)  # [batch, agents]
        any_close = (close_trigger > 0.5).any(dim=-1).float()  # [batch]
        close_boost = 1.0 + (close_weight_mult - 1.0) * any_close
        weight = weight * close_boost

    # --- Distance-based inverse frequency weighting ---
    if near_dist_reweight:
        drone_pos = state_self[..., :3].to(device)  # [batch, agents, 3]
        assigned_wp = state_self[..., 13:16].to(device)  # [batch, agents, 3]
        wp_dist = (assigned_wp - drone_pos).norm(dim=-1).mean(dim=-1)  # [batch]
        # Inverse distance: closer frames get higher weight
        inv_dist = 1.0 / (1.0 + wp_dist)
        # Normalize to mean=1 so it only redistributes weight, doesn't change total
        inv_dist = inv_dist / inv_dist.mean().clamp_min(1e-8)
        weight = weight * inv_dist

    # Re-normalize to mean=1
    weight = weight / weight.mean().clamp_min(1e-8)
    return weight


def filter_chunk_steps(
    chunk: dict,
    *,
    min_episode_len: int = -1,
    max_episode_len: int = -1,
    keep_prefix_steps: int = -1,
    keep_suffix_steps: int = 0,
) -> dict:
    episode_lengths = chunk["episode_lengths"].to(dtype=torch.long)
    if episode_lengths.numel() == 0:
        return chunk

    keep_episode_lengths = []
    keep_step_indices = []
    cursor = 0

    for length_tensor in episode_lengths:
        length = int(length_tensor.item())
        start = cursor
        end = cursor + length
        cursor = end

        if min_episode_len > 0 and length < min_episode_len:
            continue
        if max_episode_len > 0 and length > max_episode_len:
            continue

        prefix = length if keep_prefix_steps <= 0 else min(length, keep_prefix_steps)
        suffix = 0 if keep_suffix_steps <= 0 else min(length, keep_suffix_steps)

        if prefix + suffix >= length:
            indices = torch.arange(start, end, dtype=torch.long)
        else:
            head = torch.arange(start, start + prefix, dtype=torch.long)
            tail = torch.arange(end - suffix, end, dtype=torch.long) if suffix > 0 else torch.empty(0, dtype=torch.long)
            indices = torch.cat([head, tail], dim=0)

        if indices.numel() == 0:
            continue

        keep_episode_lengths.append(indices.numel())
        keep_step_indices.append(indices)

    if not keep_step_indices:
        empty_obs = {
            key: value[:0].clone()
            for key, value in chunk["obs"].items()
        }
        filtered = {
            "obs": empty_obs,
            "prev_action": chunk.get("prev_action", chunk["action_raw"])[:0].clone(),
            "action_raw": chunk["action_raw"][:0].clone(),
            "episode_lengths": torch.empty(0, dtype=torch.long),
            "num_success_episodes": 0,
            "num_success_steps": 0,
        }
        return filtered

    keep_indices = torch.cat(keep_step_indices, dim=0)
    filtered_obs = {
        key: value[keep_indices].clone()
        for key, value in chunk["obs"].items()
    }
    filtered = dict(chunk)
    filtered["obs"] = filtered_obs
    if "prev_action" in chunk:
        filtered["prev_action"] = chunk["prev_action"][keep_indices].clone()
    filtered["action_raw"] = chunk["action_raw"][keep_indices].clone()
    filtered["episode_lengths"] = torch.as_tensor(keep_episode_lengths, dtype=torch.long)
    filtered["num_success_episodes"] = int(len(keep_episode_lengths))
    filtered["num_success_steps"] = int(keep_indices.numel())
    return filtered


def find_latest_tp_weight(project_root: str):
    candidates = sorted(glob.glob(os.path.join(project_root, "checkpoints", "**", "tp_only_*.pt"), recursive=True))
    return candidates[-1] if candidates else None


def checkpoint_has_bc_aux(checkpoint_path: str) -> bool:
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        return False
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if bool(state.get("actor_has_bc_aux", False)):
        return True
    actor_params = state.get("actor_params", None)
    if actor_params is None:
        return False
    keys = actor_params.to_tensordict().keys(True, True)
    return any("aux_heads" in "/".join(map(str, key)) for key in keys)


def carry_actor_rnn_state(policy, src_td: TensorDict, dst_td: TensorDict):
    rnn_key = f"{policy.agent_spec.name}.actor_rnn_state"
    if rnn_key in src_td.keys():
        dst_td[rnn_key] = src_td[rnn_key].detach()


def summarize_results(results):
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


def run_policy_eval(env, base_env, policy, max_steps: int, n_eval: int):
    was_training = policy.actor.training
    env.eval()
    base_env.eval()
    policy.actor.eval()
    policy.critic.eval()
    if hasattr(policy, "TP_net") and policy.TP_net is not None:
        policy.TP_net.eval()

    results = []
    eval_done = 0
    wave = 0
    while eval_done < n_eval:
        wave += 1
        env.set_seed(19000 + wave * 97)
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

            with torch.no_grad():
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

    if was_training:
        policy.actor.train()
    env.train()
    base_env.train()
    return summarize_results(results)


def save_eval_history(save_dir: str, eval_history: list):
    json_path = os.path.join(save_dir, "bc_eval_history.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eval_history, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(save_dir, "bc_eval_history.csv")
    fieldnames = [
        "epoch",
        "episodes",
        "capture_rate",
        "goal_rate",
        "landed_rate",
        "timeout_rate",
        "capture_steps_mean",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in eval_history:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


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
    eval_num_envs = max(1, min(int(args.eval_batch_envs), int(args.n_eval)))
    cfg.task.num_envs = eval_num_envs
    cfg.task.env.num_envs = eval_num_envs
    if hasattr(cfg, "env"):
        cfg.env.num_envs = eval_num_envs
        cfg.env.max_episode_length = int(args.episode_length)
    cfg.task.max_episode_length = int(args.episode_length)
    cfg.task.env.max_episode_length = int(args.episode_length)
    cfg.task.v_drone = float(args.v_drone_test)
    use_bc_aux = any(
        float(v) != 0.0
        for v in (
            args.aux_vel_cmd_coef,
            args.aux_waypoint_coef,
            args.aux_assignment_coef,
            args.aux_trap_coef,
        )
    )
    if cfg.get("model_dir", None):
        use_bc_aux = use_bc_aux or checkpoint_has_bc_aux(cfg.model_dir)
    cfg.algo.actor.bc_aux.enabled = bool(use_bc_aux)

    dataset_dir = args.dataset_dir
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.join(os.getcwd(), dataset_dir)
    chunk_files = sorted(glob.glob(os.path.join(dataset_dir, "expert_success_wave_*.pt")))
    if not chunk_files:
        raise FileNotFoundError(f"No expert dataset chunks found in {dataset_dir}")
    if args.max_chunks > 0:
        chunk_files = chunk_files[:args.max_chunks]
    if args.shuffle_chunks:
        rng = random.Random(cfg.seed)
        rng.shuffle(chunk_files)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_root = args.save_dir
    if not os.path.isabs(save_root):
        save_root = os.path.join(project_root, save_root)
    save_dir = os.path.join(save_root, f"{args.save_tag}_{time_str}")
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(project_root, "runs", f"{args.save_tag}_{time_str}"))

    simulation_app = init_simulation_app(cfg)

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
            actor_has_tanh = bool(cfg.algo.actor.get("tanh", False))
            transforms.append(PIDRateController(controller, actor_has_tanh=actor_has_tanh))
        elif not action_transform.lower() == "none":
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    agent_spec: AgentSpec = env.agent_spec["drone"]
    policy = MAPPOPolicy(cfg.algo, agent_spec=agent_spec, device=cfg.sim.device, TP_net=base_env.TP)

    if cfg.model_dir is not None:
        policy.load_state_dict(torch.load(cfg.model_dir))
        logging.info("Loaded model checkpoint: %s", cfg.model_dir)
    else:
        if bool(getattr(cfg.algo, "use_TP_net", False)):
            tp_model_dir = cfg.get("tp_model_dir", None)
            if tp_model_dir is None:
                tp_model_dir = find_latest_tp_weight(project_root)
            if tp_model_dir:
                tp_state = torch.load(tp_model_dir)
                if isinstance(tp_state, dict) and "TP" in tp_state:
                    tp_state = tp_state["TP"]
                policy.TP_net.load_state_dict(tp_state)
                logging.info("Loaded TP checkpoint: %s", tp_model_dir)
        else:
            logging.info("TP_net disabled for BC imitation run; skip loading TP checkpoint.")

    if args.bc_lr > 0:
        for group in policy.actor_opt.param_groups:
            group["lr"] = float(args.bc_lr)

    logging.info("BC dataset dir: %s", dataset_dir)
    logging.info("BC chunks: %d", len(chunk_files))
    logging.info("BC save dir: %s", save_dir)
    logging.info(
        "BC eval: every=%d n_eval=%d batch_envs=%d v_drone=%.2f v_prey=%.2f gate=%.1f%%",
        int(args.eval_every),
        int(args.n_eval),
        eval_num_envs,
        float(args.v_drone_test),
        float(args.v_prey_test),
        100.0 * float(args.eval_success_threshold),
    )
    logging.info(
        "BC objective: action_mse_coef=%.4f log_prob_coef=%.4f entropy_bonus_coef=%.4f "
        "aux_vel=%.4f aux_waypoint=%.4f aux_assignment=%.4f aux_trap=%.4f",
        float(args.action_mse_coef),
        float(args.log_prob_coef),
        float(args.entropy_bonus_coef),
        float(args.aux_vel_cmd_coef),
        float(args.aux_waypoint_coef),
        float(args.aux_assignment_coef),
        float(args.aux_trap_coef),
    )
    logging.info(
        "BC weights: equalize_episode_weight=%s front_weight_alpha=%.4f "
        "close_weight_mult=%.2f near_dist_reweight=%s action_dim_weights=%s",
        bool(args.equalize_episode_weight),
        float(args.front_weight_alpha),
        float(args.close_weight_mult),
        bool(args.near_dist_reweight),
        args.action_dim_weights,
    )
    logging.info(
        "BC episode filter: min_episode_len=%d max_episode_len=%d keep_prefix_steps=%d keep_suffix_steps=%d",
        int(args.min_episode_len),
        int(args.max_episode_len),
        int(args.keep_prefix_steps),
        int(args.keep_suffix_steps),
    )

    total_loaded_episodes = 0
    total_loaded_steps = 0
    for chunk_path in chunk_files:
        chunk = torch.load(chunk_path, map_location="cpu")
        chunk = filter_chunk_steps(
            chunk,
            min_episode_len=int(args.min_episode_len),
            max_episode_len=int(args.max_episode_len),
            keep_prefix_steps=int(args.keep_prefix_steps),
            keep_suffix_steps=int(args.keep_suffix_steps),
        )
        total_loaded_episodes += int(chunk.get("num_success_episodes", len(chunk.get("episode_lengths", []))))
        total_loaded_steps += int(chunk.get("num_success_steps", chunk["action_raw"].shape[0]))
    logging.info("BC dataset summary: success_episodes=%d success_steps=%d", total_loaded_episodes, total_loaded_steps)

    global_step = 0
    n_agents = int(agent_spec.n)
    best_capture = -1.0
    best_epoch = -1
    best_path = None
    threshold_reached_epoch = None
    eval_history = []

    base_env.current_pursuer_speed = float(args.v_drone_test)
    base_env.current_target_speed = float(args.v_prey_test)
    base_env.velocity_scale = max(float(args.v_drone_test), 1e-6)
    base_env.target_velocity_scale = max(float(args.v_prey_test), 1e-6)

    for epoch in range(1, int(args.epochs) + 1):
        epoch_infos = []
        epoch_steps = 0

        chunk_order = list(chunk_files)
        if args.shuffle_chunks:
            rng = random.Random(cfg.seed + epoch)
            rng.shuffle(chunk_order)

        for chunk_path in chunk_order:
            chunk = torch.load(chunk_path, map_location="cpu")
            chunk = filter_chunk_steps(
                chunk,
                min_episode_len=int(args.min_episode_len),
                max_episode_len=int(args.max_episode_len),
                keep_prefix_steps=int(args.keep_prefix_steps),
                keep_suffix_steps=int(args.keep_suffix_steps),
            )
            action_targets = get_chunk_action_targets(
                chunk,
                actor_has_tanh=bool(cfg.algo.actor.get("tanh", False)),
            )
            if action_targets.numel() == 0:
                continue
            obs_storage = chunk["obs"]
            aux_storage = chunk.get("expert_aux", {})
            num_steps = int(action_targets.shape[0])
            step_weights = build_step_weights(
                chunk["episode_lengths"],
                equalize_episode_weight=bool(args.equalize_episode_weight),
                front_weight_alpha=float(args.front_weight_alpha),
            )
            perm = torch.randperm(num_steps)

            for start in range(0, num_steps, int(args.batch_size)):
                indices = perm[start:start + int(args.batch_size)]
                if indices.numel() == 0:
                    continue
                obs_td = build_obs_tensordict(obs_storage, indices, cfg.sim.device, n_agents)
                action_batch = action_targets[indices].to(device=cfg.sim.device, dtype=torch.float32)
                if "prev_action" in chunk:
                    prev_action_batch = chunk["prev_action"][indices].to(device=cfg.sim.device, dtype=torch.float32)
                else:
                    prev_action_batch = torch.zeros_like(action_batch)
                weight_batch = step_weights[indices].to(device=cfg.sim.device, dtype=torch.float32)
                # --- Obs-aware adaptive weighting ---
                weight_batch = compute_obs_aware_weights(
                    obs_td,
                    weight_batch,
                    close_weight_mult=float(args.close_weight_mult),
                    near_dist_reweight=bool(args.near_dist_reweight),
                )
                # --- Parse action dimension weights ---
                action_dim_weights_list = [
                    float(x.strip()) for x in args.action_dim_weights.split(",")
                ]
                action_dim_weights_tensor = torch.tensor(
                    action_dim_weights_list, device=cfg.sim.device, dtype=torch.float32
                )
                batch_td = TensorDict(
                    {
                        ("agents", "observation"): obs_td,
                        ("agents", "prev_action"): prev_action_batch,
                        ("agents", "action"): action_batch,
                        "bc_weight": weight_batch,
                    },
                    batch_size=[indices.numel()],
                    device=cfg.sim.device,
                )
                for key, value in aux_storage.items():
                    aux_value = value[indices]
                    if aux_value.dtype.is_floating_point:
                        aux_value = aux_value.to(device=cfg.sim.device, dtype=torch.float32)
                    else:
                        aux_value = aux_value.to(device=cfg.sim.device)
                    batch_td["expert_aux", key] = aux_value
                info = policy.update_actor_bc(
                    batch_td,
                    entropy_bonus_coef=float(args.entropy_bonus_coef),
                    action_mse_coef=float(args.action_mse_coef),
                    log_prob_coef=float(args.log_prob_coef),
                    aux_vel_cmd_coef=float(args.aux_vel_cmd_coef),
                    aux_waypoint_coef=float(args.aux_waypoint_coef),
                    aux_assignment_coef=float(args.aux_assignment_coef),
                    aux_trap_coef=float(args.aux_trap_coef),
                    action_dim_weights=action_dim_weights_tensor,
                )
                epoch_infos.append(info)
                epoch_steps += int(indices.numel())
                global_step += int(indices.numel())

            del chunk

        if not epoch_infos:
            raise RuntimeError("No BC batches were processed; dataset may be empty.")

        summary = {
            key: float(np.mean([item[key] for item in epoch_infos]))
            for key in epoch_infos[0].keys()
        }
        summary["epoch_steps"] = float(epoch_steps)

        if hasattr(policy, "actor_opt_scheduler"):
            policy.actor_opt_scheduler.step()

        for key, value in summary.items():
            writer.add_scalar(f"bc/{key}", value, epoch)

        logging.info(
            "BC epoch %d/%d | steps=%d | loss=%.6f | mse=%.6f | logp=%.6f | "
            "pred_norm=%.6f | target_norm=%.6f | log_std=%.6f",
            epoch,
            int(args.epochs),
            epoch_steps,
            summary["bc_loss"],
            summary["bc_action_mse"],
            summary["bc_log_prob"],
            summary["bc_pred_action_norm"],
            summary["bc_target_action_norm"],
            summary["bc_log_std_mean"],
        )

        if epoch % int(args.save_every) == 0:
            ckpt_path = os.path.join(save_dir, f"bc_epoch_{epoch:03d}.pt")
            torch.save(policy.state_dict(), ckpt_path)
            logging.info("Saved BC checkpoint: %s", ckpt_path)

        need_eval = int(args.eval_every) > 0 and (
            (epoch % int(args.eval_every) == 0) or (epoch == int(args.epochs))
        )
        if need_eval:
            eval_summary = run_policy_eval(
                env,
                base_env,
                policy,
                max_steps=int(args.episode_length),
                n_eval=int(args.n_eval),
            )
            eval_summary["epoch"] = int(epoch)
            eval_history.append(eval_summary)
            save_eval_history(save_dir, eval_history)
            for key, value in eval_summary.items():
                if key == "epoch":
                    continue
                writer.add_scalar(f"bc_eval/{key}", value, epoch)
            logging.info(
                "BC eval @epoch %d | cap=%.1f%% goal=%.1f%% landed=%.1f%% timeout=%.1f%% cap_steps=%.1f",
                epoch,
                100.0 * eval_summary["capture_rate"],
                100.0 * eval_summary["goal_rate"],
                100.0 * eval_summary["landed_rate"],
                100.0 * eval_summary["timeout_rate"],
                eval_summary["capture_steps_mean"],
            )
            if eval_summary["capture_rate"] > best_capture:
                best_capture = eval_summary["capture_rate"]
                best_epoch = int(epoch)
                best_path = os.path.join(save_dir, "bc_best.pt")
                torch.save(policy.state_dict(), best_path)
                with open(os.path.join(save_dir, "bc_best_metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(eval_summary, f, ensure_ascii=False, indent=2)
            if (
                threshold_reached_epoch is None
                and eval_summary["capture_rate"] >= float(args.eval_success_threshold)
            ):
                threshold_reached_epoch = int(epoch)

    final_path = os.path.join(save_dir, "bc_final.pt")
    torch.save(policy.state_dict(), final_path)
    logging.info("Saved final BC checkpoint: %s", final_path)
    summary_path = os.path.join(save_dir, "bc_summary.json")
    summary_payload = {
        "dataset_dir": dataset_dir,
        "save_dir": save_dir,
        "final_checkpoint": final_path,
        "best_checkpoint": best_path,
        "best_capture_rate": float(best_capture),
        "best_epoch": int(best_epoch),
        "eval_success_threshold": float(args.eval_success_threshold),
        "threshold_reached": bool(
            threshold_reached_epoch is not None and best_capture >= float(args.eval_success_threshold)
        ),
        "threshold_reached_epoch": threshold_reached_epoch,
        "n_eval": int(args.n_eval),
        "eval_batch_envs": int(eval_num_envs),
        "v_drone_test": float(args.v_drone_test),
        "v_prey_test": float(args.v_prey_test),
        "epochs": int(args.epochs),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)
    logging.info(
        "BC summary saved: %s | best_capture=%.1f%% best_epoch=%d threshold_reached=%s",
        summary_path,
        100.0 * max(best_capture, 0.0),
        best_epoch,
        "yes" if summary_payload["threshold_reached"] else "no",
    )

    writer.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
