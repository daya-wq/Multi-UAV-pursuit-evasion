import argparse
import datetime
import glob
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
_extra_parser.add_argument("--min_episode_len", type=int, default=-1)
_extra_parser.add_argument("--max_episode_len", type=int, default=-1)
_extra_parser.add_argument("--keep_prefix_steps", type=int, default=-1)
_extra_parser.add_argument("--keep_suffix_steps", type=int, default=0)
_extra_parser.add_argument("--shuffle_chunks", type=lambda x: x.lower() != "false", default=True)
_extra_parser.add_argument("--max_chunks", type=int, default=-1)
_extra_parser.add_argument("--device", default="")
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
            transforms.append(PIDRateController(controller))
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
        tp_model_dir = cfg.get("tp_model_dir", None)
        if tp_model_dir is None:
            tp_model_dir = find_latest_tp_weight(project_root)
        if tp_model_dir:
            tp_state = torch.load(tp_model_dir)
            if isinstance(tp_state, dict) and "TP" in tp_state:
                tp_state = tp_state["TP"]
            policy.TP_net.load_state_dict(tp_state)
            logging.info("Loaded TP checkpoint: %s", tp_model_dir)

    if args.bc_lr > 0:
        for group in policy.actor_opt.param_groups:
            group["lr"] = float(args.bc_lr)

    logging.info("BC dataset dir: %s", dataset_dir)
    logging.info("BC chunks: %d", len(chunk_files))
    logging.info("BC save dir: %s", save_dir)
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
        "BC weights: equalize_episode_weight=%s front_weight_alpha=%.4f",
        bool(args.equalize_episode_weight),
        float(args.front_weight_alpha),
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
            action_raw = chunk["action_raw"]
            if action_raw.numel() == 0:
                continue
            obs_storage = chunk["obs"]
            aux_storage = chunk.get("expert_aux", {})
            num_steps = int(action_raw.shape[0])
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
                action_batch = action_raw[indices].to(device=cfg.sim.device, dtype=torch.float32)
                weight_batch = step_weights[indices].to(device=cfg.sim.device, dtype=torch.float32)
                batch_td = TensorDict(
                    {
                        ("agents", "observation"): obs_td,
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

    final_path = os.path.join(save_dir, "bc_final.pt")
    torch.save(policy.state_dict(), final_path)
    logging.info("Saved final BC checkpoint: %s", final_path)

    writer.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
