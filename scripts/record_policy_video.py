import argparse
import json
import os
import re
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv

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
_extra_parser.add_argument("--model_dir", nargs="+", required=True)
_extra_parser.add_argument("--tp_model_dir", default="")
_extra_parser.add_argument("--out_dir", required=True)
_extra_parser.add_argument("--seed", type=int, default=20260412)
_extra_parser.add_argument("--seeds", default="")
_extra_parser.add_argument("--stage", type=int, default=4)
_extra_parser.add_argument("--episode_length", type=int, default=1200)
_extra_parser.add_argument("--frame_stride", type=int, default=2)
_extra_parser.add_argument("--num_envs", type=int, default=1)
_extra_parser.add_argument("--deterministic", type=lambda x: x.lower() != "false", default=True)
_extra_parser.add_argument("--use_curriculum_layout", type=lambda x: x.lower() != "false", default=True)
_extra_parser.add_argument("--record_video", type=lambda x: x.lower() != "false", default=True)
_extra_parser.add_argument("--device", default="")
_extra_parser.add_argument("--active_gpu", type=int, default=-1)
_extra_parser.add_argument("--physics_gpu", type=int, default=-1)
_extra_args, _remaining_argv = _extra_parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining_argv


def load_torch_checkpoint(path: str, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def checkpoint_has_bc_aux(state_dict) -> bool:
    if bool(state_dict.get("actor_has_bc_aux", False)):
        return True
    actor_params = state_dict.get("actor_params", None)
    if actor_params is None:
        return False
    keys = actor_params.to_tensordict().keys(True, True)
    return any("aux_heads" in "/".join(map(str, key)) for key in keys)


def checkpoint_actor_rnn_hidden_size(state_dict):
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


def checkpoint_prev_action_condition_hidden_size(state_dict):
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


def enable_actor_rnn_cfg(cfg, hidden_size: int, train_seq_len: int = 8):
    cfg.algo.actor.rnn = OmegaConf.create(
        {
            "cls": "gru",
            "kwargs": {"hidden_size": int(hidden_size)},
            "train_seq_len": int(train_seq_len),
        }
    )


def enable_prev_action_conditioning_cfg(cfg, hidden_dim: int):
    cfg.algo.actor.prev_action_conditioning = OmegaConf.create(
        {
            "enabled": True,
            "hidden_dim": int(hidden_dim),
        }
    )


def carry_actor_rnn_state(policy, src_td: TensorDict, dst_td: TensorDict):
    rnn_key = f"{policy.agent_spec.name}.actor_rnn_state"
    if rnn_key in src_td.keys():
        dst_td[rnn_key] = src_td[rnn_key].detach()


def safe_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def stats_scalar(stats_td, key: str, env_idx: int = 0):
    if stats_td is None or key not in stats_td.keys():
        return float("nan")
    value = stats_td[key][env_idx]
    if value.numel() == 0:
        return float("nan")
    return float(value.reshape(-1)[0].detach().cpu().item())


def make_env(cfg, base_env):
    transforms = [InitTracker()]
    if cfg.task.get("flatten_obs", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation")))
    if cfg.task.get("flatten_state", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "state")))
    if (
        cfg.task.get("flatten_intrinsics", True)
        and ("agents", "intrinsics") in base_env.observation_spec.keys(True)
    ):
        transforms.append(
            ravel_composite(base_env.observation_spec, ("agents", "intrinsics"), start_dim=-1)
        )
    if cfg.task.get("history", False):
        transforms.append(History([("agents", "observation")], steps=4))

    action_transform = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            transforms.append(FromMultiDiscreteAction(nbins=int(action_transform.split(":")[1])))
        elif action_transform.startswith("discrete"):
            transforms.append(FromDiscreteAction(nbins=int(action_transform.split(":")[1])))
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
        elif action_transform.lower() != "none":
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    return TransformedEnv(base_env, Compose(*transforms)).eval()


def build_policy(cfg, env, base_env, state_dict, tp_model_dir: str = ""):
    agent_spec: AgentSpec = env.agent_spec["drone"]
    policy = MAPPOPolicy(cfg.algo, agent_spec=agent_spec, device=cfg.sim.device, TP_net=base_env.TP)
    policy.load_state_dict(state_dict)
    if tp_model_dir:
        tp_state = load_torch_checkpoint(tp_model_dir, map_location=cfg.sim.device)
        if isinstance(tp_state, dict) and "TP" in tp_state:
            tp_state = tp_state["TP"]
        policy.TP_net.load_state_dict(tp_state)
    try:
        policy.actor.module.encoder.transformer.use_nested_tensor = False
    except Exception:
        pass
    if hasattr(policy, "actor"):
        policy.actor.eval()
    if hasattr(policy, "critic"):
        policy.critic.eval()
    if hasattr(policy, "TP_net"):
        policy.TP_net.eval()
    return policy


@torch.no_grad()
def record_one_policy(cfg, args, env, base_env, policy, model_path: str):
    if hasattr(base_env, "_set_curriculum_stage"):
        base_env._set_curriculum_stage(int(args.stage) - 1, reset_metrics=True, announce=True)
    if hasattr(base_env, "set_training_progress"):
        base_env.set_training_progress(1.0)

    base_env.enable_render(bool(args.record_video))
    base_env.eval()
    env.eval()
    env.set_seed(int(args.seed))
    if hasattr(policy, "reset_prev_action_cache"):
        policy.reset_prev_action_cache()

    td = env.reset()
    drone_pos, _ = base_env.get_env_poses(base_env.drone.get_world_poses())
    target_pos, _ = base_env.get_env_poses(base_env.target.get_world_poses())
    initial_state = {
        "drone_pos": drone_pos[0].detach().cpu().tolist(),
        "target_pos": target_pos[0].detach().cpu().tolist(),
        "curriculum_stage": int(getattr(base_env, "curriculum_stage", 0)) + 1,
        "pursuer_speed": float(getattr(base_env, "current_pursuer_speed", float("nan"))),
        "target_speed": float(getattr(base_env, "current_target_speed", float("nan"))),
    }

    n_envs = int(base_env.num_envs)
    n_agents = int(base_env.drone.n)
    td[("agents", "prev_action")] = torch.zeros(
        n_envs, n_agents, 4, device=base_env.device, dtype=torch.float32
    )

    frames = []
    latest_stats = None
    done_step = int(args.episode_length)
    done_reason = "TIMEOUT"
    zero_action = torch.zeros(n_envs, n_agents, 4, device=base_env.device)

    for step_idx in range(int(args.episode_length)):
        if bool(args.record_video) and step_idx % max(int(args.frame_stride), 1) == 0:
            frame = env.base_env.render(mode="rgb_array")
            if frame is not None:
                frames.append(frame)

        action_td = policy(td, deterministic=bool(args.deterministic))
        td = env.step(action_td)
        td_next = td.get("next")
        if ("agents", "action") in action_td.keys(True):
            prev_action = action_td[("agents", "action")]
        else:
            prev_action = zero_action
        td_next[("agents", "prev_action")] = prev_action.detach()
        carry_actor_rnn_state(policy, action_td, td_next)
        latest_stats = td_next.get("stats")
        done = bool(td_next.get("done").reshape(-1)[0].item())
        if done:
            done_step = step_idx
            if stats_scalar(latest_stats, "success") > 0.0:
                done_reason = "SUCCESS"
            elif stats_scalar(latest_stats, "goal_reached") > 0.0:
                done_reason = "GOAL_ZONE"
            elif stats_scalar(latest_stats, "any_landed") > 0.0:
                done_reason = "LANDED"
            elif step_idx >= int(args.episode_length) - 1:
                done_reason = "TIMEOUT"
            else:
                done_reason = "DONE_OTHER"
            break
        td = td_next

    if hasattr(policy, "reset_prev_action_cache"):
        policy.reset_prev_action_cache()

    os.makedirs(args.out_dir, exist_ok=True)
    video_name = f"{safe_stem(model_path)}_stage{args.stage}_seed{args.seed}_{done_reason}.mp4"
    video_path = os.path.join(args.out_dir, video_name)
    npy_path = video_path.replace(".mp4", ".npy")
    artifact_path = video_path
    if frames:
        frames_rgb = np.stack(frames)
        fps = int(0.5 / cfg.sim.dt)
        try:
            import imageio

            imageio.mimwrite(video_path, frames_rgb, fps=fps)
        except Exception:
            np.save(npy_path, frames_rgb)
            artifact_path = npy_path
    else:
        artifact_path = ""

    result = {
        "model_path": os.path.abspath(model_path),
        "artifact_path": os.path.abspath(artifact_path) if artifact_path else "",
        "done_step": int(done_step),
        "done_reason": done_reason,
        "initial_state": initial_state,
        "stats": {
            "success": stats_scalar(latest_stats, "success"),
            "goal_reached": stats_scalar(latest_stats, "goal_reached"),
            "any_landed": stats_scalar(latest_stats, "any_landed"),
            "first_capture_step": stats_scalar(latest_stats, "first_capture_step"),
            "collision": stats_scalar(latest_stats, "collision"),
            "pursuer_collisions_count": stats_scalar(latest_stats, "pursuer_collisions_count"),
            "curriculum_stage": stats_scalar(latest_stats, "curriculum_stage"),
            "curriculum_pursuer_speed": stats_scalar(latest_stats, "curriculum_pursuer_speed"),
            "curriculum_target_speed": stats_scalar(latest_stats, "curriculum_target_speed"),
        },
    }
    json_path = video_path.replace(".mp4", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        f"[record] {os.path.basename(model_path)} -> {artifact_path} | "
        f"reason={done_reason} step={done_step}"
    )
    return result


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg):
    args = _extra_args

    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    state_dicts = [load_torch_checkpoint(path, map_location="cpu") for path in args.model_dir]
    if any(checkpoint_has_bc_aux(state_dict) for state_dict in state_dicts):
        cfg.algo.actor.bc_aux.enabled = True
    rnn_hidden = next(
        (
            hidden_size
            for hidden_size in (
                checkpoint_actor_rnn_hidden_size(state_dict) for state_dict in state_dicts
            )
            if hidden_size is not None
        ),
        None,
    )
    if rnn_hidden is not None:
        enable_actor_rnn_cfg(cfg, hidden_size=int(rnn_hidden))
    prev_action_hidden = next(
        (
            hidden_dim
            for hidden_dim in (
                checkpoint_prev_action_condition_hidden_size(state_dict)
                for state_dict in state_dicts
            )
            if hidden_dim is not None
        ),
        None,
    )
    if prev_action_hidden is not None:
        enable_prev_action_conditioning_cfg(cfg, hidden_dim=int(prev_action_hidden))

    cfg.headless = True
    if args.device:
        cfg.sim.device = str(args.device)
    if int(args.active_gpu) >= 0:
        cfg.sim.active_gpu = int(args.active_gpu)
    if int(args.physics_gpu) >= 0:
        cfg.sim.physics_gpu = int(args.physics_gpu)
    cfg.task.use_eval = 1
    cfg.task.num_envs = int(args.num_envs)
    cfg.task.env.num_envs = int(args.num_envs)
    cfg.task.max_episode_length = int(args.episode_length)
    cfg.task.env.max_episode_length = int(args.episode_length)
    cfg.task.curriculum.enabled = True
    cfg.task.curriculum.eval_uses_fixed_layout = not bool(args.use_curriculum_layout)
    if hasattr(cfg, "env"):
        cfg.env.num_envs = int(args.num_envs)
        cfg.env.max_episode_length = int(args.episode_length)

    simulation_app = init_simulation_app(cfg)

    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)
    env = make_env(cfg, base_env)

    if str(args.seeds).strip():
        seed_values = [int(item.strip()) for item in str(args.seeds).split(",") if item.strip()]
    else:
        seed_values = [int(args.seed)]

    results = []
    for seed in seed_values:
        args.seed = int(seed)
        for model_path, state_dict in zip(args.model_dir, state_dicts):
            policy = build_policy(cfg, env, base_env, state_dict, args.tp_model_dir)
            results.append(record_one_policy(cfg, args, env, base_env, policy, model_path))
            del policy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    os.makedirs(args.out_dir, exist_ok=True)
    seed_label = f"seed{seed_values[0]}" if len(seed_values) == 1 else "seeds"
    summary_path = os.path.join(args.out_dir, f"summary_stage{args.stage}_{seed_label}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[record] summary -> {summary_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
