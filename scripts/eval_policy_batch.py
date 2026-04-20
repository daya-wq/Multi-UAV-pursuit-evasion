import argparse
import math
import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
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
_extra_parser.add_argument("--model_dir", required=True)
_extra_parser.add_argument("--tp_model_dir", default="")
_extra_parser.add_argument("--n_eval", type=int, default=2048)
_extra_parser.add_argument("--batch_envs", type=int, default=2048)
_extra_parser.add_argument("--episode_length", type=int, default=1200)
_extra_parser.add_argument("--v_prey_test", type=float, default=1.5)
_extra_parser.add_argument("--v_drone_test", type=float, default=1.5)
_extra_parser.add_argument("--random_init", type=lambda x: x.lower() != "false", default=True)
_extra_parser.add_argument("--seed_base", type=int, default=999)
_extra_parser.add_argument("--deterministic", type=lambda x: x.lower() != "false", default=True)
_extra_args, _remaining_argv = _extra_parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining_argv


def _stats_scalar(stats_td, key: str, env_idx: int):
    if key not in stats_td.keys():
        return float("nan")
    value = stats_td[key][env_idx]
    if value.numel() == 0:
        return float("nan")
    return float(value.reshape(-1)[0].item())


def _mean(items, key):
    vals = [r[key] for r in items if not math.isnan(r.get(key, float("nan")))]
    return float(np.mean(vals)) if vals else float("nan")


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


def enable_actor_rnn_cfg(cfg, hidden_size: int, train_seq_len: int = 8):
    cfg.algo.actor.rnn = OmegaConf.create(
        {
            "cls": "gru",
            "kwargs": {"hidden_size": int(hidden_size)},
            "train_seq_len": int(train_seq_len),
        }
    )


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


@torch.no_grad()
def run_policy_wave(env, base_env, policy, max_steps: int, deterministic: bool = True):
    tensordict = env.reset()
    device = base_env.device
    n_envs = int(base_env.num_envs)
    n_agents = int(base_env.drone.n)
    tensordict[("agents", "prev_action")] = torch.zeros(
        n_envs, n_agents, 4, device=device, dtype=torch.float32
    )

    done_mask = torch.zeros(n_envs, dtype=torch.bool, device=device)
    results = [None for _ in range(n_envs)]
    zero_action = torch.zeros(n_envs, n_agents, 4, device=device)
    min_target_dist_seen = [float("inf")] * n_envs
    min_goal_dist_seen = [float("inf")] * n_envs

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

        action_td = policy(tensordict, deterministic=deterministic)
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
                "target_predicted_error": _stats_scalar(stats, "target_predicted_error", env_idx),
            }
            done_mask[env_idx] = True

        if bool(done_mask.all()):
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
        }

    return results


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg):
    args = _extra_args

    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    cfg.headless = True
    eval_num_envs = max(1, min(int(args.batch_envs), int(args.n_eval)))
    cfg.task.num_envs = eval_num_envs
    cfg.task.env.num_envs = eval_num_envs
    if hasattr(cfg, "env"):
        cfg.env.num_envs = eval_num_envs
        cfg.env.max_episode_length = int(args.episode_length)
    cfg.task.max_episode_length = int(args.episode_length)
    cfg.task.env.max_episode_length = int(args.episode_length)
    cfg.task.v_drone = float(args.v_drone_test)
    cfg.task.use_eval = 0 if args.random_init else 1

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

    env = TransformedEnv(base_env, Compose(*transforms)).eval()
    env.set_seed(0)
    base_env.enable_render(False)
    base_env.eval()

    if hasattr(base_env, "_set_curriculum_stage"):
        base_env._set_curriculum_stage(
            len(base_env.curriculum_stages) - 1,
            reset_metrics=False,
            announce=False,
        )
    base_env.current_target_speed = float(args.v_prey_test)
    base_env.target_velocity_scale = float(args.v_prey_test)

    state_dict = torch.load(args.model_dir, weights_only=False)
    if checkpoint_has_bc_aux(state_dict):
        cfg.algo.actor.bc_aux.enabled = True
    rnn_hidden = checkpoint_actor_rnn_hidden_size(state_dict)
    if rnn_hidden is not None:
        enable_actor_rnn_cfg(cfg, hidden_size=int(rnn_hidden))
    prev_action_hidden = checkpoint_prev_action_condition_hidden_size(state_dict)
    if prev_action_hidden is not None:
        enable_prev_action_conditioning_cfg(cfg, hidden_dim=int(prev_action_hidden))

    agent_spec: AgentSpec = env.agent_spec["drone"]
    policy = MAPPOPolicy(cfg.algo, agent_spec=agent_spec, device=cfg.sim.device, TP_net=base_env.TP)
    policy.load_state_dict(state_dict)
    if args.tp_model_dir:
        tp_state = torch.load(args.tp_model_dir, map_location=cfg.sim.device, weights_only=False)
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

    print(f"[Policy eval] model={args.model_dir}")
    print(f"[Policy eval] tp_model={args.tp_model_dir or '<from model checkpoint>'}")
    if hasattr(base_env, "curriculum_stages"):
        stage_cfg = base_env.curriculum_stages[base_env.curriculum_stage]
        dist_min, dist_max = stage_cfg["distance_range"]
        print(
            f"[Policy eval] curriculum_stage={base_env.curriculum_stage + 1} "
            f"distance_range=({dist_min:.2f}, {dist_max:.2f}) "
            f"pursuer_speed={base_env.current_pursuer_speed:.2f} "
            f"target_speed={base_env.current_target_speed:.2f}"
        )
    print(
        f"[Policy eval] n_eval={args.n_eval} batch_envs={eval_num_envs} "
        f"episode_length={args.episode_length} v_drone={args.v_drone_test} "
        f"v_prey={args.v_prey_test} random_init={args.random_init} "
        f"deterministic={args.deterministic}"
    )

    results = []
    eval_done = 0
    wave = 0
    while eval_done < int(args.n_eval):
        wave += 1
        env.set_seed(int(args.seed_base) + wave * 97)
        wave_results = run_policy_wave(
            env=env,
            base_env=base_env,
            policy=policy,
            max_steps=int(args.episode_length),
            deterministic=bool(args.deterministic),
        )
        for ep_info in wave_results:
            if eval_done >= int(args.n_eval):
                break
            results.append(ep_info)
            eval_done += 1

    n_total = len(results)
    n_cap = sum(1 for r in results if r["success"])
    n_goal = sum(1 for r in results if r["goal"])
    n_landed = sum(1 for r in results if r["landed"])
    n_timeout = n_total - n_cap - n_goal - n_landed
    cap_steps = [r["steps"] for r in results if r["success"]]

    print("\n" + "=" * 60, flush=True)
    print("  Policy Batch Evaluation Summary", flush=True)
    print("=" * 60, flush=True)
    print(f"  Episodes      : {n_total}", flush=True)
    print(f"  ✅ Capture    : {n_cap / n_total:.0%}  ({n_cap})", flush=True)
    print(f"  ❌ Goal zone  : {n_goal / n_total:.0%}  ({n_goal})", flush=True)
    print(f"  ⚠️  Landed    : {n_landed / n_total:.0%}  ({n_landed})", flush=True)
    print(f"  ⏱  Timeout    : {n_timeout / n_total:.0%}  ({n_timeout})", flush=True)
    if cap_steps:
        print(
            f"  Capture steps : mean={np.mean(cap_steps):.1f}, "
            f"std={np.std(cap_steps):.1f}, min={np.min(cap_steps)}, max={np.max(cap_steps)}"
        , flush=True)
    print(
        f"  TP pred err   : all={_mean(results, 'target_predicted_error'):.4f}, "
        f"capture={_mean([r for r in results if r['success']], 'target_predicted_error'):.4f}, "
        f"timeout={_mean([r for r in results if r['timeout']], 'target_predicted_error'):.4f}, "
        f"goal={_mean([r for r in results if r['goal']], 'target_predicted_error'):.4f}"
    , flush=True)
    print(
        f"  Timeout avg   : mindist={_mean([r for r in results if r['timeout']], 'min_target_dist_seen'):.3f}, "
        f"dmin={_mean([r for r in results if r['timeout']], 'd_i_min'):.3f}, "
        f"ahead={_mean([r for r in results if r['timeout']], 'n_agents_ahead_mean'):.2f}, "
        f"team={_mean([r for r in results if r['timeout']], 'phi_team'):.3f}, "
        f"spread={_mean([r for r in results if r['timeout']], 'phi_spread'):.3f}, "
        f"cap_prog={_mean([r for r in results if r['timeout']], 'capture_progress_reward'):.4f}"
    , flush=True)
    print(
        f"  Capture avg   : mindist={_mean([r for r in results if r['success']], 'min_target_dist_seen'):.3f}, "
        f"dmin={_mean([r for r in results if r['success']], 'd_i_min'):.3f}, "
        f"ahead={_mean([r for r in results if r['success']], 'n_agents_ahead_mean'):.2f}, "
        f"team={_mean([r for r in results if r['success']], 'phi_team'):.3f}, "
        f"spread={_mean([r for r in results if r['success']], 'phi_spread'):.3f}, "
        f"cap_prog={_mean([r for r in results if r['success']], 'capture_progress_reward'):.4f}"
    , flush=True)
    print("=" * 60, flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
