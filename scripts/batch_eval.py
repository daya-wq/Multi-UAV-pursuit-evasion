"""
Batch evaluation of an RL checkpoint with configurable target_repulsion_coef.
Runs N parallel environments for one episode each and logs aggregate stats.

Usage:
  python scripts/batch_eval.py \
    task=HideAndSeek headless=true wandb.mode=disabled \
    model_dir=<checkpoint_path> \
    algo.use_TP_net=1 algo.actor.bc_aux.enabled=true \
    algo.actor.bc_aux.hidden_dim=256 algo.actor.bc_aux.condition_action_on_aux=true \
    algo.actor.bc_aux.condition_hidden_dim=256 \
    algo.actor.log_std_init=-2.0 algo.actor.log_std_min=-2.0 algo.actor.log_std_max=-1.0 \
    algo.train_every=64 \
    task.env.num_envs=512 task.v_prey=1.5 task.v_drone=1.5 \
    seed=42 \
    +eval_label="repulsion_3x"
"""

import logging
import os
import sys
import json
import datetime

import hydra
import torch
import numpy as np

from omegaconf import OmegaConf, DictConfig
from omni_drones import CONFIG_PATH, init_simulation_app
from omni_drones.utils.torchrl import SyncDataCollector, AgentSpec
from omni_drones.utils.torchrl.transforms import (
    LogOnEpisode,
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
    History
)
from omni_drones.utils.wandb import init_wandb
from omni_drones.learning import (
    MAPPOPolicy,
    Policy,
    PPOPolicy,
    PPOAdaptivePolicy, PPORNNPolicy
)
from setproctitle import setproctitle
from torchrl.envs.transforms import (
    TransformedEnv,
    InitTracker,
    Compose,
)
from tqdm import tqdm


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    eval_label = str(cfg.get("eval_label", "default"))
    num_envs = int(cfg.task.env.num_envs)

    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle("batch_eval")

    from omni_drones.envs.isaac_env import IsaacEnv
    algos = {
        "ppo": PPOPolicy,
        "ppo_adaptive": PPOAdaptivePolicy,
        "ppo_rnn": PPORNNPolicy,
        "mappo": MAPPOPolicy,
        "test": Policy
    }

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    # Build transforms
    transforms = [InitTracker()]
    action_transform = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform == "PIDrate":
            from omni_drones.controllers import PIDRateController as _PIDRateController
            from omni_drones.utils.torchrl.transforms import PIDRateController
            controller = _PIDRateController(cfg.sim.dt, 9.81, base_env.drone.params).to(base_env.device)
            transform = PIDRateController(controller)
            transforms.append(transform)
        elif action_transform == "rate":
            from omni_drones.controllers import RateController as _RateController
            from omni_drones.utils.torchrl.transforms import RateController
            controller = _RateController(9.81, base_env.drone.params).to(base_env.device)
            transform = RateController(controller)
            transforms.append(transform)
        elif not action_transform.lower() == "none":
            raise NotImplementedError(f"Unknown action_transform: {action_transform}")

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    agent_spec = env.agent_spec["drone"]
    policy = algos[cfg.algo.name.lower()](
        cfg.algo, agent_spec=agent_spec, device=cfg.sim.device,
        TP_net=base_env.TP
    )

    if cfg.model_dir is not None:
        policy.load_state_dict(torch.load(cfg.model_dir))
        print(f"✅ Model loaded: {cfg.model_dir}")

    # Set curriculum to hardest stage
    if hasattr(base_env, "_set_curriculum_stage"):
        base_env._set_curriculum_stage(
            len(base_env.curriculum_stages) - 1,
            reset_metrics=False, announce=False,
        )
    if hasattr(base_env, "set_training_progress"):
        base_env.set_training_progress(1.0)

    # Print config
    repulsion_coef = float(getattr(cfg.task, "target_repulsion_coef", 1.0))
    accel_limit = float(getattr(cfg.task, "target_accel_limit", 2.0))
    max_tilt = float(getattr(cfg.task, "target_uav_max_tilt_deg", 25.0))
    print(f"\n{'='*70}")
    print(f"  BATCH EVALUATION: {eval_label}")
    print(f"  Checkpoint: {cfg.model_dir}")
    print(f"  Num envs: {num_envs}")
    print(f"  Target repulsion_coef: {repulsion_coef}")
    print(f"  Target accel_limit: {accel_limit}")
    print(f"  Target max_tilt_deg: {max_tilt}")
    print(f"  v_prey = {cfg.task.v_prey}, v_drone = {cfg.task.v_drone}")
    print(f"  actual target speed = {cfg.task.v_drone * cfg.task.v_prey}")
    print(f"{'='*70}\n")

    base_env.eval()
    env.eval()

    tensordict = env.reset()
    torch.cuda.empty_cache()

    # Track per-env completion
    done_mask = torch.zeros(num_envs, dtype=torch.bool, device=base_env.device)
    results = {
        "success": torch.zeros(num_envs, device=base_env.device),
        "goal_reached": torch.zeros(num_envs, device=base_env.device),
        "out_of_arena": torch.zeros(num_envs, device=base_env.device),
        "collision": torch.zeros(num_envs, device=base_env.device),
        "steps": torch.zeros(num_envs, device=base_env.device),
        "return": torch.zeros(num_envs, device=base_env.device),
    }

    with torch.no_grad():
        for step in tqdm(range(base_env.max_episode_length), desc="Eval"):
            tensordict = env.step(policy(tensordict, deterministic=True))

            done = tensordict.get(("next", "done")).squeeze(-1)
            newly_done = done & (~done_mask)

            if newly_done.any():
                idx = newly_done.nonzero(as_tuple=True)[0]
                stats = tensordict.get(("next", "stats"))
                for k in results.keys():
                    if k in stats.keys():
                        results[k][idx] = stats[k][idx].squeeze(-1)
                results["steps"][idx] = step + 1
                done_mask |= newly_done

                n_done = done_mask.sum().item()
                n_success = results["success"][done_mask].sum().item()
                if n_done % 50 == 0 or n_done == num_envs:
                    print(f"  [{n_done}/{num_envs}] success={n_success/n_done:.1%}")

            if done_mask.all():
                break

            tensordict = tensordict.get("next")

    # Handle envs that timed out
    timeout_mask = ~done_mask
    if timeout_mask.any():
        results["steps"][timeout_mask] = base_env.max_episode_length

    # Compute summary
    n = num_envs
    success_rate = results["success"].sum().item() / n
    goal_rate = results["goal_reached"].sum().item() / n
    ooa_rate = results["out_of_arena"].sum().item() / n
    timeout_rate = timeout_mask.sum().item() / n
    collision_rate = results["collision"].mean().item()
    avg_steps = results["steps"].mean().item()
    success_steps = results["steps"][results["success"].bool()].mean().item() if results["success"].any() else float("nan")

    print(f"\n{'='*70}")
    print(f"  EVALUATION RESULTS: {eval_label}")
    print(f"{'='*70}")
    print(f"  🏆 Success rate:      {success_rate:.1%}  ({int(results['success'].sum().item())}/{n})")
    print(f"  🎯 Goal reached:      {goal_rate:.1%}")
    print(f"  🚧 Out of arena:      {ooa_rate:.1%}")
    print(f"  ⏱  Timeout:           {timeout_rate:.1%}")
    print(f"  💥 Collision rate:     {collision_rate:.3f}")
    print(f"  📊 Avg steps:         {avg_steps:.0f}")
    print(f"  📊 Avg success steps: {success_steps:.0f}")
    print(f"{'='*70}\n")

    # Save results
    out_dir = "analysis/batch_eval"
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(out_dir, f"eval_{eval_label}_{timestamp}.json")
    result_data = {
        "label": eval_label,
        "checkpoint": str(cfg.model_dir),
        "timestamp": timestamp,
        "num_envs": n,
        "seed": int(cfg.seed),
        "target_repulsion_coef": repulsion_coef,
        "target_accel_limit": accel_limit,
        "target_uav_max_tilt_deg": max_tilt,
        "v_prey": float(cfg.task.v_prey),
        "v_drone": float(cfg.task.v_drone),
        "actual_target_speed": float(cfg.task.v_drone * cfg.task.v_prey),
        "results": {
            "success_rate": success_rate,
            "goal_reached_rate": goal_rate,
            "out_of_arena_rate": ooa_rate,
            "timeout_rate": timeout_rate,
            "collision_rate": collision_rate,
            "avg_steps": avg_steps,
            "avg_success_steps": success_steps,
        },
    }
    with open(result_file, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"  Results saved: {result_file}")
    sys.stdout.flush()

    try:
        import wandb as _wandb
        _wandb.finish()
    except Exception:
        pass
    try:
        simulation_app.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
