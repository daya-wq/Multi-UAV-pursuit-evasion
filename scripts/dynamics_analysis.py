"""
Detailed dynamics analysis of a single episode.
Logs positions, velocities, accelerations for all pursuers + target at every step.
Produces analysis of the capture moment.
"""
import os, sys, json, math, datetime
import hydra
import torch
import numpy as np
from omegaconf import OmegaConf
from omni_drones import CONFIG_PATH, init_simulation_app
from omni_drones.utils.wandb import init_wandb
from omni_drones.learning import MAPPOPolicy
from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle("dynamics_analysis")

    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    transforms = [InitTracker()]
    action_transform = cfg.task.get("action_transform", None)
    if action_transform == "PIDrate":
        from omni_drones.controllers import PIDRateController as _PIDRateController
        from omni_drones.utils.torchrl.transforms import PIDRateController
        controller = _PIDRateController(cfg.sim.dt, 9.81, base_env.drone.params).to(base_env.device)
        transforms.append(PIDRateController(controller))

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    agent_spec = env.agent_spec["drone"]
    policy = MAPPOPolicy(
        cfg.algo, agent_spec=agent_spec, device=cfg.sim.device,
        TP_net=base_env.TP
    )
    policy.load_state_dict(torch.load(cfg.model_dir))
    print(f"Model loaded: {cfg.model_dir}")

    if hasattr(base_env, "_set_curriculum_stage"):
        base_env._set_curriculum_stage(len(base_env.curriculum_stages) - 1, reset_metrics=False, announce=False)
    if hasattr(base_env, "set_training_progress"):
        base_env.set_training_progress(1.0)

    base_env.eval()
    env.eval()

    # Seed for reproducibility
    seed = int(cfg.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    tensordict = env.reset()
    dt = float(cfg.sim.dt)
    num_agents = base_env.num_agents

    # Data storage
    history = {
        "dt": dt,
        "seed": seed,
        "catch_radius": float(base_env.catch_radius),
        "v_drone_config": float(cfg.task.v_drone),
        "v_prey_config": float(cfg.task.v_prey),
        "actual_target_speed_limit": float(base_env.current_target_speed),
        "target_accel_limit": float(base_env.target_accel_limit),
        "target_uav_max_tilt_deg": float(base_env.target_uav_max_tilt_deg),
        "target_repulsion_coef": float(getattr(base_env, "target_repulsion_coef", 1.0)),
        "steps": [],
    }

    prev_drone_vel = None
    prev_target_vel = None

    with torch.no_grad():
        for step in range(base_env.max_episode_length):
            # Get current state BEFORE stepping
            drone_pos, _ = base_env.get_env_poses(base_env.drone.get_world_poses(False))
            target_pos, _ = base_env.get_env_poses(base_env.target.get_world_poses())

            drone_state = base_env.drone.get_state()
            drone_vel = drone_state[0, :, 7:10].cpu()  # [num_agents, 3]
            
            if base_env.target_dynamics_mode == "uav":
                target_state = base_env.target.get_state()
                target_vel = target_state[0, :, 7:10].cpu()  # [1, 3]
            else:
                target_v = base_env.target.get_velocities()
                target_vel = target_v[0, :, :3].cpu()

            dp = drone_pos[0].cpu()  # [num_agents, 3]
            tp = target_pos[0].cpu()  # [1, 3]

            # Compute distances
            dists = torch.norm(dp - tp, dim=-1)  # [num_agents]
            min_dist = dists.min().item()

            # Compute speeds
            drone_speeds = torch.norm(drone_vel, dim=-1)  # [num_agents]
            target_speed = torch.norm(target_vel, dim=-1)  # [1]

            # Compute accelerations
            drone_accel_mag = [0.0] * num_agents
            target_accel_mag = 0.0
            if prev_drone_vel is not None:
                drone_acc = (drone_vel - prev_drone_vel) / dt
                drone_accel_mag = torch.norm(drone_acc, dim=-1).tolist()
                target_acc = (target_vel - prev_target_vel) / dt
                target_accel_mag = torch.norm(target_acc, dim=-1).item()

            step_data = {
                "step": step,
                "drone_pos": dp.tolist(),
                "target_pos": tp[0].tolist(),
                "drone_vel": drone_vel.tolist(),
                "target_vel": target_vel[0].tolist(),
                "drone_speeds": drone_speeds.tolist(),
                "target_speed": target_speed.item(),
                "drone_accel": drone_accel_mag,
                "target_accel": target_accel_mag,
                "dists_to_target": dists.tolist(),
                "min_dist": min_dist,
            }
            history["steps"].append(step_data)

            prev_drone_vel = drone_vel.clone()
            prev_target_vel = target_vel.clone()

            # Step env
            tensordict = env.step(policy(tensordict, deterministic=True))
            done = tensordict.get(("next", "done")).squeeze(-1).item()

            if done:
                stats = tensordict.get(("next", "stats"))
                history["done_reason"] = "SUCCESS" if stats["success"].item() > 0.5 else (
                    "GOAL_ZONE" if stats["goal_reached"].item() > 0.5 else "OTHER"
                )
                history["total_steps"] = step + 1
                print(f"Episode done at step {step+1}: {history['done_reason']}")
                break

            tensordict = tensordict.get("next")

    # Save raw data
    out_dir = str(cfg.get("out_dir", "analysis/dynamics"))
    os.makedirs(out_dir, exist_ok=True)
    raw_file = os.path.join(out_dir, f"dynamics_seed{seed}.json")
    with open(raw_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Raw data saved: {raw_file}")

    # ===== ANALYSIS =====
    steps = history["steps"]
    total = len(steps)
    print(f"\n{'='*70}")
    print(f"  DYNAMICS ANALYSIS — seed={seed}, {total} steps, dt={dt}")
    print(f"  Done reason: {history.get('done_reason', 'TIMEOUT')}")
    print(f"  catch_radius: {history['catch_radius']}")
    print(f"{'='*70}")

    # Aggregate stats
    drone_speeds_all = [s["drone_speeds"] for s in steps]
    target_speeds_all = [s["target_speed"] for s in steps]
    drone_accel_all = [s["drone_accel"] for s in steps[1:]]
    target_accel_all = [s["target_accel"] for s in steps[1:]]
    min_dists = [s["min_dist"] for s in steps]

    avg_drone_speed = np.mean([np.mean(ds) for ds in drone_speeds_all])
    max_drone_speed = max([max(ds) for ds in drone_speeds_all])
    avg_target_speed = np.mean(target_speeds_all)
    max_target_speed = max(target_speeds_all)

    avg_drone_accel = np.mean([np.mean(da) for da in drone_accel_all]) if drone_accel_all else 0
    max_drone_accel = max([max(da) for da in drone_accel_all]) if drone_accel_all else 0
    avg_target_accel = np.mean(target_accel_all) if target_accel_all else 0
    max_target_accel = max(target_accel_all) if target_accel_all else 0

    print(f"\n  --- Speed (m/s) ---")
    print(f"  Pursuer avg: {avg_drone_speed:.3f}  max: {max_drone_speed:.3f}")
    print(f"  Target  avg: {avg_target_speed:.3f}  max: {max_target_speed:.3f}")
    print(f"  Speed ratio (target/pursuer): {avg_target_speed/max(avg_drone_speed,1e-6):.2f}x")

    print(f"\n  --- Acceleration (m/s²) ---")
    print(f"  Pursuer avg: {avg_drone_accel:.3f}  max: {max_drone_accel:.3f}")
    print(f"  Target  avg: {avg_target_accel:.3f}  max: {max_target_accel:.3f}")

    print(f"\n  --- Distance ---")
    print(f"  Initial min dist: {min_dists[0]:.3f}")
    print(f"  Final min dist:   {min_dists[-1]:.3f}")
    print(f"  Overall min dist: {min(min_dists):.3f}")

    # Find capture moment and analyze last 20 steps
    capture_idx = None
    for i, s in enumerate(steps):
        if s["min_dist"] < history["catch_radius"]:
            capture_idx = i
            break

    if capture_idx is not None:
        print(f"\n  --- Capture Moment (step {capture_idx}) ---")
        window_start = max(0, capture_idx - 20)
        print(f"  Last 20 steps before capture (steps {window_start}-{capture_idx}):")
        print(f"  {'step':>5} | {'min_dist':>8} | {'tgt_speed':>9} | {'tgt_accel':>9} | {'p0_speed':>8} | {'p1_speed':>8} | {'p2_speed':>8}")
        print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*9}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
        for i in range(window_start, min(capture_idx + 1, total)):
            s = steps[i]
            ds = s["drone_speeds"]
            print(f"  {s['step']:5d} | {s['min_dist']:8.4f} | {s['target_speed']:9.4f} | {s['target_accel']:9.4f} | {ds[0]:8.4f} | {ds[1]:8.4f} | {ds[2]:8.4f}")

        # Analyze target velocity direction vs pursuer direction at capture
        cs = steps[capture_idx]
        nearest_idx = int(np.argmin(cs["dists_to_target"]))
        pursuer_dir = np.array(cs["drone_pos"][nearest_idx]) - np.array(cs["target_pos"])
        target_vel_vec = np.array(cs["target_vel"])
        
        # Check if target is moving toward or away from nearest pursuer
        pursuer_dir_norm = pursuer_dir / (np.linalg.norm(pursuer_dir) + 1e-6)
        target_vel_norm = target_vel_vec / (np.linalg.norm(target_vel_vec) + 1e-6)
        cos_angle = np.dot(pursuer_dir_norm, target_vel_norm)
        angle_deg = math.degrees(math.acos(np.clip(cos_angle, -1, 1)))
        
        print(f"\n  At capture:")
        print(f"    Nearest pursuer: drone_{nearest_idx}")
        print(f"    Target vel direction vs nearest pursuer: {angle_deg:.1f}° (0°=toward, 180°=away)")
        print(f"    Target speed: {cs['target_speed']:.4f} m/s")
        print(f"    Nearest pursuer speed: {cs['drone_speeds'][nearest_idx]:.4f} m/s")
        if angle_deg < 90:
            print(f"    ⚠️ TARGET IS MOVING TOWARD THE PURSUER (not evading!)")
        else:
            print(f"    ✅ Target is moving away from pursuer")

    print(f"\n{'='*70}")
    sys.stdout.flush()

    try:
        import wandb as _wandb
        _wandb.finish()
    except: pass
    try:
        simulation_app.close()
    except: pass


if __name__ == "__main__":
    main()
