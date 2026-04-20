"""
Record multiple evaluation videos for an RL checkpoint in a single Isaac Sim session.
Usage:
  python scripts/record_rl_videos.py \
    task=HideAndSeek headless=true wandb.mode=disabled \
    model_dir=<checkpoint_path> \
    algo.use_TP_net=1 \
    task.env.num_envs=1 task.v_prey=1.5 task.v_drone=1.5 \
    algo.train_every=64 \
    algo.actor.log_std_init=-2.0 algo.actor.log_std_min=-2.0 algo.actor.log_std_max=-1.0 \
    seed=42 \
    +video_seeds="[42,100,200,300,400,500,789,1024,2026,3000]" \
    +video_dir="analysis/rl_v7_best_videos"
"""

import logging
import os
import time
import json
import datetime

import hydra
import torch
import numpy as np
import wandb

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
    
    # Extract video config
    video_seeds = list(cfg.get("video_seeds", [42, 100, 200, 300, 400, 500]))
    video_dir = str(cfg.get("video_dir", "analysis/rl_eval_videos"))
    os.makedirs(video_dir, exist_ok=True)
    
    # Force single env for video recording
    cfg.task.env.num_envs = 1
    if hasattr(cfg.task, "num_envs"):
        cfg.task.num_envs = 1
    
    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle("rl_video_recorder")

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

    # Build transforms (same as eval.py / train.py)
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

    # Load checkpoint
    if cfg.model_dir is not None:
        policy.load_state_dict(torch.load(cfg.model_dir))
        print(f"✅ Model loaded: {cfg.model_dir}")

    dt = float(cfg.sim.dt)
    video_fps = int(cfg.get("video_fps", 24))
    video_frame_skip = int(cfg.get("video_frame_skip", 1))  # 1 = every step
    
    # Set curriculum to hardest stage
    if hasattr(base_env, "_set_curriculum_stage"):
        base_env._set_curriculum_stage(
            len(base_env.curriculum_stages) - 1,
            reset_metrics=False,
            announce=False,
        )
    if hasattr(base_env, "set_training_progress"):
        base_env.set_training_progress(1.0)

    print(f"\n{'='*60}")
    print(f"  Recording {len(video_seeds)} evaluation videos")
    print(f"  Checkpoint: {cfg.model_dir}")
    print(f"  Output: {video_dir}")
    print(f"  Seeds: {video_seeds}")
    print(f"  FPS: {video_fps} (frame_skip={video_frame_skip})")
    print(f"{'='*60}\n")

    summary = []
    
    for vid_idx, seed in enumerate(video_seeds):
        print(f"\n--- Video {vid_idx+1}/{len(video_seeds)} | seed={seed} ---")
        
        # Enable rendering
        base_env.enable_render(True)
        base_env.eval()
        env.eval()
        env.set_seed(seed)
        
        frames = []
        tensordict = env.reset()
        step_count = 0
        done_reason = "TIMEOUT"
        
        for _ in tqdm(range(base_env.max_episode_length), desc=f"Seed {seed}"):
            # Capture frame
            if step_count % video_frame_skip == 0:
                try:
                    frame = env.base_env.render(mode="rgb_array")
                    frames.append(frame)
                except Exception as e:
                    pass
            
            tensordict = env.step(policy(tensordict, deterministic=True))
            
            done = tensordict.get(("next", "done"))
            if done.any():
                stats = tensordict.get(("next", "stats"))
                if "success" in stats.keys() and stats["success"].any():
                    done_reason = "SUCCESS"
                elif "goal_reached" in stats.keys() and stats["goal_reached"].any():
                    done_reason = "GOAL_ZONE"
                elif "any_landed" in stats.keys() and stats["any_landed"].any():
                    done_reason = "CRASHED"
                else:
                    done_reason = "OUT_OF_ARENA"
                break
            
            tensordict = tensordict.get("next")
            step_count += 1
        
        # Save video
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(
            video_dir, f"v7_best_seed{seed}_{done_reason}.mp4"
        )
        
        if frames:
            try:
                import imageio
                frames_rgb = np.stack(frames)
                imageio.mimwrite(video_path, frames_rgb, fps=video_fps)
                print(f"  ✅ {done_reason} | steps={step_count} | frames={len(frames)} | saved={video_path}")
            except Exception as e:
                npy_path = video_path.replace(".mp4", ".npy")
                np.save(npy_path, np.stack(frames))
                print(f"  ⚠️ mp4 failed ({e}), saved npy: {npy_path}")
                video_path = npy_path
        else:
            print(f"  ❌ No frames captured for seed {seed}")
            video_path = None
        
        # Save episode metadata
        meta_path = os.path.join(
            video_dir, f"v7_best_seed{seed}_{done_reason}.json"
        )
        meta = {
            "seed": seed,
            "checkpoint": str(cfg.model_dir),
            "done_reason": done_reason,
            "steps": step_count,
            "n_frames": len(frames),
            "video_path": video_path,
            "timestamp": time_str,
        }
        # Try to extract stats
        try:
            stats_dict = tensordict.get(("next", "stats"), {})
            for k, v in stats_dict.items():
                meta[f"stats/{k}"] = float(torch.nanmean(v.float()).item())
        except:
            pass
        
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        
        summary.append(meta)
    
    # Print summary
    print(f"\n{'='*60}")
    print("  RECORDING SUMMARY")
    print(f"{'='*60}")
    
    outcomes = {}
    for s in summary:
        r = s["done_reason"]
        outcomes[r] = outcomes.get(r, 0) + 1
    
    for reason, count in sorted(outcomes.items()):
        icon = {"SUCCESS": "🏆", "GOAL_ZONE": "🎯", "OUT_OF_ARENA": "🚧", "CRASHED": "💥", "TIMEOUT": "⏱"}.get(reason, "❓")
        print(f"  {icon} {reason}: {count}")
    
    print(f"\n  Total videos: {len(summary)}")
    print(f"  Output dir: {video_dir}")
    
    # Save full summary
    summary_path = os.path.join(video_dir, "recording_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {summary_path}")
    
    wandb.finish()
    simulation_app.close()


if __name__ == "__main__":
    main()
