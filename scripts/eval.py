import logging
import os
import time

import hydra
import torch
import numpy as np
import wandb
import datetime

from functorch import vmap
from omegaconf import OmegaConf

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

class Every:
    def __init__(self, func, steps):
        self.func = func
        self.steps = steps
        self.i = 0

    def __call__(self, *args, **kwargs):
        if self.i % self.steps == 0:
            self.func(*args, **kwargs)
        self.i += 1

from typing import Sequence
from tensordict import TensorDictBase

class EpisodeStats:
    def __init__(self, in_keys: Sequence[str] = None):
        self.in_keys = in_keys
        self._stats = []
        self._episodes = 0

    def __call__(self, tensordict: TensorDictBase) -> TensorDictBase:
        done = tensordict.get(("next", "done"))
        truncated = tensordict.get(("next", "truncated"), None)
        done_or_truncated = (
            (done | truncated) if truncated is not None else done.clone()
        )
        if done_or_truncated.any():
            done_or_truncated = done_or_truncated.squeeze(-1) # [env_num, 1, 1]
            self._episodes += done_or_truncated.sum().item()
            self._stats.extend(
                # [env, n, 1]
                tensordict.select(*self.in_keys)[:, 1:][done_or_truncated[:, :-1]].clone().unbind(0)
            )
    
    def pop(self):
        stats: TensorDictBase = torch.stack(self._stats).to_tensordict()
        self._stats.clear()
        return stats

    def __len__(self):
        return len(self._stats)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

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

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    transforms = [InitTracker()]

    # a CompositeSpec is by deafault processed by a entity-based encoder
    # flatten it to use a MLP encoder instead
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
    
    # optionally discretize the action space or use a controller
    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromMultiDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform == "velocity":
            from omni_drones.controllers import LeePositionController
            from omni_drones.utils.torchrl.transforms import VelController
            controller = LeePositionController(9.81, base_env.drone.params).to(base_env.device)
            transform = VelController(controller)
            transforms.append(transform)
        elif action_transform == "attitude":
            from omni_drones.controllers import AttitudeController as Controller
            from omni_drones.utils.torchrl.transforms import AttitudeController
            controller = Controller(9.81, base_env.drone.params).to(base_env.device)
            transform = AttitudeController(controller)
            transforms.append(transform)
        elif action_transform == "rate":
            from omni_drones.controllers import RateController as _RateController
            from omni_drones.utils.torchrl.transforms import RateController
            # from torch.distributions.transforms import TanhTransform
            controller = _RateController(9.81, base_env.drone.params).to(base_env.device)
            transform = RateController(controller)
            # transforms.append(TanhTransform)
            transforms.append(transform)
        elif action_transform == "PIDrate":
            from omni_drones.controllers import PIDRateController as _PIDRateController
            from omni_drones.utils.torchrl.transforms import PIDRateController
            controller = _PIDRateController(cfg.sim.dt, 9.81, base_env.drone.params).to(base_env.device)
            transform = PIDRateController(controller)
            # transforms.append(TanhTransform)
            transforms.append(transform)
        elif not action_transform.lower() == "none":
            raise NotImplementedError(f"Unknown action transform: {action_transform}")
    
    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    agent_spec: AgentSpec = env.agent_spec["drone"]
    # add base_env.TP to MAPPOPolicy
    policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=agent_spec, device=cfg.sim.device, TP_net=base_env.TP)

    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    eval_interval = cfg.get("eval_interval", -1)
    save_interval = cfg.get("save_interval", -1)

    if cfg.model_dir is not None:
        # torch.save(policy.state_dict(), ckpt_path)
        policy.load_state_dict(torch.load(cfg.model_dir))
        print("Successfully load model!")

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys)

    @torch.no_grad()
    def evaluate(
        seed: int=0
    ):
        frames = []

        base_env.enable_render(True)
        base_env.eval()
        env.eval()
        env.set_seed(seed)
        if hasattr(base_env, "set_training_progress"):
            base_env.set_training_progress(1.0)

        from tqdm import tqdm
        
        tensordict = env.reset()
        step_count = 0

        for _ in tqdm(range(base_env.max_episode_length)):
            if step_count % 2 == 0:
                frame = env.base_env.render(mode="rgb_array")
                frames.append(frame)

            tensordict = env.step(policy(tensordict, deterministic=True))

            done = tensordict.get(("next", "done"))
            if done.any():
                stats = tensordict.get(("next", "stats"))
                print(f"\n" + "="*60)
                print(f"🎬 Video Recording Stopped at Step {step_count}")
                if "success" in stats.keys() and stats["success"].any():
                    print("🏆 Termination Reason: 捕获成功 (Success) !!!")
                elif "goal_reached" in stats.keys() and stats["goal_reached"].any():
                    print("🎯 Termination Reason: 目标进入守区 (Goal Reached) !!!")
                elif "any_landed" in stats.keys() and stats["any_landed"].any():
                    print("💥 Termination Reason: 坠机或撞地 (Crashed/Landed) !!!")
                else:
                    print("🚧 Termination Reason: 出界或受到碰撞等其它中断条件 (Out of bounds / Collision) !!!")
                print("="*60 + "\n")
                break

            tensordict = tensordict.get("next")
            step_count += 1
            
        if step_count >= base_env.max_episode_length:
            print(f"\n" + "="*60)
            print(f"🎬 Video Recording Stopped at Step {step_count}")
            print("⏳ Termination Reason: 达到最大步数限制 (Timeout 800 steps) !!!")
            print("="*60 + "\n")

        base_env.enable_render(not cfg.headless)

        # Get final stats
        stats_dict = tensordict.get(("next", "stats"), {})
        info = {}
        for k, v in stats_dict.items():
            info["eval/stats." + k] = torch.nanmean(v.float()).item()

        if len(frames):
            # video_array = torch.stack(frames)
            video_array = np.stack(frames).transpose(0, 3, 1, 2)
            frames_rgb = np.stack(frames)  # keep HWC for local save

            # Save the local artifact first so local review does not depend on wandb.
            time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            video_dir = os.path.join(project_root, "eval_videos")
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f"{cfg.task.name}_{time_str}.mp4")
            fps = int(0.5 / cfg.sim.dt)
            saved_artifact_path = None
            try:
                import imageio
                imageio.mimwrite(video_path, frames_rgb, fps=fps)
                saved_artifact_path = video_path
                logging.info(f"Video saved to {video_path}")
            except Exception as exc:
                # Fallback: keep raw frames even if ffmpeg/imageio export fails.
                npy_path = video_path.replace(".mp4", ".npy")
                np.save(npy_path, frames_rgb)
                saved_artifact_path = npy_path
                logging.warning(
                    "Failed to save mp4 (%s); frames saved to %s",
                    exc,
                    npy_path,
                )

            info["eval/local_recording_path"] = saved_artifact_path
            frames.clear()
            try:
                info["recording"] = wandb.Video(
                    video_array, fps=fps, format="mp4"
                )
            except Exception as exc:
                logging.warning("wandb.Video export failed: %s", exc)

        return info

    info = {}
    info.update(evaluate())
    
    import pprint
    print("-" * 40)
    print("Evaluation Stats:")
    pprint.pprint({k: v for k, v in info.items() if not isinstance(v, wandb.Video)})
    print("-" * 40)

    run.log(info)

    wandb.finish()
    
    simulation_app.close()


if __name__ == "__main__":
    main()
