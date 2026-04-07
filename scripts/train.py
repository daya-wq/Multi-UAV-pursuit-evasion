import logging
import os
import time

import hydra
import torch
import numpy as np
import wandb
import datetime
from torch.utils.tensorboard import SummaryWriter

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

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg):
    set_seed(cfg.seed)

    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

    # Project-local output directory with timestamp
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_output_dir = os.path.join(project_root, "checkpoints", f"{cfg.task.name}_{time_str}")
    os.makedirs(run_output_dir, exist_ok=True)
    logging.info(f"Run output dir: {run_output_dir}")

    # TensorBoard writer — run name is the timestamp
    tb_log_dir = os.path.join(project_root, "runs", f"{cfg.task.name}_{time_str}")
    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    logging.info(f"TensorBoard log dir: {tb_log_dir}")

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

    checkpoint_state = None
    if cfg.model_dir is not None:
        checkpoint_state = torch.load(cfg.model_dir, map_location="cpu")
        if checkpoint_has_bc_aux(checkpoint_state):
            cfg.algo.actor.bc_aux.enabled = True
        checkpoint_rnn_hidden = checkpoint_actor_rnn_hidden_size(checkpoint_state)
        if checkpoint_rnn_hidden is not None:
            enable_actor_rnn_cfg(cfg, hidden_size=int(checkpoint_rnn_hidden))
        checkpoint_prev_action_hidden = checkpoint_prev_action_condition_hidden_size(checkpoint_state)
        if checkpoint_prev_action_hidden is not None:
            enable_prev_action_conditioning_cfg(cfg, hidden_dim=int(checkpoint_prev_action_hidden))

    agent_spec: AgentSpec = env.agent_spec["drone"]
    # add base_env.TP to MAPPOPolicy
    policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=agent_spec, device=cfg.sim.device, TP_net=base_env.TP)

    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    eval_interval = cfg.get("eval_interval", -1)
    save_interval = cfg.get("save_interval", -1)

    if checkpoint_state is not None:
        # torch.save(policy.state_dict(), ckpt_path)
        policy.load_state_dict(checkpoint_state)
        print("Successfully load model!")
    elif cfg.get("tp_model_dir", None) is not None:
        if not getattr(policy, "use_TP_net", False) or getattr(policy, "TP_net", None) is None:
            raise ValueError("tp_model_dir was provided, but the current policy does not use TP_net.")
        tp_state = torch.load(cfg.tp_model_dir)
        if isinstance(tp_state, dict) and "TP" in tp_state:
            tp_state = tp_state["TP"]
        policy.TP_net.load_state_dict(tp_state)
        print("Successfully load TP model!")

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )

    entropy_schedule_cfg = cfg.algo.get("entropy_schedule", None)
    entropy_schedule_enabled = bool(
        entropy_schedule_cfg is not None and entropy_schedule_cfg.get("enabled", False)
    )
    entropy_success_ema = None
    if entropy_schedule_enabled:
        entropy_hold_coef = float(entropy_schedule_cfg.get("hold_coef", cfg.algo.entropy_coef))
        entropy_release_coef = float(entropy_schedule_cfg.get("release_coef", cfg.algo.entropy_coef))
        entropy_success_threshold = float(entropy_schedule_cfg.get("success_threshold", 0.5))
        entropy_success_target = float(entropy_schedule_cfg.get("success_target", entropy_success_threshold))
        entropy_ema_alpha = float(entropy_schedule_cfg.get("ema_alpha", 0.05))
        entropy_min_env_frames = int(entropy_schedule_cfg.get("min_env_frames", 0))
        if hasattr(policy, "set_entropy_coef"):
            policy.set_entropy_coef(entropy_hold_coef)
        else:
            policy.entropy_coef = entropy_hold_coef
    else:
        entropy_hold_coef = float(cfg.algo.entropy_coef)
        entropy_release_coef = float(cfg.algo.entropy_coef)
        entropy_success_threshold = 0.5
        entropy_success_target = 0.5
        entropy_ema_alpha = 0.05
        entropy_min_env_frames = 0

    def update_entropy_schedule(current_frames: int, train_stats: dict):
        nonlocal entropy_success_ema

        if not entropy_schedule_enabled:
            return {
                "drone/effective_entropy_coef": float(getattr(policy, "entropy_coef", cfg.algo.entropy_coef))
            }

        batch_success = train_stats.get("train/stats.success")
        if batch_success is not None:
            batch_success = float(batch_success)
            if entropy_success_ema is None:
                entropy_success_ema = batch_success
            else:
                entropy_success_ema = (
                    (1.0 - entropy_ema_alpha) * entropy_success_ema
                    + entropy_ema_alpha * batch_success
                )

        effective_entropy_coef = entropy_hold_coef
        ready_to_release = (
            entropy_success_ema is not None
            and current_frames >= entropy_min_env_frames
            and entropy_success_ema >= entropy_success_threshold
        )
        if ready_to_release:
            success_span = max(entropy_success_target - entropy_success_threshold, 1e-6)
            release_progress = min(
                max((entropy_success_ema - entropy_success_threshold) / success_span, 0.0),
                1.0,
            )
            effective_entropy_coef = entropy_hold_coef + (
                entropy_release_coef - entropy_hold_coef
            ) * release_progress

        if hasattr(policy, "set_entropy_coef"):
            policy.set_entropy_coef(effective_entropy_coef)
        else:
            policy.entropy_coef = effective_entropy_coef

        return {
            "drone/effective_entropy_coef": float(effective_entropy_coef),
            "train/stats.success_ema_for_entropy": float(entropy_success_ema or 0.0),
        }

    def training_progress(frames: int, iteration: int) -> float:
        if total_frames > 0:
            return min(float(frames) / float(total_frames), 1.0)
        if max_iters > 0:
            return min(float(iteration + 1) / float(max_iters), 1.0)
        return 0.0

    if hasattr(base_env, "set_training_progress"):
        base_env.set_training_progress(0.0)

    @torch.no_grad()
    def evaluate(
        seed: int=0,
        progress: float=1.0,
    ):
        frames = []
        record_eval_video = bool(cfg.get("record_eval_video", False))

        base_env.enable_render(record_eval_video)
        base_env.eval()
        env.eval()
        env.set_seed(seed)
        prev_progress = getattr(base_env, "training_progress", 0.0)
        if hasattr(base_env, "set_training_progress"):
            base_env.set_training_progress(progress)

        from tqdm import tqdm
        t = tqdm(total=base_env.max_episode_length)
        
        def record_frame(*args, **kwargs):
            frame = env.base_env.render(mode="rgb_array")
            if frame is not None:
                frames.append(frame)
            t.update(2)

        rollout_callback = Every(record_frame, 2) if record_eval_video else None
        trajs = env.rollout(
            max_steps=base_env.max_episode_length,
            policy=lambda x: policy(x, deterministic=True),
            callback=rollout_callback,
            auto_reset=True,
            break_when_any_done=False,
            return_contiguous=False
        ).clone()
        # np.save('track.npy', trajs[0]['stats']['drone_state'].to('cpu').numpy())

        base_env.enable_render(not cfg.headless)
        if hasattr(base_env, "set_training_progress"):
            base_env.set_training_progress(prev_progress)
        env.reset()

        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        def take_first_episode(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
            return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

        traj_stats = {
            k: take_first_episode(v)
            for k, v in trajs[("next", "stats")].cpu().items()
        }

        info = {
            "eval/stats." + k: torch.nanmean(v.float()).item() 
            for k, v in traj_stats.items()
        }

        if record_eval_video and len(frames):
            # video_array = torch.stack(frames)
            video_array = np.stack(frames).transpose(0, 3, 1, 2)
            frames.clear()
            info["recording"] = wandb.Video(
                video_array, fps=0.5 / cfg.sim.dt, format="mp4"
            )
        
        return info

    pbar = tqdm(collector)
    env.train()
    fps = []
    for i, data in enumerate(pbar):
        # fps.append(collector._fps)
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
        episode_stats(data.to_tensordict())
        stats = {}

        if len(episode_stats) >= base_env.num_envs:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v).item() 
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

        info.update(update_entropy_schedule(collector._frames, stats))
        
        info.update(policy.train_op(data.to_tensordict()))

        if eval_interval > 0 and i % eval_interval == 0:
            logging.info(f"Eval at {collector._frames} steps.")
            info.update(evaluate(progress=training_progress(collector._frames, i)))
            env.train()

        if save_interval > 0 and i % save_interval == 0:
            if hasattr(policy, "state_dict"):
                ckpt_path = os.path.join(run_output_dir, f"checkpoint_{collector._frames}.pt")
                logging.info(f"Save checkpoint to {str(ckpt_path)}")
                torch.save(policy.state_dict(), ckpt_path)

        run.log(info)
        # Log scalars to TensorBoard
        global_step = collector._frames
        for k, v in info.items():
            if isinstance(v, (int, float)):
                tb_writer.add_scalar(k, v, global_step)
        tb_writer.flush()
        print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))

        pbar.set_postfix({
            "rollout_fps": collector._fps,
            "frames": collector._frames,
        })

        if hasattr(base_env, "set_training_progress"):
            base_env.set_training_progress(training_progress(collector._frames, i))

        if max_iters > 0 and i >= max_iters - 1:
            break 

        # if len(fps) > 50:
        #     fps = np.array(fps)[10:]
        #     print(fps.mean(), fps.std())
        #     exit()
    
    logging.info(f"Final Eval at {collector._frames} steps.")
    info = {"env_frames": collector._frames}
    info.update(evaluate(progress=training_progress(collector._frames, i if 'i' in locals() else 0)))
    run.log(info)

    if hasattr(policy, "state_dict"):
        ckpt_path = os.path.join(run_output_dir, "checkpoint_final.pt")
        logging.info(f"Save checkpoint to {str(ckpt_path)}")
        torch.save(policy.state_dict(), ckpt_path)

    wandb.save(os.path.join(run_output_dir, "checkpoint*"))
    wandb.finish()
    tb_writer.close()
    
    simulation_app.close()


if __name__ == "__main__":
    main()
