# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from functorch import vmap
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.utils import expand_right
from tensordict.nn import make_functional, TensorDictModule, TensorDictParams
from torch.optim import lr_scheduler

from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    MultiDiscreteTensorSpec,
    DiscreteTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec as UnboundedTensorSpec,
)

from omni_drones.utils.torchrl.env import AgentSpec

from .utils import valuenorm
from .utils.gae import compute_gae

LR_SCHEDULER = lr_scheduler._LRScheduler
from torchrl.modules import TanhNormal, IndependentNormal


class MAPPOPolicy(object):
    def __init__(
        self, cfg, agent_spec: AgentSpec, device="cuda", TP_net=None
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.agent_spec = agent_spec
        self.device = device

        print(self.agent_spec.observation_spec)
        print(self.agent_spec.action_spec)

        self.clip_param = cfg.clip_param
        self.ppo_epoch = int(cfg.ppo_epochs)
        self.TP_epoch = int(cfg.TP_epochs)
        self.num_minibatches = int(cfg.num_minibatches)
        self.normalize_advantages = cfg.normalize_advantages

        self.entropy_coef = cfg.entropy_coef
        self.gae_gamma = cfg.gamma
        self.gae_lambda = cfg.gae_lambda

        self.act_dim = agent_spec.action_spec.shape[-1]

        if cfg.reward_weights is not None:
            self.reward_weights = torch.as_tensor(cfg.reward_weights, device=device).float()
        else:
            self.reward_weights = torch.ones(
                self.agent_spec.reward_spec.shape, device=device
            )

        self.obs_name = ("agents", "observation")
        self.act_name = ("agents", "action")
        self.prev_act_name = ("agents", "prev_action")
        self.raw_act_name = ("agents", "action_raw")
        self.expert_act_name = ("agents", "expert_action")
        self.reward_name = ("agents", "reward")
        self._cached_prev_action = None

        self.make_actor()
        self.make_critic()
        self.use_TP_net = self.cfg.use_TP_net
        self.TP_net = TP_net
        self.TP_optimizer = torch.optim.Adam(self.TP_net.parameters(), lr=0.0001)
        self.TP_criterion = nn.MSELoss()
        expert_kl_cfg = self.cfg.get("warmstart", {}).get("expert_kl", None)
        self.expert_kl_enabled = bool(expert_kl_cfg is not None and expert_kl_cfg.get("enabled", False))
        self.expert_kl_coef: float = 0.0
        self.expert_kl_loss_type: str = "nll"
        self.expert_kl_max_loss: Optional[float] = None
        self.expert_kl_fixed_std: float = 0.25
        self.expert_kl_mse_boost_enabled: bool = False
        self.expert_kl_mse_boost_threshold: float = 0.02
        self.expert_kl_mse_boost_coef: float = 2.0
        self.expert_kl_mse_boost_ema_beta: float = 0.95
        self.expert_kl_mse_ema: Optional[float] = None
        if self.expert_kl_enabled:
            self.expert_kl_loss_type = str(expert_kl_cfg.get("loss_type", "nll"))
            max_loss = expert_kl_cfg.get("max_loss", None)
            self.expert_kl_max_loss = None if max_loss is None else float(max_loss)
            self.expert_kl_fixed_std = float(expert_kl_cfg.get("fixed_std", self.expert_kl_fixed_std))
            self.expert_kl_mse_boost_enabled = bool(expert_kl_cfg.get("mse_boost_enabled", False))
            self.expert_kl_mse_boost_threshold = float(
                expert_kl_cfg.get("mse_boost_threshold", self.expert_kl_mse_boost_threshold)
            )
            self.expert_kl_mse_boost_coef = float(
                expert_kl_cfg.get("mse_boost_coef", self.expert_kl_mse_boost_coef)
            )
            self.expert_kl_mse_boost_ema_beta = float(
                expert_kl_cfg.get("mse_boost_ema_beta", self.expert_kl_mse_boost_ema_beta)
            )

        self.train_in_keys = list(
            set(
                self.actor_in_keys
                + self.actor_out_keys
                + self.critic_in_keys
                + self.critic_out_keys
                + [
                    "next",
                    self.act_logps_name,
                    self.raw_act_name,
                    ("reward", self.reward_name),
                    "state_value",
                ]
                + ["progress", ("collector", "traj_ids")]
                + ([self.expert_act_name] if self.expert_kl_enabled else [])
            )
        )

        self.n_updates = 0
        self._tp_debug_emitted = False
        self.actor_frozen = False
        self.ppo_target_kl: Optional[float] = None
        self.ppo_kl_coef: float = 0.0
        self.ppo_kl_hard_stop: bool = False
        self.anchor_actor_params: Optional[TensorDictParams] = None
        self.anchor_loss_coef: float = 0.0
        self.anchor_loss_type: str = "mse"

    @property
    def act_logps_name(self):
        return f"{self.agent_spec.name}.action_logp"

    def make_actor(self):
        cfg = self.cfg.actor

        self.actor_in_keys = [self.obs_name, self.act_name]
        prev_action_cfg = cfg.get("prev_action_conditioning", None)
        self.actor_has_prev_action_conditioning = bool(
            prev_action_cfg is not None and prev_action_cfg.get("enabled", False)
        )
        if self.actor_has_prev_action_conditioning:
            self.actor_in_keys = [self.obs_name, self.prev_act_name, self.act_name]
        self.actor_out_keys = [
            self.act_name,
            self.act_logps_name,
            f"{self.agent_spec.name}.action_entropy",
        ]
        self.bc_aux_out_keys = []
        bc_aux_cfg = cfg.get("bc_aux", None)
        self.actor_has_bc_aux = bool(
            bc_aux_cfg is not None and bc_aux_cfg.get("enabled", False)
        )
        if self.actor_has_bc_aux:
            self.bc_aux_out_keys = [
                ("bc_aux_pred", "vel_cmd"),
                ("bc_aux_pred", "waypoint"),
                ("bc_aux_pred", "target_pos_pred"),
                ("bc_aux_pred", "forward_dir"),
                ("bc_aux_pred", "assignment_logits"),
                ("bc_aux_pred", "trap_logit"),
            ]
            self.actor_out_keys.extend(self.bc_aux_out_keys)

        if cfg.get("rnn", None):
            self.actor_in_keys.extend(
                [f"{self.agent_spec.name}.actor_rnn_state", "is_init"]
            )
            self.actor_out_keys.append(f"{self.agent_spec.name}.actor_rnn_state")
            self.minibatch_seq_len = self.cfg.actor.rnn.train_seq_len
            assert self.minibatch_seq_len <= self.cfg.train_every

        create_actor_fn = lambda: TensorDictModule(
            make_ppo_actor(
                cfg, self.agent_spec.observation_spec, self.agent_spec.action_spec
            ),
            in_keys=self.actor_in_keys,
            out_keys=self.actor_out_keys
        ).to(self.device)

        if self.cfg.share_actor:
            self.actor = create_actor_fn()
            self.actor_params = TensorDictParams(make_functional(self.actor))
        else:
            actors = nn.ModuleList([create_actor_fn() for _ in range(self.agent_spec.n)])
            self.actor = actors[0]
            stacked_params = torch.stack([make_functional(actor) for actor in actors])
            self.actor_params = TensorDictParams(stacked_params.to_tensordict())
        
        self.actor_opt = torch.optim.Adam(self.actor_params.parameters(), lr=cfg.lr)
        actor_scheduler = cfg.get("lr_scheduler", None) if hasattr(cfg, "get") else getattr(cfg, "lr_scheduler", None)
        if actor_scheduler is not None:
            actor_scheduler = eval(actor_scheduler)
            actor_scheduler_kwargs = cfg.get("lr_scheduler_kwargs", {}) if hasattr(cfg, "get") else getattr(cfg, "lr_scheduler_kwargs", {})
            if actor_scheduler_kwargs is None:
                actor_scheduler_kwargs = {}
            self.actor_opt_scheduler: LR_SCHEDULER = actor_scheduler(
                self.actor_opt, **actor_scheduler_kwargs
            )

    def _reset_actor_optimizer(self):
        cfg = self.cfg.actor
        self.actor_opt = torch.optim.Adam(self.actor_params.parameters(), lr=cfg.lr)
        actor_scheduler = cfg.get("lr_scheduler", None) if hasattr(cfg, "get") else getattr(cfg, "lr_scheduler", None)
        if actor_scheduler is None:
            if hasattr(self, "actor_opt_scheduler"):
                delattr(self, "actor_opt_scheduler")
            return
        actor_scheduler = eval(actor_scheduler)
        actor_scheduler_kwargs = cfg.get("lr_scheduler_kwargs", {}) if hasattr(cfg, "get") else getattr(cfg, "lr_scheduler_kwargs", {})
        if actor_scheduler_kwargs is None:
            actor_scheduler_kwargs = {}
        self.actor_opt_scheduler = actor_scheduler(self.actor_opt, **actor_scheduler_kwargs)

    def make_critic(self):
        cfg = self.cfg.critic

        if cfg.use_huber_loss:
            self.critic_loss_fn = nn.HuberLoss(delta=cfg.huber_delta)
        else:
            self.critic_loss_fn = nn.MSELoss()

        assert self.cfg.critic_input in ("state", "obs")
        if self.cfg.critic_input == "state" and self.agent_spec.state_spec is not None:
            self.critic_in_keys = [("agents", "state")]
            self.critic_out_keys = ["state_value"]
            if cfg.get("rnn", None):
                self.critic_in_keys.extend([
                    f"{self.agent_spec.name}.critic_rnn_state", "is_init"
                ])
                self.critic_out_keys.append(f"{self.agent_spec.name}.critic_rnn_state")
            reward_spec = self.agent_spec.reward_spec
            reward_spec = reward_spec.expand(self.agent_spec.n, *reward_spec.shape)
            critic = make_critic(cfg, self.agent_spec.state_spec, reward_spec, centralized=True)
            self.critic = TensorDictModule(
                critic,
                in_keys=self.critic_in_keys,
                out_keys=self.critic_out_keys,
            ).to(self.device)
            self.value_func = self.critic
        else:
            self.critic_in_keys = [self.obs_name]
            self.critic_out_keys = ["state_value"]
            if cfg.get("rnn", None):
                self.critic_in_keys.extend([
                    f"{self.agent_spec.name}.critic_rnn_state", "is_init"
                ])
                self.critic_out_keys.append(f"{self.agent_spec.name}.critic_rnn_state")
            critic = make_critic(cfg, self.agent_spec.observation_spec, self.agent_spec.reward_spec, centralized=False)
            self.critic = TensorDictModule(
                critic,
                in_keys=self.critic_in_keys,
                out_keys=self.critic_out_keys,
            ).to(self.device)
            self.value_func = vmap(self.critic, in_dims=1, out_dims=1)

        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = cfg.lr_scheduler
        if scheduler is not None:
            scheduler = eval(scheduler)
            self.critic_opt_scheduler: LR_SCHEDULER = scheduler(
                self.critic_opt, **cfg.lr_scheduler_kwargs
            )

        if hasattr(cfg, "value_norm") and cfg.value_norm is not None:
            # The original MAPPO implementation uses ValueNorm1 with a very large beta,
            # and normalizes advantages at batch level.
            # Tianshou (https://github.com/thu-ml/tianshou) uses ValueNorm2 with subtract_mean=False,
            # and normalizes advantages at mini-batch level.
            # Empirically the performance is similar on most of the tasks.
            cls = getattr(valuenorm, cfg.value_norm["class"])
            self.value_normalizer: valuenorm.Normalizer = cls(
                # input_shape=self.agent_spec.reward_spec.shape[-2:],
                input_shape=self.agent_spec.reward_spec.shape[-1:],
                **cfg.value_norm["kwargs"],
            ).to(self.device)

    def _ensure_prev_action(self, actor_input: TensorDict):
        if not getattr(self, "actor_has_prev_action_conditioning", False):
            return actor_input
        if self.prev_act_name in actor_input.keys(True):
            return actor_input

        prev_action = None
        expected_shape = (*actor_input.batch_size, self.agent_spec.n, self.act_dim)
        if self._cached_prev_action is not None:
            cache = self._cached_prev_action
            if tuple(cache.shape) == expected_shape:
                prev_action = cache.clone()

        if prev_action is None:
            prev_action = torch.zeros(
                *expected_shape,
                device=self.device,
                dtype=torch.float32,
            )

        if "is_init" in actor_input.keys():
            is_init = actor_input["is_init"].to(device=prev_action.device, dtype=torch.bool)
            while is_init.ndim < prev_action.ndim:
                is_init = is_init.unsqueeze(-1)
            prev_action = torch.where(is_init, torch.zeros_like(prev_action), prev_action)

        actor_input[self.prev_act_name] = prev_action
        return actor_input

    def _cache_prev_action(self, action: Optional[torch.Tensor]):
        if not getattr(self, "actor_has_prev_action_conditioning", False):
            return
        if action is None:
            self._cached_prev_action = None
            return
        self._cached_prev_action = action.detach().clone()

    def reset_prev_action_cache(self):
        self._cached_prev_action = None

    def set_actor_frozen(self, frozen: bool):
        self.actor_frozen = bool(frozen)

    def set_actor_lr(self, lr: float):
        lr = float(lr)
        for group in self.actor_opt.param_groups:
            group["lr"] = lr

    def set_critic_lr(self, lr: float):
        lr = float(lr)
        for group in self.critic_opt.param_groups:
            group["lr"] = lr

    def set_kl_reg(self, target_kl: Optional[float], coef: float = 0.0, hard_stop: bool = False):
        self.ppo_target_kl = None if target_kl is None else float(target_kl)
        self.ppo_kl_coef = float(coef)
        self.ppo_kl_hard_stop = bool(hard_stop)

    def capture_anchor_actor(self):
        anchor_td = self.actor_params.to(self.device).to_tensordict().clone()
        self.anchor_actor_params = TensorDictParams(anchor_td)

    def clear_anchor_actor(self):
        self.anchor_actor_params = None

    def set_anchor_loss(self, coef: float = 0.0, loss_type: str = "mse"):
        self.anchor_loss_coef = float(coef)
        self.anchor_loss_type = str(loss_type)

    def set_expert_kl_loss(
        self,
        coef: float = 0.0,
        loss_type: Optional[str] = None,
        max_loss: Optional[float] = None,
    ):
        self.expert_kl_coef = float(coef)
        if loss_type is not None:
            self.expert_kl_loss_type = str(loss_type)
        if max_loss is not None:
            self.expert_kl_max_loss = float(max_loss)

    def _run_actor(
        self,
        actor_input: TensorDict,
        actor_params,
        deterministic: bool = False,
        eval_action: bool = False,
    ):
        randomness = "different" if (not deterministic and not eval_action) else "error"
        if hasattr(self, "minibatch_seq_len"):
            agent_dim = len(actor_input.batch_size) - 1
            if self.cfg.share_actor:
                return vmap(
                    self.actor,
                    in_dims=(agent_dim, None),
                    out_dims=agent_dim,
                    randomness=randomness,
                )(
                    actor_input, actor_params, deterministic=deterministic, eval_action=eval_action
                )
            return vmap(
                self.actor,
                in_dims=(agent_dim, 0),
                out_dims=agent_dim,
                randomness=randomness,
            )(
                actor_input, actor_params, deterministic=deterministic, eval_action=eval_action
            )
        if self.cfg.share_actor:
            return self.actor(
                actor_input,
                actor_params,
                deterministic=deterministic,
                eval_action=eval_action,
            )
        return vmap(
            self.actor,
            in_dims=(1, 0),
            out_dims=1,
            randomness=randomness,
        )(
            actor_input,
            actor_params,
            deterministic=deterministic,
            eval_action=eval_action,
        )

    def value_op(self, tensordict: TensorDict) -> TensorDict:
        critic_input = tensordict.select(*self.critic_in_keys, strict=False)
        if "is_init" in critic_input.keys():
            critic_input["is_init"] = expand_right(
            critic_input["is_init"], (*critic_input.batch_size, self.agent_spec.n, 1)
        )
        if self.cfg.critic_input == "obs":
            critic_input.batch_size = [*critic_input.batch_size, self.agent_spec.n]
        elif "is_init" in critic_input.keys() and critic_input["is_init"].shape[-1] != 1:
            critic_input["is_init"] = critic_input["is_init"].all(-1, keepdim=True)
        tensordict = self.value_func(critic_input)
        return tensordict

    def __call__(self, tensordict: TensorDict, deterministic: bool = False):
        actor_input = tensordict.select(*self.actor_in_keys, strict=False)
        actor_input = self._ensure_prev_action(actor_input)
        if "is_init" in actor_input.keys():
            actor_input["is_init"] = expand_right(
            actor_input["is_init"], (*actor_input.batch_size, self.agent_spec.n, 1)
        )
        actor_input.batch_size = [*actor_input.batch_size, self.agent_spec.n] # [env_num, drone_num]
        actor_output = self._run_actor(
            actor_input,
            self.actor_params,
            deterministic=deterministic,
            eval_action=False,
        )

        self._cache_prev_action(actor_output.get(self.act_name, None))
        tensordict.update(actor_output)
        tensordict.update(self.value_op(tensordict))
        return tensordict

    def update_TP(self, batch: TensorDict) -> Dict[str, Any]:
        batch_size = batch['TP_groundtruth'].shape[0]
        TP_input = batch['TP_input']
        TP_groundtruth = batch['TP_groundtruth'] # range: (-1, 1)
        
        # TP_output = self.TP_net(TP_input).reshape(-1, future_step, pos_dim) # range: (-1, 1)
        # loss = torch.mean(torch.norm(TP_output - TP_groundtruth, dim=-1).mean(1), dim=0)
        
        TP_outsput = self.TP_net(TP_input)
        loss = self.TP_criterion(TP_outsput, TP_groundtruth.reshape(batch_size, -1))
                
        self.TP_optimizer.zero_grad()
        loss.backward()
        self.TP_optimizer.step()
        
        return {
            "TP_loss": loss.item()
        }

    def update_actor(self, batch: TensorDict) -> Dict[str, Any]:
        if self.actor_frozen:
            return {
                "policy_loss": 0.0,
                "actor_grad_norm": 0.0,
                "entropy": float(self._get_actor_entropy_bonus().item()) if self._get_actor_entropy_bonus() is not None else 0.0,
                "ESS": 1.0,
                "approx_kl": 0.0,
                "kl_penalty": 0.0,
                "anchor_loss": 0.0,
                "expert_kl_loss": 0.0,
                "expert_action_mse": 0.0,
                "expert_kl_effective_coef": float(self.expert_kl_coef),
                "expert_action_mse_ema": float(self.expert_kl_mse_ema or 0.0),
                "actor_update_skipped": 1.0,
            }
        advantages = batch["advantages"]
        actor_input = batch.select(*self.actor_in_keys, strict=False).clone()
        if self.raw_act_name in batch.keys(True):
            actor_input[self.act_name] = batch[self.raw_act_name]
        actor_input = self._ensure_prev_action(actor_input)
            
        if "is_init" in actor_input.keys():
            actor_input["is_init"] = expand_right(
            actor_input["is_init"], (*actor_input.batch_size, self.agent_spec.n, 1)
        )
        actor_input.batch_size = [*actor_input.batch_size, self.agent_spec.n]

        log_probs_old = batch[self.act_logps_name]
        actor_output = self._run_actor(
            actor_input,
            self.actor_params,
            deterministic=False,
            eval_action=True,
        )

        log_probs_new = actor_output[self.act_logps_name]

        ratio = torch.exp(log_probs_new - log_probs_old)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * advantages
        )
        policy_loss = - torch.mean(torch.min(surr1, surr2) * self.act_dim)
        approx_kl = torch.mean(log_probs_old - log_probs_new)
        kl_penalty = torch.zeros((), device=policy_loss.device)
        if self.ppo_target_kl is not None and self.ppo_kl_coef > 0.0:
            kl_penalty = torch.relu(approx_kl - float(self.ppo_target_kl))
        entropy_bonus = self._get_actor_entropy_bonus()
        if entropy_bonus is None:
            entropy_bonus = -torch.mean(-log_probs_new.detach())
        entropy_loss = -entropy_bonus
        anchor_loss = torch.zeros((), device=policy_loss.device)
        if self.anchor_actor_params is not None and self.anchor_loss_coef > 0.0:
            current_pred = self._run_actor(
                actor_input,
                self.actor_params,
                deterministic=True,
                eval_action=False,
            )[self.act_name]
            with torch.no_grad():
                anchor_pred = self._run_actor(
                    actor_input,
                    self.anchor_actor_params,
                    deterministic=True,
                    eval_action=False,
                )[self.act_name]
            if self.anchor_loss_type == "l1":
                anchor_loss = F.l1_loss(current_pred, anchor_pred)
            else:
                anchor_loss = F.mse_loss(current_pred, anchor_pred)

        expert_kl_loss = torch.zeros((), device=policy_loss.device)
        expert_action_mse = torch.zeros((), device=policy_loss.device)
        expert_kl_effective_coef = float(self.expert_kl_coef)
        if self.expert_kl_enabled and self.expert_kl_coef > 0.0 and self.expert_act_name in batch.keys(True):
            expert_action = batch[self.expert_act_name].to(device=policy_loss.device, dtype=log_probs_new.dtype)
            expert_loss_type = self.expert_kl_loss_type.lower()
            if expert_loss_type in {"mse", "fixed_std_nll", "fixed_std_kl", "mse_fixed_std"}:
                current_pred = self._run_actor(
                    actor_input,
                    self.actor_params,
                    deterministic=True,
                    eval_action=False,
                )[self.act_name]
                sq_error = (current_pred - expert_action).pow(2)
                expert_action_mse_loss = sq_error.mean()
                if expert_loss_type == "mse":
                    expert_kl_loss = expert_action_mse_loss
                else:
                    # Use a fixed Gaussian std for the expert anchor. This keeps
                    # the expert gradient alive even when the actor's own std is
                    # intentionally very small for low-exploration rollouts.
                    fixed_var = max(self.expert_kl_fixed_std, 1e-6) ** 2
                    expert_kl_loss = 0.5 * expert_action_mse_loss / fixed_var
                expert_action_mse = expert_action_mse_loss.detach()
                mse_value = float(expert_action_mse.item())
                if self.expert_kl_mse_ema is None:
                    self.expert_kl_mse_ema = mse_value
                else:
                    beta = min(max(float(self.expert_kl_mse_boost_ema_beta), 0.0), 0.9999)
                    self.expert_kl_mse_ema = beta * self.expert_kl_mse_ema + (1.0 - beta) * mse_value
                if (
                    self.expert_kl_mse_boost_enabled
                    and self.expert_kl_mse_ema > self.expert_kl_mse_boost_threshold
                ):
                    expert_kl_effective_coef = max(
                        expert_kl_effective_coef,
                        float(self.expert_kl_mse_boost_coef),
                    )
            else:
                expert_actor_input = actor_input.clone()
                expert_actor_input[self.act_name] = expert_action
                expert_output = self._run_actor(
                    expert_actor_input,
                    self.actor_params,
                    deterministic=False,
                    eval_action=True,
                )
                expert_log_probs = expert_output[self.act_logps_name]
                expert_kl_loss = -expert_log_probs.mean()
                if self.expert_kl_max_loss is not None:
                    expert_kl_loss = torch.clamp(expert_kl_loss, max=float(self.expert_kl_max_loss))
                with torch.no_grad():
                    current_pred = self._run_actor(
                        actor_input,
                        self.actor_params,
                        deterministic=True,
                        eval_action=False,
                    )[self.act_name]
                    expert_action_mse = F.mse_loss(current_pred, expert_action)

        total_actor_loss = (
            policy_loss
            + entropy_loss * self.entropy_coef
            + kl_penalty * self.ppo_kl_coef
            + anchor_loss * self.anchor_loss_coef
            + expert_kl_loss * expert_kl_effective_coef
        )
        if self.ppo_target_kl is not None and self.ppo_kl_hard_stop and float(approx_kl.item()) > float(self.ppo_target_kl):
            return {
                "policy_loss": policy_loss.item(),
                "actor_grad_norm": 0.0,
                "entropy": entropy_bonus.item(),
                "ESS": ((2 * ratio.logsumexp(0) - (2 * ratio).logsumexp(0)).exp().mean() / ratio.shape[0]).item(),
                "approx_kl": approx_kl.item(),
                "kl_penalty": kl_penalty.item(),
                "anchor_loss": anchor_loss.item(),
                "expert_kl_loss": expert_kl_loss.item(),
                "expert_action_mse": expert_action_mse.item(),
                "expert_kl_effective_coef": float(expert_kl_effective_coef),
                "expert_action_mse_ema": float(self.expert_kl_mse_ema or 0.0),
                "actor_update_skipped": 1.0,
            }

        self.actor_opt.zero_grad()
        total_actor_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor_opt.param_groups[0]["params"], self.cfg.max_grad_norm
        )
        
        self.actor_opt.step()

        ess = (2 * ratio.logsumexp(0) - (2 * ratio).logsumexp(0)).exp().mean() / ratio.shape[0]
        return {
            "policy_loss": policy_loss.item(),
            "actor_grad_norm": grad_norm.item(),
            "entropy": entropy_bonus.item(),
            "ESS": ess.item(),
            "approx_kl": approx_kl.item(),
            "kl_penalty": kl_penalty.item(),
            "anchor_loss": anchor_loss.item(),
            "expert_kl_loss": expert_kl_loss.item(),
            "expert_action_mse": expert_action_mse.item(),
            "expert_kl_effective_coef": float(expert_kl_effective_coef),
            "expert_action_mse_ema": float(self.expert_kl_mse_ema or 0.0),
            "actor_update_skipped": 0.0,
        }

    def update_actor_bc(
        self,
        batch: TensorDict,
        entropy_bonus_coef: float = 0.0,
        action_mse_coef: float = 1.0,
        log_prob_coef: float = 0.0,
        aux_vel_cmd_coef: float = 0.0,
        aux_waypoint_coef: float = 0.0,
        aux_target_pos_coef: float = 0.0,
        aux_forward_dir_coef: float = 0.0,
        aux_assignment_coef: float = 0.0,
        aux_trap_coef: float = 0.0,
        action_dim_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        actor_input = batch.select(*self.actor_in_keys, strict=False).clone()
        target_action = batch.get(self.raw_act_name, batch[self.act_name])
        sample_weight = batch.get("bc_weight", None)
        actor_input[self.act_name] = target_action
        actor_input = self._ensure_prev_action(actor_input)

        if "is_init" in actor_input.keys():
            actor_input["is_init"] = expand_right(
                actor_input["is_init"], (*actor_input.batch_size, self.agent_spec.n, 1)
            )
        actor_input.batch_size = [*actor_input.batch_size, self.agent_spec.n]

        need_logprob = (float(log_prob_coef) != 0.0) or (float(entropy_bonus_coef) != 0.0)

        if hasattr(self, "minibatch_seq_len"):
            agent_dim = len(actor_input.batch_size) - 1
            if self.cfg.share_actor:
                actor_output = (
                    vmap(self.actor, in_dims=(agent_dim, None), out_dims=agent_dim)(
                        actor_input, self.actor_params, eval_action=True
                    )
                    if need_logprob else None
                )
                pred_output = vmap(self.actor, in_dims=(agent_dim, None), out_dims=agent_dim)(
                    actor_input, self.actor_params, deterministic=True
                )
            else:
                actor_output = (
                    vmap(self.actor, in_dims=(agent_dim, 0), out_dims=agent_dim)(
                        actor_input, self.actor_params, eval_action=True
                    )
                    if need_logprob else None
                )
                pred_output = vmap(self.actor, in_dims=(agent_dim, 0), out_dims=agent_dim)(
                    actor_input, self.actor_params, deterministic=True
                )
        else:
            if self.cfg.share_actor:
                actor_output = (
                    self.actor(actor_input, self.actor_params, eval_action=True)
                    if need_logprob else None
                )
                pred_output = self.actor(actor_input, self.actor_params, deterministic=True)
            else:
                actor_output = (
                    vmap(self.actor, in_dims=(1, 0), out_dims=1)(
                        actor_input, self.actor_params, eval_action=True
                    )
                    if need_logprob else None
                )
                pred_output = vmap(self.actor, in_dims=(1, 0), out_dims=1)(
                    actor_input, self.actor_params, deterministic=True
                )

        pred_action = pred_output[self.act_name]

        def weighted_mean(value: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
            if weight is None:
                return value.mean()
            w = weight.to(device=value.device, dtype=value.dtype)
            while w.ndim < value.ndim:
                w = w.unsqueeze(-1)
            return (value * w).sum() / w.expand_as(value).sum().clamp_min(1e-8)

        sq_err_raw = (pred_action - target_action).pow(2)
        # Per-dimension action weighting (e.g. upweight yaw/thrust)
        if action_dim_weights is not None:
            dim_w = action_dim_weights.to(device=sq_err_raw.device, dtype=sq_err_raw.dtype)
            # Normalize so mean weight = 1 (preserves loss scale)
            dim_w = dim_w / dim_w.mean().clamp_min(1e-8)
            sq_err = sq_err_raw * dim_w
        else:
            sq_err = sq_err_raw
        action_mse = sq_err_raw.mean()
        if sample_weight is not None:
            weight = sample_weight.to(device=pred_action.device, dtype=pred_action.dtype)
            weighted_action_mse = weighted_mean(sq_err, weight)
            bc_loss = weighted_action_mse * float(action_mse_coef)
            weight_mean = float(weight.mean().item())
        else:
            weighted_action_mse = action_mse
            bc_loss = action_mse * float(action_mse_coef)
            weight_mean = 1.0

        aux_vel_cmd_loss = torch.tensor(0.0, device=pred_action.device)
        aux_waypoint_loss = torch.tensor(0.0, device=pred_action.device)
        aux_target_pos_loss = torch.tensor(0.0, device=pred_action.device)
        aux_forward_dir_loss = torch.tensor(0.0, device=pred_action.device)
        aux_assignment_loss = torch.tensor(0.0, device=pred_action.device)
        aux_trap_loss = torch.tensor(0.0, device=pred_action.device)
        aux_assignment_acc = float("nan")
        aux_trap_acc = float("nan")

        if ("expert_aux", "vel_cmd") in batch.keys(True) and ("bc_aux_pred", "vel_cmd") in pred_output.keys(True):
            target_vel_cmd = batch["expert_aux", "vel_cmd"].to(device=pred_action.device, dtype=pred_action.dtype)
            pred_vel_cmd = pred_output["bc_aux_pred", "vel_cmd"]
            aux_vel_cmd_loss = weighted_mean((pred_vel_cmd - target_vel_cmd).pow(2), sample_weight)
            bc_loss = bc_loss + float(aux_vel_cmd_coef) * aux_vel_cmd_loss

        if ("expert_aux", "waypoint") in batch.keys(True) and ("bc_aux_pred", "waypoint") in pred_output.keys(True):
            target_waypoint = batch["expert_aux", "waypoint"].to(device=pred_action.device, dtype=pred_action.dtype)
            pred_waypoint = pred_output["bc_aux_pred", "waypoint"]
            aux_waypoint_loss = weighted_mean((pred_waypoint - target_waypoint).pow(2), sample_weight)
            bc_loss = bc_loss + float(aux_waypoint_coef) * aux_waypoint_loss

        if ("expert_aux", "target_pos_pred") in batch.keys(True) and ("bc_aux_pred", "target_pos_pred") in pred_output.keys(True):
            target_pos_pred = batch["expert_aux", "target_pos_pred"].to(device=pred_action.device, dtype=pred_action.dtype)
            pred_target_pos = pred_output["bc_aux_pred", "target_pos_pred"]
            while target_pos_pred.ndim < pred_target_pos.ndim:
                target_pos_pred = target_pos_pred.unsqueeze(-2)
            aux_target_pos_loss = weighted_mean((pred_target_pos - target_pos_pred).pow(2), sample_weight)
            bc_loss = bc_loss + float(aux_target_pos_coef) * aux_target_pos_loss

        if ("expert_aux", "forward_dir") in batch.keys(True) and ("bc_aux_pred", "forward_dir") in pred_output.keys(True):
            target_forward_dir = batch["expert_aux", "forward_dir"].to(device=pred_action.device, dtype=pred_action.dtype)
            pred_forward_dir = pred_output["bc_aux_pred", "forward_dir"]
            while target_forward_dir.ndim < pred_forward_dir.ndim:
                target_forward_dir = target_forward_dir.unsqueeze(-2)
            aux_forward_dir_loss = weighted_mean((pred_forward_dir - target_forward_dir).pow(2), sample_weight)
            bc_loss = bc_loss + float(aux_forward_dir_coef) * aux_forward_dir_loss

        if ("expert_aux", "assignment") in batch.keys(True) and ("bc_aux_pred", "assignment_logits") in pred_output.keys(True):
            target_assignment = batch["expert_aux", "assignment"].to(device=pred_action.device, dtype=torch.long)
            pred_assignment_logits = pred_output["bc_aux_pred", "assignment_logits"]
            ce = F.cross_entropy(
                pred_assignment_logits.reshape(-1, pred_assignment_logits.shape[-1]),
                target_assignment.reshape(-1),
                reduction="none",
            ).reshape_as(target_assignment)
            aux_assignment_loss = weighted_mean(ce, sample_weight)
            bc_loss = bc_loss + float(aux_assignment_coef) * aux_assignment_loss
            pred_assignment = pred_assignment_logits.argmax(dim=-1)
            aux_assignment_acc = float((pred_assignment == target_assignment).float().mean().item())

        if ("expert_aux", "trap_mode") in batch.keys(True) and ("bc_aux_pred", "trap_logit") in pred_output.keys(True):
            target_trap = batch["expert_aux", "trap_mode"].to(device=pred_action.device, dtype=pred_action.dtype)
            pred_trap_logit = pred_output["bc_aux_pred", "trap_logit"]
            while target_trap.ndim < pred_trap_logit.ndim:
                target_trap = target_trap.unsqueeze(-1)
            target_trap = target_trap.expand_as(pred_trap_logit)
            bce = F.binary_cross_entropy_with_logits(pred_trap_logit, target_trap, reduction="none")
            aux_trap_loss = weighted_mean(bce, sample_weight)
            bc_loss = bc_loss + float(aux_trap_coef) * aux_trap_loss
            pred_trap = (torch.sigmoid(pred_trap_logit) > 0.5).to(target_trap.dtype)
            aux_trap_acc = float((pred_trap == target_trap).float().mean().item())

        action_log_prob_mean = float("nan")
        log_prob_loss_value = 0.0
        entropy_value = float("nan")
        if actor_output is not None:
            action_log_probs = actor_output[self.act_logps_name]
            action_log_prob_mean = float(action_log_probs.mean().item())
            log_prob_loss = -(action_log_probs.mean() * self.act_dim)
            bc_loss = bc_loss + float(log_prob_coef) * log_prob_loss
            log_prob_loss_value = float(log_prob_loss.item())

            if not self.cfg.actor.tanh:
                dist_entropy = actor_output[f"{self.agent_spec.name}.action_entropy"]
                entropy = dist_entropy.mean()
                bc_loss = bc_loss - float(entropy_bonus_coef) * entropy
                entropy_value = float(entropy.item())
            else:
                # For tanh actor: use -log_prob as entropy proxy so that
                # entropy_bonus_coef can keep log_std from collapsing.
                neg_logp_entropy = -action_log_probs.mean()
                if float(entropy_bonus_coef) != 0.0:
                    bc_loss = bc_loss - float(entropy_bonus_coef) * neg_logp_entropy
                entropy_value = float(neg_logp_entropy.item())

        log_std_values = []
        for name, param in self.actor_params.named_parameters():
            if "log_std" in name:
                log_std_values.append(param.detach().reshape(-1))
        if log_std_values:
            log_std_tensor = torch.cat(log_std_values)
            log_std_mean = float(log_std_tensor.mean().item())
            log_std_min = float(log_std_tensor.min().item())
            log_std_max = float(log_std_tensor.max().item())
        else:
            log_std_mean = float("nan")
            log_std_min = float("nan")
            log_std_max = float("nan")

        self.actor_opt.zero_grad()
        bc_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor_opt.param_groups[0]["params"], self.cfg.max_grad_norm
        )
        self.actor_opt.step()

        # Per-dimension MSE diagnostics (unweighted, for monitoring)
        per_dim_mse = {}
        if sq_err_raw.shape[-1] >= 4:
            dim_names = ["roll", "pitch", "yaw", "thrust"]
            for di, dn in enumerate(dim_names):
                per_dim_mse[f"bc_mse_{dn}"] = float(sq_err_raw[..., di].mean().item())

        result = {
            "bc_loss": float(bc_loss.item()),
            "bc_action_mse_loss": float(weighted_action_mse.item() * float(action_mse_coef)),
            "bc_log_prob_loss": float(log_prob_loss_value * float(log_prob_coef)),
            "bc_action_mse": float(action_mse.item()),
            "bc_weighted_action_mse": float(weighted_action_mse.item()),
            "bc_aux_vel_cmd_loss": float(aux_vel_cmd_loss.item()),
            "bc_aux_waypoint_loss": float(aux_waypoint_loss.item()),
            "bc_aux_target_pos_loss": float(aux_target_pos_loss.item()),
            "bc_aux_forward_dir_loss": float(aux_forward_dir_loss.item()),
            "bc_aux_assignment_loss": float(aux_assignment_loss.item()),
            "bc_aux_trap_loss": float(aux_trap_loss.item()),
            "bc_aux_assignment_acc": float(aux_assignment_acc),
            "bc_aux_trap_acc": float(aux_trap_acc),
            "bc_log_prob": float(action_log_prob_mean),
            "bc_entropy": float(entropy_value),
            "bc_grad_norm": float(grad_norm.item()),
            "bc_pred_action_norm": float(pred_action.norm(dim=-1).mean().item()),
            "bc_target_action_norm": float(target_action.norm(dim=-1).mean().item()),
            "bc_weight_mean": float(weight_mean),
            "bc_log_std_mean": float(log_std_mean),
            "bc_log_std_min": float(log_std_min),
            "bc_log_std_max": float(log_std_max),
        }
        result.update(per_dim_mse)
        return result

    def update_critic(self, batch: TensorDict) -> Dict[str, Any]:
        critic_input = batch.select(*self.critic_in_keys)
        values = self.value_op(critic_input)["state_value"]
        b_values = batch["state_value"]
        b_returns = batch["returns"]
        assert values.shape == b_values.shape == b_returns.shape
        value_pred_clipped = b_values + (values - b_values).clamp(
            -self.clip_param, self.clip_param
        )

        value_loss_clipped = self.critic_loss_fn(b_returns, value_pred_clipped)
        value_loss_original = self.critic_loss_fn(b_returns, values)

        value_loss = torch.max(value_loss_original, value_loss_clipped)

        value_loss.backward()  # do not multiply weights here
        grad_norm = nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.cfg.max_grad_norm
        )
        self.critic_opt.step()
        self.critic_opt.zero_grad(set_to_none=True)
        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        return {
            "value_loss": value_loss.mean(),
            "critic_grad_norm": grad_norm.item(),
            "explained_var": explained_var.item()
        }

    def _get_dones(self, tensordict: TensorDict):
        env_done = tensordict[("next", "done")].unsqueeze(-1)
        agent_done = tensordict.get(
            ("next", f"{self.agent_spec.name}.done"),
            env_done.expand(*env_done.shape[:-2], self.agent_spec.n, 1),
        )
        done = agent_done | env_done
        return done

    def _prepare_tp_batch(self, tensordict: TensorDict):
        TP_groundtruth = tensordict["next"]["agents"]["TP"]["TP_groundtruth"]
        TP_input = tensordict["next"]["agents"]["TP"]["TP_input"]
        TP_done = tensordict["next"]["agents"]["TP"]["TP_done"]
        env_done = tensordict["next"]["done"]

        window_size = self.TP_net.future_predcition_step
        window_step = self.TP_net.window_step
        windows = (
            TP_groundtruth.unfold(dimension=1, size=window_size + 1, step=window_step)
            .transpose(2, 3)[:, :, 1:]
        )
        _, window_count, future_step, pos_dim = windows.shape

        timeout_valid = TP_done.reshape(*TP_done.shape[:2], -1).bool().all(dim=-1)
        timeout_valid = timeout_valid[:, :window_count]

        env_done = env_done.reshape(*env_done.shape[:2], -1).bool().any(dim=-1)
        continuity_windows = env_done.unfold(
            dimension=1, size=window_size, step=window_step
        )[:, :window_count]
        continuity_valid = ~continuity_windows.any(dim=-1)

        valid_windows = timeout_valid & continuity_valid
        valid_counts = valid_windows.sum(dim=-1)

        if not self._tp_debug_emitted or valid_counts.min() != valid_counts.max():
            logging.warning(
                "TP window debug: TP_groundtruth=%s TP_input=%s TP_done=%s env_done=%s "
                "windows=%s valid_windows(total=%d,min=%d,max=%d,sample=%s)",
                tuple(TP_groundtruth.shape),
                tuple(TP_input.shape),
                tuple(TP_done.shape),
                tuple(env_done.shape),
                tuple(windows.shape),
                int(valid_windows.sum().item()),
                int(valid_counts.min().item()),
                int(valid_counts.max().item()),
                valid_counts[:8].tolist(),
            )
            self._tp_debug_emitted = True

        flat_mask = valid_windows.reshape(-1)
        flat_inputs = TP_input[:, :window_count].reshape(-1, *TP_input.shape[2:])
        flat_windows = windows.reshape(-1, future_step, pos_dim)

        selected_inputs = flat_inputs[flat_mask]
        selected_windows = flat_windows[flat_mask]

        debug_info = {
            "TP_valid_windows": float(selected_windows.shape[0]),
            "TP_valid_windows_min": float(valid_counts.min().item()),
            "TP_valid_windows_max": float(valid_counts.max().item()),
        }
        return selected_inputs, selected_windows, debug_info

    def _get_actor_log_std_stats(self) -> Dict[str, float]:
        try:
            log_std = self.actor_params["module"]["act_dist"]["log_std"]
        except KeyError:
            return {}

        log_std = log_std.clamp(
            self.cfg.actor.get("log_std_min", -5.0),
            self.cfg.actor.get("log_std_max", 1.0),
        )

        return {
            "actor_log_std_mean": log_std.mean().item(),
            "actor_log_std_min": log_std.min().item(),
            "actor_log_std_max": log_std.max().item(),
        }

    def _get_actor_entropy_bonus(self) -> Optional[torch.Tensor]:
        try:
            log_std = self.actor_params["module"]["act_dist"]["log_std"]
        except KeyError:
            return None

        log_std = log_std.clamp(
            self.cfg.actor.get("log_std_min", -5.0),
            self.cfg.actor.get("log_std_max", 1.0),
        )
        entropy_per_dim = log_std + (0.5 * math.log(2.0 * math.pi * math.e))
        if entropy_per_dim.ndim == 1:
            return entropy_per_dim.sum()
        return entropy_per_dim.sum(dim=-1).mean()

    def train_op(self, tensordict: TensorDict):
        if self.use_TP_net and hasattr(self, "TP_net") and self.TP_net is not None:
            self.TP_net.train()
        if hasattr(self, "actor"):
            self.actor.train()
        if hasattr(self, "critic"):
            self.critic.train()

        tensordict = tensordict.select(*self.train_in_keys, strict=False)
        next_tensordict = tensordict["next"][:, -1]
        with torch.no_grad():
            value_output = self.value_op(next_tensordict)

        rewards = tensordict.get(("next", *self.reward_name))
        if rewards.shape[-1] != 1:
            rewards = rewards.sum(-1, keepdim=True)

        values = tensordict["state_value"]
        next_value = value_output["state_value"].squeeze(0)

        if hasattr(self, "value_normalizer"):
            values = self.value_normalizer.denormalize(values)
            next_value = self.value_normalizer.denormalize(next_value)

        dones = self._get_dones(tensordict)

        tensordict["advantages"], tensordict["returns"] = compute_gae(
            rewards,
            dones,
            values,
            next_value,
            gamma=self.gae_gamma,
            lmbda=self.gae_lambda,
        )

        advantages_mean = tensordict["advantages"].mean()
        advantages_std = tensordict["advantages"].std()
        if self.normalize_advantages:
            tensordict["advantages"] = (tensordict["advantages"] - advantages_mean) / (
                advantages_std + 1e-8
            )

        if hasattr(self, "value_normalizer"):
            self.value_normalizer.update(tensordict["returns"])
            tensordict["returns"] = self.value_normalizer.normalize(
                tensordict["returns"]
            )

        train_info = []
        TP_info = []
        tp_debug_info = {}
        
        if self.use_TP_net:
            selected_inputs, selected_windows, tp_debug_info = self._prepare_tp_batch(
                tensordict
            )
            if selected_windows.shape[0] > 0:
                TP_tensordict = TensorDict(
                    {
                        "TP_input": selected_inputs,
                        "TP_groundtruth": selected_windows,
                    },
                    batch_size=[selected_windows.shape[0]],
                )
                for _ in range(self.TP_epoch):
                    dataset = make_dataset_naive(
                        TP_tensordict,
                        int(self.cfg.num_minibatches),
                        1,
                    )
                    for minibatch in dataset:
                        TP_info.append(
                            TensorDict(
                                {
                                    **self.update_TP(minibatch),
                                },
                                batch_size=[],
                            )
                        )
            else:
                logging.warning("TP update skipped: no valid prediction windows in batch.")
            
        # RL_update
        for ppo_epoch in range(self.ppo_epoch):
            dataset = make_dataset_naive(
                tensordict,
                int(self.cfg.num_minibatches),
                self.minibatch_seq_len if hasattr(self, "minibatch_seq_len") else 1,
            )            
            for minibatch in dataset:
                train_info.append(
                    TensorDict(
                        {
                            **self.update_actor(minibatch),
                            **self.update_critic(minibatch),
                        },
                        batch_size=[],
                    )
                )
        
        train_info = {k: v.mean().item() for k, v in torch.stack(train_info).items()}
        if self.use_TP_net:
            if TP_info:
                TP_info = {k: v.mean().item() for k, v in torch.stack(TP_info).items()}
                train_info.update(TP_info)
            else:
                train_info["TP_loss"] = 0.0
            train_info.update(tp_debug_info)
        train_info["advantages_mean"] = advantages_mean.item()
        train_info["advantages_std"] = advantages_std.item()
        if isinstance(self.agent_spec.action_spec, (BoundedTensorSpec, UnboundedTensorSpec)):
            if self.raw_act_name in tensordict.keys(True):
                raw_action = tensordict[self.raw_act_name]
                cmd_action = tensordict[self.act_name]
                train_info["action_norm"] = raw_action.norm(dim=-1).mean().item()
                train_info["cmd_norm"] = cmd_action.norm(dim=-1).mean().item()
                if self.cfg.actor.tanh:
                    # actor already outputs squashed actions in (-1,1)
                    train_info["action_squash_sat_frac"] = (
                        raw_action.abs() > 0.98
                    ).float().mean().item()
                else:
                    # actor outputs raw unbounded actions; check post-tanh saturation
                    train_info["action_squash_sat_frac"] = (
                        torch.tanh(raw_action).abs() > 0.98
                    ).float().mean().item()
                train_info["cmd_saturation_frac"] = (
                    cmd_action.abs() > 0.98
                ).float().mean().item()
            else:
                train_info["action_norm"] = tensordict[self.act_name].norm(dim=-1).mean().item()
                train_info["cmd_norm"] = train_info["action_norm"]
            train_info.update(self._get_actor_log_std_stats())
        if hasattr(self, "value_normalizer"):
            train_info["value_running_mean"] = self.value_normalizer.running_mean.mean().item()
        
        self.n_updates += 1
        # step LR schedulers if configured
        if hasattr(self, "critic_opt_scheduler"):
            self.critic_opt_scheduler.step()
        if hasattr(self, "actor_opt_scheduler") and not self.actor_frozen:
            self.actor_opt_scheduler.step()
        return {f"{self.agent_spec.name}/{k}": v for k, v in train_info.items()}

    def state_dict(self):
        state_dict = {
            "TP": self.TP_net.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_params": self.actor_params,
            "value_normalizer": self.value_normalizer.state_dict(),
            "actor_has_bc_aux": bool(getattr(self, "actor_has_bc_aux", False)),
            "actor_has_prev_action_conditioning": bool(
                getattr(self, "actor_has_prev_action_conditioning", False)
            ),
            "actor_prev_action_condition_hidden_dim": int(
                self.cfg.actor.prev_action_conditioning.hidden_dim
            )
            if getattr(self, "actor_has_prev_action_conditioning", False)
            else 0,
            "actor_has_rnn": bool(self.cfg.actor.get("rnn", None) is not None),
            "actor_rnn_hidden_size": int(self.cfg.actor.rnn.kwargs.hidden_size)
            if self.cfg.actor.get("rnn", None) is not None
            else 0,
            # Optimizer states for seamless checkpoint resume
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }
        return state_dict

    def set_entropy_coef(self, value: float):
        self.entropy_coef = float(value)

    def set_actor_log_std(self, value: float):
        try:
            log_std = self.actor_params["module"]["act_dist"]["log_std"]
        except KeyError:
            return None
        value = float(value)
        log_std_min = float(self.cfg.actor.get("log_std_min", -5.0))
        log_std_max = float(self.cfg.actor.get("log_std_max", 1.0))
        value = max(min(value, log_std_max), log_std_min)
        with torch.no_grad():
            log_std.fill_(value)
        return value
    
    def load_state_dict(self, state_dict):
        self.TP_net.load_state_dict(state_dict["TP"])
        loaded_actor_params = state_dict["actor_params"].to(self.device).to_tensordict()
        current_actor_params = self.actor_params.to_tensordict()
        current_actor_params.update(loaded_actor_params)
        self.actor_params = TensorDictParams(current_actor_params)
        self._reset_actor_optimizer()
        # Restore optimizer states if available (backward compatible with old checkpoints)
        if "actor_opt" in state_dict:
            try:
                self.actor_opt.load_state_dict(state_dict["actor_opt"])
                logging.info("Restored actor optimizer state from checkpoint.")
            except Exception as e:
                logging.warning("Failed to restore actor optimizer state: %s. Using fresh optimizer.", e)
        else:
            logging.info("No actor optimizer state in checkpoint. Using fresh optimizer.")
        self.critic.load_state_dict(state_dict["critic"])
        if "critic_opt" in state_dict:
            try:
                self.critic_opt.load_state_dict(state_dict["critic_opt"])
                logging.info("Restored critic optimizer state from checkpoint.")
            except Exception as e:
                logging.warning("Failed to restore critic optimizer state: %s. Using fresh optimizer.", e)
        else:
            logging.info("No critic optimizer state in checkpoint. Using fresh optimizer.")
        self.value_normalizer.load_state_dict(state_dict["value_normalizer"])
        self.reset_prev_action_cache()

    def load_actor_tp_state_dict(self, state_dict):
        self.TP_net.load_state_dict(state_dict["TP"])
        loaded_actor_params = state_dict["actor_params"].to(self.device).to_tensordict()
        current_actor_params = self.actor_params.to_tensordict()
        current_actor_params.update(loaded_actor_params)
        self.actor_params = TensorDictParams(current_actor_params)
        self._reset_actor_optimizer()
        self.reset_prev_action_cache()

def make_dataset_naive(
    tensordict: TensorDict, num_minibatches: int = 4, seq_len: int = 1
):
    if seq_len > 1:
        N, T = tensordict.shape
        T = (T // seq_len) * seq_len
        tensordict = tensordict[:, :T].reshape(-1, seq_len)
        perm = torch.randperm(
            (tensordict.shape[0] // num_minibatches) * num_minibatches,
            device=tensordict.device,
        ).reshape(num_minibatches, -1)
        for indices in perm:
            yield tensordict[indices]
    else:
        tensordict = tensordict.reshape(-1)
        perm = torch.randperm(
            (tensordict.shape[0] // num_minibatches) * num_minibatches,
            device=tensordict.device,
        ).reshape(num_minibatches, -1)
        for indices in perm:
            yield tensordict[indices]


from .modules.distributions import (
    DiagGaussian,
    MultiCategoricalModule,
    TanhIndependentNormalModule
)

from .modules.rnn import GRU
from .common import make_encoder

def make_ppo_actor(cfg, observation_spec: TensorSpec, action_spec: TensorSpec):
    encoder = make_encoder(cfg, observation_spec)
    feature_dim = encoder.output_shape.numel()

    if isinstance(action_spec, MultiDiscreteTensorSpec):
        act_dist = MultiCategoricalModule(
            feature_dim, 
            torch.as_tensor(action_spec.nvec.storage().float()).long()
        )
    elif isinstance(action_spec, DiscreteTensorSpec):
        act_dist = MultiCategoricalModule(feature_dim, [action_spec.space.n])
    elif isinstance(action_spec, (UnboundedTensorSpec, BoundedTensorSpec)):
        action_dim = action_spec.shape[-1]
        if cfg.tanh:
            act_dist = TanhIndependentNormalModule(
                feature_dim,
                action_dim,
                log_std_init=cfg.get("log_std_init", 0.0),
                log_std_min=cfg.get("log_std_min", -5.0),
                log_std_max=cfg.get("log_std_max", 1.0),
            )
        else:
            act_dist = DiagGaussian(
                feature_dim,
                action_dim,
                False,
                0.01,
                log_std_init=cfg.get("log_std_init", 0.0),
                log_std_min=cfg.get("log_std_min", -5.0),
                log_std_max=cfg.get("log_std_max", 1.0),
            )
    #     # act_dist = IndependentNormalModule(encoder.output_shape.numel(), action_dim, False)
    else:
        raise NotImplementedError(action_spec)

    if cfg.get("rnn", None):
        rnn_cls = {"gru": GRU}[cfg.rnn.cls.lower()]
        rnn = rnn_cls(input_size=feature_dim, **cfg.rnn.kwargs)
    else:
        rnn = None

    aux_heads = None
    action_conditioner = None
    prev_action_conditioner = None
    bc_aux_cfg = cfg.get("bc_aux", None)
    if bc_aux_cfg is not None and bc_aux_cfg.get("enabled", False):
        aux_heads = ActorAuxHeads(
            feature_dim,
            hidden_dim=int(bc_aux_cfg.get("hidden_dim", feature_dim)),
        )
        if bc_aux_cfg.get("condition_action_on_aux", False):
            hidden_dim = int(bc_aux_cfg.get("condition_hidden_dim", bc_aux_cfg.get("hidden_dim", feature_dim)))
            action_conditioner = nn.Sequential(
                nn.Linear(feature_dim + 13, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, feature_dim),
            )
            # Keep backward compatibility: older checkpoints start with zero residual here.
            nn.init.zeros_(action_conditioner[-1].weight)
            nn.init.zeros_(action_conditioner[-1].bias)

    prev_action_cfg = cfg.get("prev_action_conditioning", None)
    if prev_action_cfg is not None and prev_action_cfg.get("enabled", False):
        hidden_dim = int(prev_action_cfg.get("hidden_dim", feature_dim))
        action_dim = action_spec.shape[-1]
        prev_action_conditioner = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )
        # Preserve checkpoint compatibility: new branch starts as a no-op residual.
        nn.init.zeros_(prev_action_conditioner[-1].weight)
        nn.init.zeros_(prev_action_conditioner[-1].bias)

    return Actor(
        encoder,
        act_dist,
        rnn,
        aux_heads=aux_heads,
        action_conditioner=action_conditioner,
        prev_action_conditioner=prev_action_conditioner,
    )

def make_critic(cfg, state_spec: TensorSpec, reward_spec: TensorSpec, centralized=False):
    assert isinstance(reward_spec, (UnboundedTensorSpec, BoundedTensorSpec))
    encoder = make_encoder(cfg, state_spec)
    
    if cfg.get("rnn", None):
        rnn_cls = {"gru": GRU}[cfg.rnn.cls.lower()]
        rnn = rnn_cls(input_size=encoder.output_shape.numel(), **cfg.rnn.kwargs)
    else:
        rnn = None

    if centralized:
        v_out = nn.Linear(encoder.output_shape.numel(), reward_spec.shape[-2:].numel())
        nn.init.orthogonal_(v_out.weight, cfg.gain)
        return Critic(encoder, rnn, v_out, reward_spec.shape[-2:])
    else:
        v_out = nn.Linear(encoder.output_shape.numel(), reward_spec.shape[-1])
        nn.init.orthogonal_(v_out.weight, cfg.gain)
        return Critic(encoder, rnn, v_out, reward_spec.shape[-1:])

class TP_net(nn.Module):
    def __init__(self, input_dim, output_dim, future_predcition_step, window_step):
        super(TP_net, self).__init__()
        self.hidden_dim = 64
        self.num_layers = 1
        self.future_predcition_step = future_predcition_step # for data unfold
        self.window_step = window_step
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])  # get the output of the last layer
        # norm: -1 ~ 1
        return torch.tanh(out)


class ActorAuxHeads(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()

        def mlp(out_dim: int):
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim),
            )

        self.vel_cmd = mlp(3)
        self.waypoint = mlp(3)
        self.target_pos_pred = mlp(3)
        self.forward_dir = mlp(3)
        self.assignment_logits = mlp(3)
        self.trap_logit = mlp(1)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "vel_cmd": self.vel_cmd(features),
            "waypoint": self.waypoint(features),
            "target_pos_pred": self.target_pos_pred(features),
            "forward_dir": self.forward_dir(features),
            "assignment_logits": self.assignment_logits(features),
            "trap_logit": self.trap_logit(features),
        }

class Actor(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        act_dist: nn.Module,
        rnn: Optional[nn.Module] = None,
        aux_heads: Optional[nn.Module] = None,
        action_conditioner: Optional[nn.Module] = None,
        prev_action_conditioner: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.act_dist = act_dist
        self.rnn = rnn
        self.aux_heads = aux_heads
        self.action_conditioner = action_conditioner
        self.prev_action_conditioner = prev_action_conditioner

    def forward(
        self,
        obs: Union[torch.Tensor, TensorDict],
        prev_action: torch.Tensor = None,
        action: torch.Tensor = None,
        rnn_state=None,
        is_init=None,
        deterministic=False,
        eval_action=False
    ):
        if eval_action and action is None and prev_action is not None and self.prev_action_conditioner is None:
            action = prev_action
            prev_action = None
        actor_features = self.encoder(obs)
        if self.rnn is not None:
            actor_features, rnn_state = self.rnn(actor_features, rnn_state, is_init)    
        else:
            rnn_state = None
        aux_outputs = self.aux_heads(actor_features) if self.aux_heads is not None else {}
        dist_features = actor_features
        if self.action_conditioner is not None and aux_outputs:
            role_probs = torch.softmax(aux_outputs["assignment_logits"], dim=-1)
            trap_prob = torch.sigmoid(aux_outputs["trap_logit"])
            cond_input = torch.cat(
                [
                    actor_features,
                    aux_outputs["waypoint"],
                    aux_outputs["target_pos_pred"],
                    aux_outputs["forward_dir"],
                    role_probs,
                    trap_prob,
                ],
                dim=-1,
            )
            dist_features = actor_features + self.action_conditioner(cond_input)
        if self.prev_action_conditioner is not None and prev_action is not None:
            prev_action = torch.tanh(prev_action).to(dist_features.dtype)
            prev_action_input = torch.cat([actor_features, prev_action], dim=-1)
            dist_features = dist_features + self.prev_action_conditioner(prev_action_input)
        action_dist = self.act_dist(dist_features)

        if eval_action:
            action_log_probs = action_dist.log_prob(action).unsqueeze(-1)
            outputs = [action, action_log_probs, None]
            if aux_outputs:
                outputs.extend(
                    [
                        aux_outputs["vel_cmd"],
                        aux_outputs["waypoint"],
                        aux_outputs["target_pos_pred"],
                        aux_outputs["forward_dir"],
                        aux_outputs["assignment_logits"],
                        aux_outputs["trap_logit"],
                    ]
                )
            if self.rnn is not None:
                outputs.append(rnn_state)
            return tuple(outputs)
        else:
            action = action_dist.mode if deterministic else action_dist.sample()
            action_log_probs = action_dist.log_prob(action).unsqueeze(-1)
            outputs = [action, action_log_probs, None]
            if aux_outputs:
                outputs.extend(
                    [
                        aux_outputs["vel_cmd"],
                        aux_outputs["waypoint"],
                        aux_outputs["target_pos_pred"],
                        aux_outputs["forward_dir"],
                        aux_outputs["assignment_logits"],
                        aux_outputs["trap_logit"],
                    ]
                )
            if self.rnn is not None:
                outputs.append(rnn_state)
            return tuple(outputs)

class Critic(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        rnn: nn.Module,
        v_out: nn.Module,
        output_shape: torch.Size=torch.Size((-1,)),
    ):
        super().__init__()
        self.base = base
        self.rnn = rnn
        self.v_out = v_out
        self.output_shape = output_shape

    def forward(
        self, 
        critic_input: torch.Tensor,
        rnn_state: torch.Tensor=None,
        is_init: torch.Tensor=None,
    ):
        critic_features = self.base(critic_input)
        if self.rnn is not None:
            critic_features, rnn_state = self.rnn(critic_features, rnn_state, is_init)
        else:
            rnn_state = None

        values = self.v_out(critic_features)

        if len(self.output_shape) > 1:
            values = values.unflatten(-1, self.output_shape)
        return values, rnn_state
