#!/usr/bin/env python3
"""
BC Actor 离线前向推理诊断
=========================
不需要 Isaac Sim。直接加载 BC checkpoint，对数据集中的 obs 做前向推理，
对比 actor 预测动作 vs 专家标签，按角色和 close 模式分桶统计。

核心问题：BC actor 是否学到了角色分化的动作？

Usage:
  CUDA_VISIBLE_DEVICES=1 conda run -n sim python3 scripts/bc_actor_forward_diag.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

# ── 配置 ──────────────────────────────────────────────────────────────────
DATASET_CHUNK = "expert_datasets/expert2_step5_aligned_histfix_1024x5_20260415_154654/expert_success_wave_00001.pt"
CHECKPOINT    = "checkpoints/debug_bc_5ep_histfix_20260415_155725/bc_best.pt"
DEVICE        = "cpu"      # actor forward 在 CPU 上，省得 CUDA 跨卡问题
N_SAMPLE      = 8000       # 随机采样步数（足够快）
SEED          = 42

DIM_NAMES   = ["roll", "pitch", "yaw", "thrust"]
ROLE_NAMES  = ["chaser", "front_left", "front_right"]

def fmt4(t):
    if isinstance(t, torch.Tensor):
        return [round(float(v), 4) for v in t.tolist()]
    return t

# ── 1. 加载数据 ──────────────────────────────────────────────────────────
print(f"Loading chunk: {DATASET_CHUNK}")
chunk = torch.load(DATASET_CHUNK, map_location="cpu", weights_only=False)
obs   = chunk["obs"]
act_tgt = chunk["action_raw"].float()          # [T, A, 4]  expert labels

ss   = obs["state_self"].float().squeeze(-2)   # [T, A, 23]
so   = obs["state_others"].float().squeeze(-2) # [T, A, 2, 9] — may have extra dim
coop = obs["cooperation"].float().squeeze(-2)  # [T, A, 24]

if so.ndim == 4 and so.shape[-2] == 2:
    pass  # [T, A, 2, 9] already correct
elif so.ndim == 5:
    so = so.squeeze(-3)  # try to remove any extra dim

T, A, _ = act_tgt.shape
print(f"  T={T}  A={A}  state_self={ss.shape}  state_others={so.shape}  coop={coop.shape}")

# ── 2. 提取角色/close 标签 ──────────────────────────────────────────────
role_id    = ss[..., 20:23].argmax(-1)          # [T, A]  0/1/2
close_flag = ss[..., 19] > 0.5                 # [T, A]

# ── 3. 构建 flat 索引，随机采样 ──────────────────────────────────────────
torch.manual_seed(SEED)
all_indices = torch.randperm(T * A)[:N_SAMPLE]
t_idx = all_indices // A
a_idx = all_indices % A

# 采样的 expert 标签
tgt_sample = act_tgt[t_idx, a_idx]   # [N, 4]
role_sample = role_id[t_idx, a_idx]   # [N]
close_sample = close_flag[t_idx, a_idx]  # [N]

# ── 4. 尝试加载 actor 做前向推理 ──────────────────────────────────────────
print(f"\nLoading checkpoint: {CHECKPOINT}")
state_dict = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
print("  Checkpoint top-level keys:", list(state_dict.keys())[:8])

# 尝试找 actor_params
actor_params = state_dict.get("actor_params", None)
if actor_params is None:
    print("  [WARN] 'actor_params' not found in checkpoint. Trying other keys...")
    for k in state_dict:
        print(f"    {k}: {type(state_dict[k])}")

# 检查是否有 actor 网络权重
# 尝试加载 omni_drones 的 policy
actor_loaded = False
pred_sample  = None
try:
    # 尝试直接加载 MAPPOPolicy 的方式
    import hydra
    from omegaconf import OmegaConf
    from omni_drones import CONFIG_PATH
    from omni_drones.utils.torchrl import AgentSpec
    from omni_drones.learning import MAPPOPolicy

    # 构造一个最小 AgentSpec 和 cfg
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cfg")
    with hydra.initialize_config_dir(config_dir=cfg_path, job_name="diag"):
        cfg = hydra.compose(config_name="train", overrides=["task=HideAndSeek", "headless=true", "wandb.mode=disabled"])
    OmegaConf.set_struct(cfg, False)

    # We can't build env here (no Isaac), but we can try to extract actor weights
    # and inspect them directly
    actor_loaded = False
    print("  [INFO] hydra cfg loaded, but skipping env build (no Isaac Sim)")
except Exception as e:
    print(f"  [WARN] Could not load via hydra: {e}")

# ── 5. 直接检查 actor_params 中的权重 ──────────────────────────────────────
print("\n── Actor Weights Inspection ──")
if "actor_params" in state_dict:
    ap = state_dict["actor_params"]
    print(f"  actor_params type: {type(ap)}")
    if hasattr(ap, "keys"):
        all_keys = list(ap.keys(True, True)) if hasattr(ap, 'keys') else []
        # Try iterating
        try:
            for k in ap.keys(True, True):
                t = ap.get(k)
                if hasattr(t, "shape"):
                    print(f"    {k}: {tuple(t.shape)}")
        except Exception:
            try:
                td = ap.to_tensordict() if hasattr(ap, 'to_tensordict') else ap
                for k, v in td.items(True, True):
                    if hasattr(v, "shape"):
                        print(f"    {k}: {tuple(v.shape)}")
            except Exception as e2:
                print(f"    Error iterating: {e2}")
    # Check log_std
    try:
        log_std = ap["module"]["act_dist"]["log_std"]
        print(f"\n  log_std: shape={tuple(log_std.shape)} values={fmt4(log_std)}")
        print(f"  → std (exp(log_std)): {[round(float(v.exp()), 4) for v in log_std.reshape(-1)]}")
    except Exception as e:
        print(f"  Could not extract log_std: {e}")

# ── 6. 标签分布分析（不依赖 actor 前向） ──────────────────────────────────
print("\n── Expert Label Distribution (sampled N={}) ──".format(N_SAMPLE))
print(f"  Sample role dist: chaser={int((role_sample==0).sum())} fl={int((role_sample==1).sum())} fr={int((role_sample==2).sum())}")
print(f"  Sample close dist: close={int(close_sample.sum())} normal={int((~close_sample).sum())}")

# 角色分化检验:
# 如果专家的 action 对不同角色有显著差异，但 actor 输出相同，就是角色混淆
print("\n  Expert action by role (mean/std):")
for r, rn in enumerate(ROLE_NAMES):
    m = role_sample == r
    a = tgt_sample[m]
    if a.shape[0] > 0:
        print(f"    {rn}: mean={fmt4(a.mean(0))}  std={fmt4(a.std(0))}")

print("\n  Expert action by close mode:")
for flag, fn in [(True,"close"), (False,"normal")]:
    m = close_sample == flag
    a = tgt_sample[m]
    if a.shape[0] > 0:
        print(f"    {fn}: n={a.shape[0]}  mean={fmt4(a.mean(0))}  std={fmt4(a.std(0))}")

# 关键检验：role x close 交叉分析（每种情况的 yaw mean/std）
print("\n  Expert action by role x close_mode:")
for r, rn in enumerate(ROLE_NAMES):
    for flag, fn in [(True,"close"), (False,"normal")]:
        m = (role_sample == r) & (close_sample == flag)
        a = tgt_sample[m]
        if a.shape[0] > 10:
            row_str = (f"    {rn} x {fn}: n={a.shape[0]:4d}  "
                       f"yaw: mean={a[:,2].mean():.4f} std={a[:,2].std():.4f}  "
                       f"thrust: mean={a[:,3].mean():.4f} std={a[:,3].std():.4f}")
            print(row_str)

# ── 7. 检验 obs 是否确实包含 role 信息（actor 输入层面） ──────────────────
print("\n── Obs Role Signal Check ──")
# 取各角色的 obs，看 role_onehot 是否正确变化，其他字段是否有差异
for r, rn in enumerate(ROLE_NAMES):
    m = role_sample == r
    if m.sum() > 0:
        ss_r = ss[t_idx[m], a_idx[m]]  # [N_r, 23]
        role_hot = ss_r[:, 20:23]
        print(f"  {rn}: role_onehot mean={fmt4(role_hot.mean(0))}  (should be [{1 if r==0 else 0},{1 if r==1 else 0},{1 if r==2 else 0}] approx)")
        wp = ss_r[:, 13:16]
        print(f"           assigned_wp mean={fmt4(wp.mean(0))}  std={fmt4(wp.std(0))}")

# ── 8. 检查 cooperation step5 预测值合理性 ──────────────────────────────
print("\n── Cooperation Field Check (correct layout) ──")
# target_vel = coop[0:3]
# target_pred_step1..step5 = coop[3:18]
# forward_dir = coop[18:21]
# lateral = coop[21:24]
coop_sample = coop[t_idx, a_idx]  # [N, 24]
print(f"  target_vel mean={fmt4(coop_sample[:, 0:3].mean(0))}  std={fmt4(coop_sample[:, 0:3].std(0))}")
print(f"  pred_step1 mean={fmt4(coop_sample[:, 3:6].mean(0))}  std={fmt4(coop_sample[:, 6:9].std(0))}")
print(f"  pred_step5 mean={fmt4(coop_sample[:, 15:18].mean(0))}  std={fmt4(coop_sample[:, 15:18].std(0))}")
print(f"  forward_dir mean={fmt4(coop_sample[:, 18:21].mean(0))}  std={fmt4(coop_sample[:, 18:21].std(0))}")
print(f"  lateral mean={fmt4(coop_sample[:, 21:24].mean(0))}  std={fmt4(coop_sample[:, 21:24].std(0))}")

# 距离计算（使用正确字段）:
# self_pos = ss[..., 0:3]
# target_pred_step1 = coop[..., 3:6]  (closest to current target position)
self_pos_sample  = ss[t_idx, a_idx][:, 0:3]   # [N, 3]
tgt_pred1_sample = coop_sample[:, 3:6]         # [N, 3]
dist_sample = (tgt_pred1_sample - self_pos_sample).norm(dim=-1)  # [N]
print(f"\n  dist to target (pred_step1 - self_pos): mean={dist_sample.mean():.3f}  std={dist_sample.std():.3f}  min={dist_sample.min():.3f}  max={dist_sample.max():.3f}")

# 距离分桶的 action 统计
print("\n  Expert action by CORRECT distance bucket:")
for dname, lo, hi in [("dist>3", 3.0, 99), ("dist_2_3", 2.0, 3.0), ("dist_1_2", 1.0, 2.0), ("dist_0.5_1", 0.5, 1.0), ("dist<0.5", 0, 0.5)]:
    m = (dist_sample >= lo) & (dist_sample < hi)
    a = tgt_sample[m]
    if a.shape[0] > 10:
        print(f"    [{dname}] n={a.shape[0]:4d}  yaw: mean={a[:,2].mean():.3f} std={a[:,2].std():.3f}  thrust: mean={a[:,3].mean():.3f} std={a[:,3].std():.3f}")

print("\nDone.")
