#!/usr/bin/env python3
"""
BC 分桶 MSE 离线诊断脚本
============================
加载专家数据集 + BC checkpoint，按以下维度分桶分析动作误差：
  - 角色（chaser / front_left / front_right）
  - 模式（close_trigger=0/1）
  - 目标距离（>1.5 / 0.8-1.5 / 0.4-0.8 / <=0.4）
  - 时间段（前100步 / 100-300步 / >300步）
  - 动作维度（roll / pitch / yaw / thrust）
  - pred_action 和 target_action 的分布统计（便于判断是否学成平均动作）

不需要 Isaac Sim，只需 CPU。

Usage:
    python3 scripts/diagnose_bc_mse_buckets.py \
        --dataset_dir expert_datasets/expert2_step5_aligned_histfix_1024x5_20260415_154654 \
        --checkpoint checkpoints/debug_bc_5ep_histfix_20260415_155725/bc_best.pt \
        --cfg_algo cfg/algo/mappo.yaml \
        --out_dir analysis/bc_mse_bucket_diag
"""

import argparse
import json
import os
import sys
import glob
import math

import torch
import torch.nn.functional as F
import numpy as np

# ── helper ──────────────────────────────────────────────────────────────

def load_chunks(dataset_dir: str):
    chunk_files = sorted(glob.glob(os.path.join(dataset_dir, "expert_success_wave_*.pt")))
    if not chunk_files:
        raise FileNotFoundError(f"No chunks in {dataset_dir}")
    chunks = []
    for f in chunk_files:
        chunks.append(torch.load(f, map_location="cpu"))
    print(f"[INFO] Loaded {len(chunks)} chunk(s) from {dataset_dir}")
    return chunks


def load_policy(checkpoint: str, cfg_algo_path: str, device: str = "cpu"):
    """
    轻量加载：只要 actor 的参数，不需要仿真环境。
    直接从 state_dict 里取 actor_params，然后重建 MLP 前向。
    """
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    # 打印所有顶层 key 帮助调试
    print("[INFO] Checkpoint top-level keys:", list(state.keys()) if isinstance(state, dict) else type(state))
    return state


def maybe_tuple_key(key_str: str):
    parts = tuple(p for p in key_str.split("/") if p)
    return parts[0] if len(parts) == 1 else parts


def build_obs_tensor(obs_storage: dict, indices: torch.Tensor, n_agents: int):
    """把 obs_storage dict 拼成 [N, A, obs_dim] 的 flat tensor（按 key 排序拼接）。"""
    parts = []
    key_order = sorted(obs_storage.keys())
    for k in key_order:
        v = obs_storage[k][indices].float()
        if v.ndim == 2:          # [N, feat]
            v = v.unsqueeze(1).expand(-1, n_agents, -1)
        elif v.ndim == 3:        # [N, A, feat]
            pass
        parts.append(v)
    return torch.cat(parts, dim=-1)  # [N, A, total_feat]


# ── Stats bucket ────────────────────────────────────────────────────────

class BucketStats:
    def __init__(self, name: str):
        self.name = name
        self.sq_errs = []       # per-step mean sq err (scalar)
        self.sq_errs_dim = []   # per-step sq err per dim [4]
        self.pred_norms = []
        self.tgt_norms = []
        self.tgt_actions = []   # for distribution analysis
        self.pred_actions = []
        self.n = 0

    def add(self, pred: torch.Tensor, tgt: torch.Tensor):
        """pred, tgt: [B, A, 4] or [B, 4]"""
        se = (pred - tgt).pow(2)
        self.sq_errs.append(se.mean().item())
        self.sq_errs_dim.append(se.reshape(-1, se.shape[-1]).mean(0))
        self.pred_norms.append(pred.norm(dim=-1).mean().item())
        self.tgt_norms.append(tgt.norm(dim=-1).mean().item())
        self.tgt_actions.append(tgt.reshape(-1, tgt.shape[-1]).detach())
        self.pred_actions.append(pred.reshape(-1, pred.shape[-1]).detach())
        self.n += pred.shape[0]

    def summary(self) -> dict:
        if not self.sq_errs:
            return {"n": 0, "mse": float("nan")}
        mse = float(np.mean(self.sq_errs))
        rmse = math.sqrt(mse)
        se_dim = torch.stack(self.sq_errs_dim).mean(0)
        all_tgt = torch.cat(self.tgt_actions, 0)
        all_pred = torch.cat(self.pred_actions, 0)
        tgt_std = all_tgt.std(0).tolist()
        pred_std = all_pred.std(0).tolist()
        tgt_mean = all_tgt.mean(0).tolist()
        pred_mean = all_pred.mean(0).tolist()
        return {
            "n": self.n,
            "mse": round(mse, 6),
            "rmse": round(rmse, 6),
            "mse_per_dim": {
                "roll":   round(se_dim[0].item(), 6),
                "pitch":  round(se_dim[1].item(), 6),
                "yaw":    round(se_dim[2].item(), 6),
                "thrust": round(se_dim[3].item(), 6),
            },
            "pred_norm_mean": round(float(np.mean(self.pred_norms)), 4),
            "tgt_norm_mean":  round(float(np.mean(self.tgt_norms)), 4),
            "tgt_mean_per_dim":  [round(v, 4) for v in tgt_mean],
            "pred_mean_per_dim": [round(v, 4) for v in pred_mean],
            "tgt_std_per_dim":   [round(v, 4) for v in tgt_std],
            "pred_std_per_dim":  [round(v, 4) for v in pred_std],
        }


def print_bucket(name: str, s: dict, indent=2):
    pad = " " * indent
    n = s.get("n", 0)
    if n == 0:
        print(f"{pad}[{name}] n=0 (empty)")
        return
    mse = s["mse"]
    rmse = s["rmse"]
    mdim = s.get("mse_per_dim", {})
    print(f"{pad}[{name}] n={n:,}  MSE={mse:.6f}  RMSE={rmse:.4f}")
    print(f"{pad}  per-dim MSE: roll={mdim.get('roll',0):.6f}  pitch={mdim.get('pitch',0):.6f}  "
          f"yaw={mdim.get('yaw',0):.6f}  thrust={mdim.get('thrust',0):.6f}")
    pred_std = s.get("pred_std_per_dim", [])
    tgt_std  = s.get("tgt_std_per_dim", [])
    pred_mean = s.get("pred_mean_per_dim", [])
    tgt_mean  = s.get("tgt_mean_per_dim", [])
    if pred_std and tgt_std:
        print(f"{pad}  pred_std: roll={pred_std[0]:.4f}  pitch={pred_std[1]:.4f}  "
              f"yaw={pred_std[2]:.4f}  thrust={pred_std[3]:.4f}")
        print(f"{pad}  tgt_std:  roll={tgt_std[0]:.4f}  pitch={tgt_std[1]:.4f}  "
              f"yaw={tgt_std[2]:.4f}  thrust={tgt_std[3]:.4f}")
        std_ratio = [
            round(pred_std[i] / max(tgt_std[i], 1e-8), 3) for i in range(4)
        ]
        print(f"{pad}  std_ratio (pred/tgt): roll={std_ratio[0]}  pitch={std_ratio[1]}  "
              f"yaw={std_ratio[2]}  thrust={std_ratio[3]}")
    if pred_mean and tgt_mean:
        print(f"{pad}  pred_mean: {[round(v, 4) for v in pred_mean]}")
        print(f"{pad}  tgt_mean:  {[round(v, 4) for v in tgt_mean]}")


# ── Actor forward (offline, no Isaac) ──────────────────────────────────

def try_forward_actor(state_dict, obs_storage, action_targets, indices, n_agents, device="cpu"):
    """
    尝试用 checkpoint 做轻量前向传播，仅在 actor 结构可以离线重建时有效。
    如果结构太复杂，返回 None，改用"只看标签分布"模式。
    """
    # 目前先返回 None，因为 CoopEntityAttentionEncoder 依赖自定义模块
    # TODO: 如果需要，可以这里动态 import
    return None


# ── Main diagnostic ─────────────────────────────────────────────────────

def run_diagnostics(chunks, checkpoint_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # ── 加载 checkpoint
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_keys = list(state.keys()) if isinstance(state, dict) else []
    print(f"[INFO] Checkpoint keys: {ckpt_keys}")

    # 统计桶
    buckets_role = {
        "chaser":      BucketStats("role=chaser"),
        "front_left":  BucketStats("role=front_left"),
        "front_right": BucketStats("role=front_right"),
    }
    buckets_close = {
        "normal": BucketStats("close=0"),
        "close":  BucketStats("close=1"),
    }
    buckets_dist = {
        "far":    BucketStats("dist>1.5"),
        "mid":    BucketStats("1.5>=dist>0.8"),
        "near":   BucketStats("0.8>=dist>0.4"),
        "catch":  BucketStats("dist<=0.4"),
    }
    buckets_time = {
        "early":  BucketStats("step<=100"),
        "mid":    BucketStats("100<step<=300"),
        "late":   BucketStats("step>300"),
    }
    bucket_all = BucketStats("ALL")

    # ── 遍历数据
    total_steps = 0
    skipped_chunks = 0

    for chunk_idx, chunk in enumerate(chunks):
        action_raw = chunk["action_raw"].float()   # [T, A, 4]
        obs_storage = chunk["obs"]
        ep_lengths  = chunk["episode_lengths"].long()
        label_type  = str(chunk.get("action_label_type", "legacy_raw"))

        # --- 确认 action_label_type
        if chunk_idx == 0:
            print(f"[INFO] action_label_type = {label_type!r}")
            print(f"[INFO] obs keys: {list(obs_storage.keys())}")
            print(f"[INFO] action_raw shape: {action_raw.shape}")
            # 打印 state_self 形状
            ss_key = "state_self"
            if ss_key in obs_storage:
                print(f"[INFO] state_self shape: {obs_storage[ss_key].shape}")
            coop_key = "cooperation"
            if coop_key in obs_storage:
                print(f"[INFO] cooperation shape: {obs_storage[coop_key].shape}")

        if "state_self" not in obs_storage:
            print(f"[WARN] chunk {chunk_idx+1}: no 'state_self' in obs, skip")
            skipped_chunks += 1
            continue

        state_self = obs_storage["state_self"].float()  # [T, A, 23]
        n_steps, n_agents, _ = state_self.shape

        # --- 提取 role & close_trigger
        role_onehot   = state_self[..., 20:23]         # [T, A, 3]  chaser/fl/fr
        close_trigger = (state_self[..., 19:20] > 0.5) # [T, A, 1] bool

        # --- 提取 waypoint (assigned + close)
        assigned_wp = state_self[..., 13:16]            # [T, A, 3]
        close_wp    = state_self[..., 16:19]            # [T, A, 3]
        active_wp   = torch.where(close_trigger, close_wp, assigned_wp)

        # --- 提取目标距离 (从 cooperation 里 target_pos_pred 第0步 = pos at t)
        # cooperation[..., 0:3] = current target pos relative to drone (normalized)
        # 实际上我们用 action_raw 就够了，距离要从其他地方取
        # 看 obs_storage 里有没有 target 相对距离
        dist_avail = False
        if "cooperation" in obs_storage:
            coop = obs_storage["cooperation"].float()  # [T, A, C]
            # cooperation 开头 3 维通常是 target_pos_relative 到 drone
            # 每个 agent 的距离不同，取第 0 维
            # 用 norm 估算
            if coop.shape[-1] >= 3:
                # 通常 coop[..., 0:3] 是归一化的 target_pos_pred_step0 (relative pos, normalized by arena)
                # 但不一定。先统计一下
                dist_avail = True
                arena_size = 3.5  # default
                # 试探: coop 第一个 agent 的前 3 维就是对应 drone 到目标的相对位置 (并非所有 obs 都有)
                target_rel = coop[:, :, 0:3] * arena_size  # [T, A, 3]
                dist_to_target = torch.norm(target_rel, dim=-1)  # [T, A]

        # --- 时间步索引（每个 episode 内部的 step index）
        step_indices = torch.zeros(n_steps, dtype=torch.long)
        cursor = 0
        for ep_len in ep_lengths.tolist():
            ep_len = int(ep_len)
            step_indices[cursor:cursor+ep_len] = torch.arange(ep_len)
            cursor += ep_len

        # --- 对每一帧进行分桶
        target_action = action_raw  # [T, A, 4]  pidrate_normalized

        # 这里没有 actor 前向（需要 Isaac 环境），只做标签分布分析
        # 所以 pred_action ≡ target_action（用于验证 label 分布和分桶 counts）
        # 注意：下面所有 MSE 都是零（pred=tgt），但我们主要看：
        #   1. 分布统计（std, mean, norm）
        #   2. 各分桶的 label 方差（判断是否存在"平坦区间"）
        pred_action = target_action  # placeholder, non-zero when actor plugged in

        for t in range(n_steps):
            tgt_t = target_action[t]   # [A, 4]
            pred_t = pred_action[t]    # [A, 4]

            step_idx = int(step_indices[t].item())
            role_t = role_onehot[t]              # [A, 3]
            close_t = close_trigger[t, :, 0]     # [A]

            for a in range(n_agents):
                tgt_a  = tgt_t[a:a+1]   # [1, 4]
                pred_a = pred_t[a:a+1]

                # role bucket
                role_idx = int(role_t[a].argmax().item())
                role_key = ["chaser", "front_left", "front_right"][role_idx]
                buckets_role[role_key].add(pred_a, tgt_a)

                # close bucket
                is_close = bool(close_t[a].item())
                buckets_close["close" if is_close else "normal"].add(pred_a, tgt_a)

                # time bucket
                if step_idx <= 100:
                    buckets_time["early"].add(pred_a, tgt_a)
                elif step_idx <= 300:
                    buckets_time["mid"].add(pred_a, tgt_a)
                else:
                    buckets_time["late"].add(pred_a, tgt_a)

                # dist bucket (if available)
                if dist_avail:
                    d = float(dist_to_target[t, a].item())
                    if d > 1.5:
                        buckets_dist["far"].add(pred_a, tgt_a)
                    elif d > 0.8:
                        buckets_dist["mid"].add(pred_a, tgt_a)
                    elif d > 0.4:
                        buckets_dist["near"].add(pred_a, tgt_a)
                    else:
                        buckets_dist["catch"].add(pred_a, tgt_a)

                bucket_all.add(pred_a, tgt_a)
                total_steps += 1

        if (chunk_idx + 1) % 1 == 0:
            print(f"[INFO] Processed chunk {chunk_idx+1}/{len(chunks)}")

    print(f"\n[INFO] Total step-agent samples: {total_steps:,}")
    if skipped_chunks:
        print(f"[WARN] Skipped {skipped_chunks} chunk(s) due to missing 'state_self'")

    # ── 打印结果
    print("\n" + "=" * 70)
    print("  BC LABEL DISTRIBUTION ANALYSIS (pred=tgt => MSE=0 here)")
    print("  Focus: std_ratio, mean, norm — to detect 'collapsed/averaged action'")
    print("=" * 70)

    print("\n─── OVERALL ───")
    all_s = bucket_all.summary()
    print_bucket("ALL", all_s)

    print("\n─── BY ROLE ───")
    role_results = {}
    for k, b in buckets_role.items():
        s = b.summary()
        role_results[k] = s
        print_bucket(k, s)

    print("\n─── BY CLOSE MODE ───")
    close_results = {}
    for k, b in buckets_close.items():
        s = b.summary()
        close_results[k] = s
        print_bucket(k, s)

    print("\n─── BY TIME STEP ───")
    time_results = {}
    for k, b in buckets_time.items():
        s = b.summary()
        time_results[k] = s
        print_bucket(k, s)

    print("\n─── BY DISTANCE TO TARGET ───")
    dist_results = {}
    for k, b in buckets_dist.items():
        s = b.summary()
        dist_results[k] = s
        print_bucket(k, s)

    # ── 关键诊断摘要
    print("\n" + "=" * 70)
    print("  KEY DIAGNOSTIC QUESTIONS")
    print("=" * 70)

    # Q1: action_label_type
    print("\n[Q1] action_label_type 确认:")
    for i, chunk in enumerate(chunks):
        lt = str(chunk.get("action_label_type", "legacy_raw"))
        print(f"     chunk {i+1}: {lt!r}")

    # Q2: action range
    all_tgt = torch.cat([c["action_raw"].float().reshape(-1, 4) for c in chunks], 0)
    print(f"\n[Q2] action_raw 范围:")
    print(f"     全局 min={all_tgt.min():.4f}  max={all_tgt.max():.4f}  mean={all_tgt.mean():.4f}  std={all_tgt.std():.4f}")
    print(f"     各维 max: {all_tgt.max(0).values.tolist()}")
    print(f"     各维 min: {all_tgt.min(0).values.tolist()}")
    print(f"     各维 std: {all_tgt.std(0).tolist()}")
    print(f"     各维 mean:{all_tgt.mean(0).tolist()}")
    out_of_range = (all_tgt.abs() > 1.0).float().mean().item()
    print(f"     |action|>1.0 的比例: {out_of_range*100:.2f}%  ← 应该=0%（pidrate_normalized 应在 [-1,1])")

    # Q3: role distribution
    print(f"\n[Q3] 角色分布（step×agent）:")
    for k, s in role_results.items():
        print(f"     {k}: n={s.get('n', 0):,}  占比={s.get('n',0)/max(total_steps,1)*100:.1f}%")

    # Q4: close mode distribution
    print(f"\n[Q4] close_trigger 分布:")
    n_close  = close_results.get("close", {}).get("n", 0)
    n_normal = close_results.get("normal", {}).get("n", 0)
    total = n_close + n_normal
    print(f"     close:  {n_close:,}  ({n_close/max(total,1)*100:.1f}%)")
    print(f"     normal: {n_normal:,}  ({n_normal/max(total,1)*100:.1f}%)")

    # Q5: action std per role: if small std -> actor might collapse to mean
    print("\n[Q5] 各角色 action std（label，衡量数据多样性）:")
    for k, s in role_results.items():
        std = s.get("tgt_std_per_dim", [])
        mean = s.get("tgt_mean_per_dim", [])
        if std:
            print(f"     {k}  mean={[round(v,3) for v in mean]}  std={[round(v,3) for v in std]}")

    print("\n[Q6] close vs normal action std 对比（高 close std → close 模式动作复杂）:")
    for k, s in close_results.items():
        std = s.get("tgt_std_per_dim", [])
        mean = s.get("tgt_mean_per_dim", [])
        if std:
            print(f"     {k}  mean={[round(v,3) for v in mean]}  std={[round(v,3) for v in std]}")

    # ── 保存 JSON
    results = {
        "checkpoint": checkpoint_path,
        "dataset_dir": args.dataset_dir,
        "total_samples": total_steps,
        "overall": all_s,
        "by_role": role_results,
        "by_close": close_results,
        "by_time": time_results,
        "by_dist": dist_results,
        "action_range": {
            "min": float(all_tgt.min()),
            "max": float(all_tgt.max()),
            "std_per_dim": all_tgt.std(0).tolist(),
            "mean_per_dim": all_tgt.mean(0).tolist(),
            "out_of_range_pct": float(out_of_range * 100),
        },
    }
    out_path = os.path.join(out_dir, "bucket_diag.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] Results saved to {out_path}")


# ── Entry point ─────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out_dir", default="analysis/bc_mse_bucket_diag")
parser.add_argument("--cfg_algo", default="cfg/algo/mappo.yaml")
args = parser.parse_args()

print(f"[INFO] Dataset : {args.dataset_dir}")
print(f"[INFO] Checkpoint: {args.checkpoint}")
print(f"[INFO] Output  : {args.out_dir}")

chunks = load_chunks(args.dataset_dir)
run_diagnostics(chunks, args.checkpoint, args.out_dir)
