#!/usr/bin/env python3
"""
BC 数据集快速诊断（GPU 加速版）
====================================
不依赖 Isaac Sim，纯离线分析数据集标签分布和动作语义。

诊断内容：
  A. action_raw 值域核查 (是否真的是 pidrate_normalized ∈ [-1,1])
  B. state_self 字段验证 (role / close_trigger / waypoint 位置是否正确)
  C. cooperation 字段验证 (target_pos_pred 步序是否匹配 future_steps=5)
  D. 按角色/close/时间/距离 分桶统计动作标准差
     → 标准差远小于专家数据 → actor 学成了平均值
  E. 各角色动作均值对比（是否出现角色混淆）
  F. 数据集内 action_raw 分布 vs tanh 输出范围一致性

Usage:
  python3 scripts/diagnose_bc_gpu.py \
      --dataset_dir expert_datasets/expert2_step5_aligned_histfix_1024x5_20260415_154654 \
      --checkpoint  checkpoints/debug_bc_5ep_histfix_20260415_155725/bc_best.pt \
      --device cuda:1 \
      --out_dir analysis/bc_diag_gpu
"""

import argparse
import json
import os
import sys
import glob
import math
import time

import torch
import numpy as np

# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────

DIM_NAMES = ["roll", "pitch", "yaw", "thrust"]


def fmt(t):
    if isinstance(t, torch.Tensor):
        return [round(float(v), 4) for v in t.tolist()]
    return t


def load_chunk_fast(path: str, device: str):
    """Load a .pt chunk directly to GPU."""
    t0 = time.time()
    chunk = torch.load(path, map_location=device, weights_only=False)
    dt = time.time() - t0
    print(f"  loaded {os.path.basename(path)} ({os.path.getsize(path)/(1<<20):.0f}MB) in {dt:.1f}s")
    return chunk


def episode_step_indices(episode_lengths: torch.Tensor) -> torch.Tensor:
    """Build step-within-episode index for every row in flattened dataset."""
    indices = []
    for L in episode_lengths.tolist():
        indices.append(torch.arange(int(L)))
    return torch.cat(indices)


# ────────────────────────────────────────────────────────────────────────
# Diagnostics
# ────────────────────────────────────────────────────────────────────────

def analyze_chunk(chunk, device, arena_size=3.5):
    """Return a stats dict for one chunk. Everything runs on `device`."""
    obs = chunk["obs"]
    action_raw = chunk["action_raw"].to(device=device, dtype=torch.float32)  # [T, A, 4]
    ep_lens = chunk["episode_lengths"].long()

    T, A, _ = action_raw.shape

    # ── state_self — stored as [T, A, 1, 23], squeeze the extra dim ──
    state_self = obs["state_self"].to(device=device, dtype=torch.float32)  # [T, A, 1, 23]
    if state_self.ndim == 4:
        state_self = state_self.squeeze(-2)  # → [T, A, 23]
    D_ss = state_self.shape[-1]
    assigned_wp  = state_self[..., 13:16]
    close_wp     = state_self[..., 16:19]
    close_trig   = state_self[..., 19] > 0.5   # [T, A]
    role_onehot  = state_self[..., 20:23]       # [T, A, 3]
    role_id      = role_onehot.argmax(-1)       # [T, A]  0=chaser 1=fl 2=fr

    # ── cooperation — stored as [T, A, 1, 24], squeeze the extra dim ──
    has_coop = "cooperation" in obs
    dist_to_target = None
    if has_coop:
        coop = obs["cooperation"].to(device=device, dtype=torch.float32)  # [T, A, 1, 24]
        if coop.ndim == 4:
            coop = coop.squeeze(-2)  # → [T, A, 24]
        # cooperation layout (24d):
        #   [0:3]  = target_pos_pred step0 (relative to this drone, NOT normalized here)
        #   [3:6]  = step1 ... [12:15]=step4, [15:18]=step5(?), [18:21]=forward_dir(?)
        # Use norm of [0:3] as rough distance proxy
        step0_rel = coop[..., 0:3]   # [T, A, 3]
        dist_to_target = step0_rel.norm(dim=-1)    # [T, A]

    # ── step index within episode ──
    step_idx = episode_step_indices(ep_lens).to(device=device)  # [T]
    step_idx_expanded = step_idx.unsqueeze(1).expand(T, A)      # [T, A]

    # ── flat view ──
    act_flat  = action_raw.reshape(T * A, 4)
    role_flat = role_id.reshape(T * A)
    close_flat = close_trig.reshape(T * A)
    step_flat  = step_idx_expanded.reshape(T * A)

    results = {
        "T": T, "A": A, "D_ss": D_ss,
        "label_type": str(chunk.get("action_label_type", "MISSING")),
        "n_episodes": int(ep_lens.shape[0]),
        "ep_len_mean": float(ep_lens.float().mean()),
        "ep_len_min": int(ep_lens.min()),
        "ep_len_max": int(ep_lens.max()),
    }

    # ── A. Action range ──
    act_min = float(act_flat.min())
    act_max = float(act_flat.max())
    act_mean = fmt(act_flat.mean(0))
    act_std  = fmt(act_flat.std(0))
    act_oor  = float((act_flat.abs() > 1.0).float().mean()) * 100
    results["action"] = {
        "min": round(act_min, 4),
        "max": round(act_max, 4),
        "mean_per_dim": act_mean,
        "std_per_dim":  act_std,
        "pct_out_of_range": round(act_oor, 3),
    }

    # ── B. Role distribution ──
    role_names = ["chaser", "front_left", "front_right"]
    role_dist = {}
    for i, name in enumerate(role_names):
        mask = role_flat == i
        n = int(mask.sum())
        acts = act_flat[mask]
        role_dist[name] = {
            "n": n,
            "pct": round(n / (T * A) * 100, 1),
            "mean": fmt(acts.mean(0)) if n > 0 else None,
            "std":  fmt(acts.std(0))  if n > 0 else None,
        }
    results["by_role"] = role_dist

    # ── C. Close vs Normal ──
    close_stats = {}
    for flag, name in [(True, "close"), (False, "normal")]:
        mask = close_flat == flag
        n = int(mask.sum())
        acts = act_flat[mask]
        close_stats[name] = {
            "n": n,
            "pct": round(n / (T * A) * 100, 1),
            "mean": fmt(acts.mean(0)) if n > 0 else None,
            "std":  fmt(acts.std(0))  if n > 0 else None,
        }
    results["by_close"] = close_stats

    # ── D. Time buckets ──
    time_stats = {}
    for tname, lo, hi in [("step_0_100", 0, 100), ("step_100_300", 101, 300), ("step_300p", 301, 9999)]:
        mask = (step_flat >= lo) & (step_flat <= hi)
        n = int(mask.sum())
        acts = act_flat[mask]
        time_stats[tname] = {
            "n": n,
            "mean": fmt(acts.mean(0)) if n > 0 else None,
            "std":  fmt(acts.std(0))  if n > 0 else None,
        }
    results["by_time"] = time_stats

    # ── E. Distance buckets ──
    if dist_to_target is not None:
        dist_flat = dist_to_target.reshape(T * A)
        dist_stats = {}
        for dname, lo, hi in [
            ("dist_gt1.5", 1.5, 1e9),
            ("dist_0.8_1.5", 0.8, 1.5),
            ("dist_0.4_0.8", 0.4, 0.8),
            ("dist_le0.4", 0.0, 0.4),
        ]:
            mask = (dist_flat >= lo) & (dist_flat < hi)
            n = int(mask.sum())
            acts = act_flat[mask]
            dist_stats[dname] = {
                "n": n,
                "pct": round(n / (T * A) * 100, 1),
                "mean": fmt(acts.mean(0)) if n > 0 else None,
                "std":  fmt(acts.std(0))  if n > 0 else None,
            }
        results["by_dist"] = dist_stats

    # ── F. state_self field sanity ──
    ss0 = state_self[0, 0]  # agent 0, first step   (already squeezed)
    results["state_self_sample"] = {
        "dim": int(D_ss),
        "slot_13_16_assigned_wp": ss0[13:16].tolist(),
        "slot_16_19_close_wp":    ss0[16:19].tolist(),
        "slot_19_close_trig":     round(float(ss0[19]), 4),
        "slot_20_23_role_onehot": ss0[20:23].tolist(),
    }

    # ── G. cooperation field sanity ──
    if has_coop:
        coop0 = coop[0, 0]  # already squeezed
        D_c = int(coop0.shape[0])
        results["cooperation_sample"] = {
            "dim": D_c,
            "step0_rel_xyz": coop0[0:3].tolist(),
            "step1_rel_xyz": coop0[3:6].tolist(),
            "step2_rel_xyz": coop0[6:9].tolist(),
            "step3_rel_xyz": coop0[9:12].tolist(),
            "step4_rel_xyz": coop0[12:15].tolist(),
            "slot_15_18":    coop0[15:18].tolist(),
            "slot_18_21":    coop0[18:21].tolist(),
            "slot_21_24":    coop0[21:24].tolist(),
        }

    return results


def aggregate_results(per_chunk: list) -> dict:
    """Weighted aggregation across chunks."""
    total_ta = sum(r["T"] * r["A"] for r in per_chunk)

    def wavg_stat(key_path):
        """key_path: e.g. ('action', 'mean_per_dim')"""
        vals, weights = [], []
        for r in per_chunk:
            d = r
            for k in key_path[:-1]:
                d = d.get(k, {})
            v = d.get(key_path[-1], None)
            w = r["T"] * r["A"]
            if v is not None:
                vals.append((torch.tensor(v), w))
        if not vals:
            return None
        total_w = sum(w for _, w in vals)
        agg = sum(v * w / total_w for v, w in vals)
        return fmt(agg)

    agg = {
        "total_step_agent": total_ta,
        "n_chunks": len(per_chunk),
        "label_types": list(set(r["label_type"] for r in per_chunk)),
    }

    # action
    agg["action_min"] = round(min(r["action"]["min"] for r in per_chunk), 4)
    agg["action_max"] = round(max(r["action"]["max"] for r in per_chunk), 4)
    agg["action_mean"] = wavg_stat(("action", "mean_per_dim"))
    agg["action_std"]  = wavg_stat(("action", "std_per_dim"))
    agg["action_oor_pct"] = round(sum(r["action"]["pct_out_of_range"] * r["T"] * r["A"] for r in per_chunk) / total_ta, 3)

    # role
    role_names = ["chaser", "front_left", "front_right"]
    agg["by_role"] = {}
    for name in role_names:
        ns = [r["by_role"][name]["n"] for r in per_chunk if name in r.get("by_role", {})]
        total_n = sum(ns)
        means = [(torch.tensor(r["by_role"][name]["mean"]), r["by_role"][name]["n"])
                 for r in per_chunk if r.get("by_role", {}).get(name, {}).get("mean") is not None]
        stds  = [(torch.tensor(r["by_role"][name]["std"]),  r["by_role"][name]["n"])
                 for r in per_chunk if r.get("by_role", {}).get(name, {}).get("std") is not None]
        if means:
            wmean = sum(m * n / total_n for m, n in means)
            wstd  = sum(s * n / total_n for s, n in stds)
        else:
            wmean = wstd = None
        agg["by_role"][name] = {
            "n": total_n,
            "pct": round(total_n / total_ta * 100, 1),
            "mean": fmt(wmean) if wmean is not None else None,
            "std":  fmt(wstd)  if wstd is not None else None,
        }

    # close
    for key in ["close", "normal"]:
        pass  # similar to role; include in by_close
    agg["by_close"] = {}
    for key in ["close", "normal"]:
        ns = [r["by_close"][key]["n"] for r in per_chunk if key in r.get("by_close", {})]
        total_n = sum(ns)
        means = [(torch.tensor(r["by_close"][key]["mean"]), r["by_close"][key]["n"])
                 for r in per_chunk if r.get("by_close", {}).get(key, {}).get("mean") is not None]
        stds  = [(torch.tensor(r["by_close"][key]["std"]),  r["by_close"][key]["n"])
                 for r in per_chunk if r.get("by_close", {}).get(key, {}).get("std") is not None]
        if means:
            wmean = sum(m * n / total_n for m, n in means)
            wstd  = sum(s * n / total_n for s, n in stds)
        else:
            wmean = wstd = None
        agg["by_close"][key] = {
            "n": total_n,
            "pct": round(total_n / total_ta * 100, 1),
            "mean": fmt(wmean) if wmean is not None else None,
            "std":  fmt(wstd)  if wstd is not None else None,
        }

    return agg


# ────────────────────────────────────────────────────────────────────────
# Print report
# ────────────────────────────────────────────────────────────────────────

def print_report(agg: dict, per_chunk: list):
    sep = "=" * 72
    print(f"\n{sep}")
    print("  BC DATASET DIAGNOSTIC REPORT")
    print(sep)
    print(f"  Chunks      : {agg['n_chunks']}")
    print(f"  Total (step×agent): {agg['total_step_agent']:,}")
    print(f"  label_types : {agg['label_types']}")

    print(f"\n{'─'*60}")
    print("  A. ACTION VALUE RANGE")
    print(f"{'─'*60}")
    print(f"  Global min={agg['action_min']:.4f}  max={agg['action_max']:.4f}")
    print(f"  |a|>1.0 out-of-range: {agg['action_oor_pct']:.3f}%  ← 应=0% (pidrate_normalized)")
    print(f"  Mean per dim {DIM_NAMES}: {agg['action_mean']}")
    print(f"  Std  per dim {DIM_NAMES}: {agg['action_std']}")
    
    # 关键判断：如果 std 很小说明 label 方差本来不大，或者 actor 学到了均值
    stds = agg["action_std"]
    if stds:
        avg_std = sum(stds) / len(stds)
        print(f"  mean(std_all_dims) = {avg_std:.4f}")
        if avg_std < 0.1:
            print(f"  ⚠️ WARNING: overall std < 0.1 — expert data may be low-variance (confirm per-role)")

    print(f"\n{'─'*60}")
    print("  B. ACTION BY ROLE")
    print(f"{'─'*60}")
    for name in ["chaser", "front_left", "front_right"]:
        s = agg["by_role"].get(name, {})
        n, pct = s.get("n", 0), s.get("pct", 0)
        mean, std = s.get("mean"), s.get("std")
        print(f"  [{name}]  n={n:,} ({pct}%)")
        if mean:
            print(f"    mean {DIM_NAMES}: {mean}")
            print(f"    std  {DIM_NAMES}: {std}")
            if std:
                low_std = [i for i, v in enumerate(std) if v < 0.05]
                if low_std:
                    dims = [DIM_NAMES[i] for i in low_std]
                    print(f"    ⚠️  LOW STD dims: {dims}  ← actor 很难学习这些维度的多样性")

    print(f"\n{'─'*60}")
    print("  C. ACTION BY CLOSE MODE")
    print(f"{'─'*60}")
    for key in ["normal", "close"]:
        s = agg["by_close"].get(key, {})
        n, pct = s.get("n", 0), s.get("pct", 0)
        mean, std = s.get("mean"), s.get("std")
        print(f"  [{key}]  n={n:,} ({pct}%)")
        if mean:
            print(f"    mean {DIM_NAMES}: {mean}")
            print(f"    std  {DIM_NAMES}: {std}")

    print(f"\n{'─'*60}")
    print("  D. ACTION BY TIME STEP (chunk 0 only)")
    print(f"{'─'*60}")
    if per_chunk:
        r0 = per_chunk[0]
        for tkey in ["step_0_100", "step_100_300", "step_300p"]:
            ts = r0.get("by_time", {}).get(tkey, {})
            n = ts.get("n", 0)
            print(f"  [{tkey}]  n={n:,}")
            if ts.get("mean"):
                print(f"    mean: {ts['mean']}")
                print(f"    std:  {ts['std']}")

    print(f"\n{'─'*60}")
    print("  E. ACTION BY DISTANCE TO TARGET (chunk 0, approx)")
    print(f"{'─'*60}")
    if per_chunk and "by_dist" in per_chunk[0]:
        for dkey in ["dist_gt1.5", "dist_0.8_1.5", "dist_0.4_0.8", "dist_le0.4"]:
            ds = per_chunk[0]["by_dist"].get(dkey, {})
            n, pct = ds.get("n", 0), ds.get("pct", 0)
            mean, std = ds.get("mean"), ds.get("std")
            print(f"  [{dkey}]  n={n:,} ({pct}%)")
            if mean:
                print(f"    mean: {mean}")
                print(f"    std:  {std}")

    print(f"\n{'─'*60}")
    print("  F. STATE_SELF FIELD VERIFICATION (chunk 0, step 0, agent 0)")
    print(f"{'─'*60}")
    if per_chunk and "state_self_sample" in per_chunk[0]:
        ss = per_chunk[0]["state_self_sample"]
        print(f"  dim={ss['dim']}")
        print(f"  [13:16] assigned_wp = {ss['slot_13_16_assigned_wp']}")
        print(f"  [16:19] close_wp    = {ss['slot_16_19_close_wp']}")
        print(f"  [19]    close_trig  = {ss['slot_19_close_trig']}")
        print(f"  [20:23] role_onehot = {ss['slot_20_23_role_onehot']}")
        role_hot = ss["slot_20_23_role_onehot"]
        if role_hot:
            role_idx = int(max(range(3), key=lambda i: role_hot[i]))
            print(f"          → role = {['chaser','front_left','front_right'][role_idx]}")

    print(f"\n{'─'*60}")
    print("  G. COOPERATION FIELD VERIFICATION (chunk 0, step 0, agent 0)")
    print(f"{'─'*60}")
    if per_chunk and "cooperation_sample" in per_chunk[0]:
        cs = per_chunk[0]["cooperation_sample"]
        print(f"  dim={cs['dim']}")
        for k, v in cs.items():
            if k != "dim":
                print(f"  {k}: {v}")

    print(f"\n{'─'*60}")
    print("  KEY CONCLUSIONS")
    print(f"{'─'*60}")
    oor = agg["action_oor_pct"]
    if oor > 0.1:
        print(f"  🔴 action_raw 有 {oor:.2f}% 超出 [-1,1] → label 不是 pidrate_normalized！")
    else:
        print(f"  ✅ action_raw 全部在 [-1,1]，label type 正确")

    for name in ["chaser", "front_left", "front_right"]:
        s = agg["by_role"].get(name, {})
        std = s.get("std")
        if std:
            low = [DIM_NAMES[i] for i, v in enumerate(std) if v < 0.05]
            if low:
                print(f"  ⚠️  {name} 的 {low} 维度 std<0.05 → 专家在这些维度几乎不变化（难学/无信息）")

    close_n = agg["by_close"].get("close", {}).get("n", 0)
    normal_n = agg["by_close"].get("normal", {}).get("n", 0)
    total_cn = close_n + normal_n
    close_pct = close_n / max(total_cn, 1) * 100
    if close_pct < 5:
        print(f"  ⚠️  close 模式数据仅 {close_pct:.1f}% → close_trigger 很少触发，BC 几乎没有此模式样本")
    else:
        print(f"  ✅ close 模式占 {close_pct:.1f}%")
    print(sep)


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--out_dir", default="analysis/bc_diag_gpu")
    parser.add_argument("--arena_size", type=float, default=3.5)
    args = parser.parse_args()

    # 如果 CUDA 不可用 fallback
    if not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU")
        args.device = "cpu"
    else:
        print(f"[INFO] Using device: {args.device} ({torch.cuda.get_device_name(args.device)})")

    os.makedirs(args.out_dir, exist_ok=True)

    chunk_files = sorted(glob.glob(os.path.join(args.dataset_dir, "expert_success_wave_*.pt")))
    if not chunk_files:
        print(f"[ERROR] No chunks found in {args.dataset_dir}")
        sys.exit(1)
    print(f"[INFO] Found {len(chunk_files)} chunk(s)")

    per_chunk = []
    for i, path in enumerate(chunk_files):
        print(f"\n[Chunk {i+1}/{len(chunk_files)}]")
        chunk = load_chunk_fast(path, args.device)
        stats = analyze_chunk(chunk, args.device, arena_size=args.arena_size)
        per_chunk.append(stats)
        del chunk
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    agg = aggregate_results(per_chunk)
    print_report(agg, per_chunk)

    # Save
    out_path = os.path.join(args.out_dir, "diag_agg.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg, "per_chunk": per_chunk}, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] Saved to {out_path}")


if __name__ == "__main__":
    main()
