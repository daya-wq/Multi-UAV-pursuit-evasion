"""
Lightweight diagnostic: Why actor fails in closed-loop.
Reads expert dataset (raw dict format) and actor checkpoint.
"""
import os, sys, glob, torch, numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATASET_DIR = os.path.join(PROJECT_ROOT, "expert_datasets/expert2_step5_aligned_histfix_1024x5_20260415_154654")
ACTOR_CKPT = os.path.join(PROJECT_ROOT, "checkpoints/dagger_expert2_v2_20260417_014816/dagger_best.pt")

def main():
    # --- 1. Load one chunk ---
    fs = sorted(glob.glob(os.path.join(DATASET_DIR, "expert_success_wave_*.pt")))
    print(f"Dataset: {len(fs)} chunks")
    chunk = torch.load(fs[0], map_location="cpu", weights_only=False)
    print(f"Chunk keys: {list(chunk.keys())}")
    
    # Key tensors
    action_raw = chunk["action_raw"].float()     # expert actions
    pidrate = chunk.get("pidrate_normalized", action_raw).float()
    obs = chunk["obs"]                            # dict of obs components
    ep_lens = chunk["episode_lengths"]
    
    print(f"\naction_raw: {action_raw.shape} {action_raw.dtype}")
    print(f"pidrate_normalized: {pidrate.shape}")
    print(f"episode_lengths: {ep_lens[:10]}... (total {len(ep_lens)} eps, sum={sum(ep_lens)})")
    
    for k, v in obs.items():
        if hasattr(v, 'shape'):
            print(f"obs[{k}]: {v.shape} {v.dtype}")
    
    # === ANALYSIS 1: Expert action statistics ===
    print(f"\n{'='*65}")
    print(f"  1. EXPERT ACTION STATISTICS  (N={pidrate.shape[0]} steps)")
    print(f"{'='*65}")
    
    # pidrate: [N, n_agents, 4] — thrust, roll_rate, pitch_rate, yaw_rate
    axes = ["thrust", "roll_rate", "pitch_rate", "yaw_rate"]
    for i, ax in enumerate(axes):
        a = pidrate[:, :, i].reshape(-1)
        print(f"  {ax:12s}: mean={a.mean():.4f}  std={a.std():.4f}  "
              f"|mean|={a.abs().mean():.4f}  Q10={a.quantile(0.1):.4f}  Q90={a.quantile(0.9):.4f}")
    
    # Action norm
    norm = pidrate.norm(dim=-1).reshape(-1)
    print(f"  action_norm:  mean={norm.mean():.4f}  std={norm.std():.4f}")
    print(f"  near_zero (norm<0.05): {float((norm < 0.05).float().mean()):.4f}")
    
    # === ANALYSIS 2: Distance-based action analysis ===
    print(f"\n{'='*65}")
    print(f"  2. OBS STRUCTURE & DISTANCE ANALYSIS")
    print(f"{'='*65}")
    
    # obs["state_self"]: [N, n_agents, 1, 23] or [N, n_agents, 23]
    obs_self = obs["state_self"].float()
    obs_coop = obs["cooperation"].float()
    if obs_self.ndim == 4:
        obs_self = obs_self.squeeze(-2)  # [N, n_agents, 23]
    if obs_coop.ndim == 4:
        obs_coop = obs_coop.squeeze(-2)  # [N, n_agents, 24]
    
    # self pos: [:, :, 0:3], cooperation target_pred_step1: [:, :, 3:6]
    drone_pos = obs_self[:, :, 0:3]
    drone_vel = obs_self[:, :, 3:6]
    assigned_wp = obs_self[:, :, 13:16]
    close_trigger = obs_self[:, :, 19]
    role_onehot = obs_self[:, :, 20:23]
    
    target_vel = obs_coop[:, :, 0:3]
    target_pred1 = obs_coop[:, :, 3:6]   # first step prediction ≈ current target pos
    forward_dir = obs_coop[:, :, 18:21]
    lateral_dir = obs_coop[:, :, 21:24]
    
    # Distance from each drone to target
    dist = (drone_pos - target_pred1).norm(dim=-1)  # [N, n_agents]
    
    print(f"  dist_to_target: mean={dist.mean():.3f}  std={dist.std():.3f}  "
          f"min={dist.min():.3f}  max={dist.max():.3f}")
    print(f"  close_triggered: frac={float((close_trigger > 0).float().mean()):.4f}")
    
    role_ids = role_onehot.argmax(dim=-1)
    for r in range(3):
        print(f"  role_{r} frac: {float((role_ids==r).float().mean()):.3f}")
    
    # === Distance-bucketed action analysis ===
    print(f"\n{'='*65}")
    print(f"  3. EXPERT ACTIONS BY DISTANCE (key for understanding)")
    print(f"{'='*65}")
    
    buckets = [
        ("<0.3m capture", 0, 0.3),
        ("0.3-0.5m close", 0.3, 0.5),
        ("0.5-1.0m", 0.5, 1.0),
        ("1.0-2.0m", 1.0, 2.0),
        ("2.0-3.5m approach", 2.0, 3.5),
        (">3.5m far", 3.5, 20),
    ]
    
    dist_flat = dist.reshape(-1)
    act_flat = pidrate.reshape(-1, 4)
    
    bucket_stats = {}
    for name, lo, hi in buckets:
        mask = (dist_flat >= lo) & (dist_flat < hi)
        n = mask.sum().item()
        frac = n / dist_flat.numel()
        if n < 10:
            print(f"\n  {name}: {n} frames (skip)")
            continue
        a = act_flat[mask]
        print(f"\n  {name}: {n} frames ({100*frac:.1f}%)")
        for i, ax in enumerate(axes):
            val = a[:, i]
            print(f"    {ax:12s}: mean={val.mean():.4f}  std={val.std():.4f}  |mean|={val.abs().mean():.4f}")
        ans = a.norm(dim=-1)
        print(f"    norm:        mean={ans.mean():.4f}  std={ans.std():.4f}")
        bucket_stats[name] = {
            "frac": frac,
            "n": n,
            "yaw_mean": float(a[:, 3].mean()),
            "yaw_std": float(a[:, 3].std()),
            "yaw_abs_mean": float(a[:, 3].abs().mean()),
            "pitch_abs_mean": float(a[:, 2].abs().mean()),
            "norm_mean": float(ans.mean()),
        }
    
    # === ANALYSIS 4: Per-episode action variability ===
    print(f"\n{'='*65}")
    print(f"  4. ACTION VARIABILITY WITHIN vs BETWEEN EPISODES")
    print(f"{'='*65}")
    
    offset = 0
    within_vars = []
    between_means = []
    for ep_len in ep_lens[:100]:  # first 100 episodes
        ep_act = pidrate[offset:offset+ep_len]  # [T, n_agents, 4]
        offset += ep_len
        within_vars.append(ep_act.var(dim=0).mean().item())
        between_means.append(ep_act.mean(dim=0))
    
    within_var = np.mean(within_vars)
    between_means_t = torch.stack(between_means)
    between_var = between_means_t.var(dim=0).mean().item()
    
    print(f"  Within-episode variance (avg):  {within_var:.6f}")
    print(f"  Between-episode variance:       {between_var:.6f}")
    print(f"  Ratio (between/within):         {between_var/max(within_var, 1e-8):.4f}")
    print(f"  -> {'HIGH between-episode diversity' if between_var > within_var * 0.5 else 'LOW between-episode diversity (most var is within-episode)'}")
    
    # === ANALYSIS 5: MSE if actor outputs global mean ===
    print(f"\n{'='*65}")
    print(f"  5. ERROR SIMULATION: WHAT IF ACTOR OUTPUTS GLOBAL MEAN?")
    print(f"{'='*65}")
    
    global_mean = pidrate.reshape(-1, 4).mean(dim=0)
    print(f"  Global mean action: {global_mean.tolist()}")
    
    # Error at each distance bucket
    for name, lo, hi in buckets:
        mask = (dist_flat >= lo) & (dist_flat < hi)
        if mask.sum() < 10:
            continue
        a = act_flat[mask]
        err = (a - global_mean.unsqueeze(0)).pow(2).mean(dim=-1).sqrt()
        print(f"  {name}: rmse_if_mean={err.mean():.4f}  vs actual_norm={a.norm(dim=-1).mean():.4f}  "
              f"rel_err={err.mean()/a.norm(dim=-1).mean().clamp_min(0.01):.3f}")
    
    # === ANALYSIS 6: Compounding error ===
    print(f"\n{'='*65}")
    print(f"  6. COMPOUNDING ERROR: ACTION CHANGE RATES")
    print(f"{'='*65}")
    
    # How fast do expert actions change? If actor is 1 step behind, how big is the error?
    delta1 = (pidrate[1:] - pidrate[:-1]).abs()
    delta5 = (pidrate[5:] - pidrate[:-5]).abs() if pidrate.shape[0] > 5 else None
    delta10 = (pidrate[10:] - pidrate[:-10]).abs() if pidrate.shape[0] > 10 else None
    
    for label, delta in [("1-step", delta1), ("5-step", delta5), ("10-step", delta10)]:
        if delta is None:
            continue
        d = delta.reshape(-1, 4)
        print(f"  {label} Δaction:")
        for i, ax in enumerate(axes):
            print(f"    {ax:12s}: mean={d[:,i].mean():.4f}  std={d[:,i].std():.4f}")
    
    # === ANALYSIS 7: Multimodality ===
    print(f"\n{'='*65}")
    print(f"  7. MULTIMODALITY CHECK (kurtosis)")
    print(f"{'='*65}")
    
    for name, lo, hi in [("<0.5m", 0, 0.5), ("0.5-1m", 0.5, 1.0), ("1-2m", 1, 2), (">3m", 3, 20)]:
        mask = (dist_flat >= lo) & (dist_flat < hi)
        if mask.sum() < 100:
            continue
        a = act_flat[mask]
        print(f"\n  {name}:")
        for i, ax in enumerate(axes):
            v = a[:, i]
            s = v.std().clamp_min(1e-6)
            z = (v - v.mean()) / s
            kurt = float(z.pow(4).mean() - 3)
            skew = float(z.pow(3).mean())
            print(f"    {ax:12s}: kurtosis={kurt:.2f}  skew={skew:.2f}  "
                  f"frac_near_zero(|v|<0.05)={float((v.abs()<0.05).float().mean()):.3f}")
    
    # === ANALYSIS 8: Actor checkpoint ===
    print(f"\n{'='*65}")
    print(f"  8. ACTOR CHECKPOINT ANALYSIS")
    print(f"{'='*65}")
    
    if os.path.isfile(ACTOR_CKPT):
        ckpt = torch.load(ACTOR_CKPT, map_location="cpu", weights_only=False)
        
        # log_std
        for k in ckpt:
            if "log_std" in k:
                v = ckpt[k]
                print(f"  {k}: mean={v.mean():.4f}  min={v.min():.4f}  max={v.max():.4f}  "
                      f"exp(mean)={v.mean().exp():.6f}")
        
        # Last layer bias (reveals systematic action output bias)
        actor_keys = sorted(k for k in ckpt if "actor" in k and "opt" not in k and "weight" in k)
        last_w = [k for k in actor_keys if 'weight' in k]
        if last_w:
            lk = last_w[-1]
            bk = lk.replace("weight", "bias")
            print(f"\n  Last actor weight: {lk} shape={ckpt[lk].shape}")
            if bk in ckpt:
                print(f"  Last actor bias ({bk}): {ckpt[bk].data.tolist()}")
                print(f"  -> tanh(bias): {ckpt[bk].data.tanh().tolist()}")
    
    # === ANALYSIS 9: Final diagnosis ===
    print(f"\n{'='*65}")
    print(f"  9. ROOT CAUSE DIAGNOSIS")
    print(f"{'='*65}")
    
    # Compute key statistics
    yaw_overall_abs = pidrate[:,:,3].abs().mean().item()
    pitch_overall_abs = pidrate[:,:,2].abs().mean().item()
    yaw_far_abs = bucket_stats.get(">3.5m far", {}).get("yaw_abs_mean", 0)
    yaw_close_abs = bucket_stats.get("<0.3m capture", {}).get("yaw_abs_mean", 0)
    
    print(f"\n  Key findings:")
    print(f"  - Expert yaw |mean| overall: {yaw_overall_abs:.4f}")
    print(f"  - Expert yaw |mean| at >3.5m: {yaw_far_abs:.4f}")
    print(f"  - Expert yaw |mean| at <0.3m: {yaw_close_abs:.4f}")
    print(f"  - Yaw dynamic range (far/close): {yaw_far_abs/max(yaw_close_abs,0.001):.2f}x")
    print(f"  - Expert action changes rapidly: 1-step Δyaw mean = {delta1[:,:,3].mean():.4f}")
    print(f"  - MSE global mean prediction vs actual: shows how much actor loses by averaging")
    print(f"  - Multimodality: check kurtosis values above")

if __name__ == "__main__":
    main()
