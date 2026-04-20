#!/usr/bin/env python3
"""
快速核查脚本：只读一个 chunk，检查关键诊断点
- action_raw 值域、统计
- state_self 各字段含义确认（用几个样本值打印）
- cooperation 字段结构
- tanh/no-tanh 下标签值域是否合理
"""
import torch
import sys
import math

dataset_dir = "expert_datasets/expert2_step5_aligned_histfix_1024x5_20260415_154654"
chunk_path = f"{dataset_dir}/expert_success_wave_00001.pt"

print(f"Loading {chunk_path} ...")
chunk = torch.load(chunk_path, map_location="cpu", weights_only=False)
print(f"Chunk keys: {list(chunk.keys())}")
print(f"action_label_type: {chunk.get('action_label_type', 'MISSING')}")

action_raw = chunk["action_raw"].float()
print(f"\n[ACTION_RAW] shape={action_raw.shape}")
print(f"  min={action_raw.min():.4f}  max={action_raw.max():.4f}")
print(f"  mean per dim: {action_raw.reshape(-1,4).mean(0).tolist()}")
print(f"  std  per dim: {action_raw.reshape(-1,4).std(0).tolist()}")
print(f"  |action|>1.0: {(action_raw.abs()>1.0).float().mean().item()*100:.2f}% (should be 0% for pidrate_normalized)")

obs = chunk["obs"]
print(f"\n[OBS KEYS]: {sorted(obs.keys())}")

if "state_self" in obs:
    ss = obs["state_self"].float()
    print(f"\n[STATE_SELF] shape={ss.shape}")
    # 打印前 3 个 step 的 agent0 数据，分段解析
    s0 = ss[0, 0]  # first step, agent 0
    print(f"  [0] total_dim={s0.shape[0]}")
    print(f"  state_self[0:3]   (pos?)          = {s0[0:3].tolist()}")
    print(f"  state_self[3:6]   (vel?)           = {s0[3:6].tolist()}")
    print(f"  state_self[6:9]   (quat?/other?)   = {s0[6:9].tolist()}")
    print(f"  state_self[9:12]  (omega?)         = {s0[9:12].tolist()}")
    print(f"  state_self[12:13] (time?)          = {s0[12:13].tolist()}")
    print(f"  state_self[13:16] (assigned_wp?)   = {s0[13:16].tolist()}")
    print(f"  state_self[16:19] (close_wp?)      = {s0[16:19].tolist()}")
    print(f"  state_self[19:20] (close_trigger?) = {s0[19:20].tolist()}")
    print(f"  state_self[20:23] (role_onehot?)   = {s0[20:23].tolist()}")
    # check if 23 == total dim
    if s0.shape[0] > 23:
        print(f"  state_self[23:]   = {s0[23:].tolist()}")

    # 统计 role distribution
    role = ss[..., 20:23].argmax(-1)  # [T, A]
    total = role.numel()
    for i, name in enumerate(["chaser", "front_left", "front_right"]):
        n = (role == i).sum().item()
        print(f"  role={name}: {n}/{total} ({n/total*100:.1f}%)")

    # close trigger stats
    close_t = ss[..., 19:20].squeeze(-1)  # [T, A]
    n_close = (close_t > 0.5).sum().item()
    print(f"  close_trigger>0.5: {n_close}/{total} ({n_close/total*100:.1f}%)")

if "cooperation" in obs:
    coop = obs["cooperation"].float()
    print(f"\n[COOPERATION] shape={coop.shape}")
    # cooperation dim from doc: usually target pos pred * future_steps + forward_dir + ...
    c0 = coop[0, 0]
    print(f"  dim={c0.shape[0]}")
    # first 3 = target_pos_pred step0 (relative, normalized)
    # next 3 = step1 ... step{future_steps}
    # then forward_dir
    print(f"  cooperation[0:3]  (target_pos_pred_step0?): {c0[0:3].tolist()}")
    print(f"  cooperation[3:6]  (step1?):                 {c0[3:6].tolist()}")
    print(f"  cooperation[6:9]  (step2?):                 {c0[6:9].tolist()}")
    print(f"  cooperation[9:12] (step3?):                 {c0[9:12].tolist()}")
    print(f"  cooperation[12:15](step4?):                 {c0[12:15].tolist()}")
    print(f"  cooperation[15:18](step5?):                 {c0[15:18].tolist()}")
    print(f"  cooperation[18:21](forward_dir?):           {c0[18:21].tolist()}")
    if c0.shape[0] > 21:
        print(f"  cooperation[21:] : {c0[21:].tolist()}")
    # norm of target pred positions
    for step in range(6):
        seg = c0[step*3:(step+1)*3]
        norm = seg.norm().item()
        print(f"  step{step} pred norm={norm:.4f}  values={[round(v.item(),3) for v in seg]}")

# expert_aux
if "expert_aux" in chunk:
    aux = chunk["expert_aux"]
    print(f"\n[EXPERT_AUX] keys: {list(aux.keys())}")
    for k, v in aux.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}  dtype={v.dtype}")
            if v.is_floating_point():
                vf = v.float()
                print(f"       min={vf.min():.4f}  max={vf.max():.4f}  mean={vf.mean():.4f}")

# episode info
ep_lens = chunk["episode_lengths"]
print(f"\n[EPISODES] n_episodes={len(ep_lens)}  mean_len={ep_lens.float().mean():.1f}  total_steps={ep_lens.sum().item()}")
print(f"  len distribution: min={ep_lens.min().item()}  max={ep_lens.max().item()}")

# action_raw detailed Stats - by role
print(f"\n[ACTION PER ROLE]")
ss_all = obs["state_self"].float()  # [T, A, 23]
role_all = ss_all[..., 20:23].argmax(-1)  # [T, A]
act_all = chunk["action_raw"].float()  # [T, A, 4]
for i, name in enumerate(["chaser", "front_left", "front_right"]):
    mask = (role_all == i)  # [T, A]
    acts = act_all[mask]    # [N, 4]
    if acts.shape[0] > 0:
        print(f"  {name}: n={acts.shape[0]:,}  mean={acts.mean(0).tolist()}  std={acts.std(0).tolist()}")
        print(f"           min={acts.min(0).values.tolist()}  max={acts.max(0).values.tolist()}")

# Compare action stats of close vs no-close
print(f"\n[ACTION CLOSE vs NORMAL]")
close_all = (ss_all[..., 19] > 0.5)  # [T, A]
for flag, name in [(True, "close"), (False, "normal")]:
    mask = (close_all == flag)
    acts = act_all[mask]
    if acts.shape[0] > 0:
        print(f"  {name}: n={acts.shape[0]:,}  mean={[round(v,4) for v in acts.mean(0).tolist()]}  std={[round(v,4) for v in acts.std(0).tolist()]}")

print("\nDone.")
