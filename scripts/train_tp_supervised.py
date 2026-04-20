import argparse
import csv
import datetime
import glob
import json
import math
import os
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from omni_drones.learning import TP_net


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class IndexedTPDataset(Dataset):
    def __init__(self, tp_input: torch.Tensor, tp_future: torch.Tensor, indices: torch.Tensor):
        self.tp_input = tp_input
        self.tp_future = tp_future
        self.indices = indices.to(dtype=torch.long)

    def __len__(self):
        return int(self.indices.numel())

    def __getitem__(self, idx):
        base_idx = int(self.indices[idx].item())
        return self.tp_input[base_idx], self.tp_future[base_idx]


def denormalize_future(x: torch.Tensor, arena_size: float, max_height: float) -> torch.Tensor:
    y = x.clone()
    y[..., :2] = y[..., :2] * float(arena_size)
    y[..., 2] = (y[..., 2] + 1.0) * 0.5 * float(max_height)
    return y


def evaluate(model, loader, device, arena_size: float, max_height: float) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    ade_sum = 0.0
    fde_sum = 0.0
    horizon_err_sum = None

    with torch.no_grad():
        for tp_input, tp_future in loader:
            tp_input = tp_input.to(device=device, dtype=torch.float32, non_blocking=True)
            tp_future = tp_future.to(device=device, dtype=torch.float32, non_blocking=True)
            pred = model(tp_input).reshape(tp_future.shape[0], tp_future.shape[1], tp_future.shape[2])
            loss = F.mse_loss(pred, tp_future, reduction="mean")

            pred_world = denormalize_future(pred, arena_size, max_height)
            gt_world = denormalize_future(tp_future, arena_size, max_height)
            err = torch.norm(pred_world - gt_world, dim=-1)

            batch_size = int(tp_input.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
            ade_sum += float(err.mean(dim=-1).sum().item())
            fde_sum += float(err[:, -1].sum().item())
            step_err = err.sum(dim=0).cpu()
            if horizon_err_sum is None:
                horizon_err_sum = step_err
            else:
                horizon_err_sum += step_err

    if total_count <= 0:
        return {
            "loss": float("nan"),
            "ade_m": float("nan"),
            "fde_m": float("nan"),
        }

    metrics = {
        "loss": total_loss / total_count,
        "ade_m": ade_sum / total_count,
        "fde_m": fde_sum / total_count,
    }
    if horizon_err_sum is not None:
        for idx, value in enumerate(horizon_err_sum.tolist()):
            metrics[f"h{idx+1}_err_m"] = float(value) / total_count
    return metrics


def append_csv(csv_path: str, row: Dict[str, float], fieldnames: List[str]):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_summary_md(
    path: str,
    args,
    metadata: Dict[str, float],
    best_metrics: Dict[str, float],
    chunk_files: List[str],
):
    lines = [
        "# TP预测网络监督训练说明",
        "",
        "## 1. 数据采集口径",
        f"- 数据目录：`{args.dataset_dir}`",
        f"- 波次数：`{len(chunk_files)}`",
        f"- 并行环境：`{metadata['batch_envs']}`",
        f"- 历史长度：`{metadata['history_step']}` 步",
        f"- 预测长度：`{metadata['future_predcition_step']}` 步",
        f"- window_step：`{metadata['window_step']}`",
        f"- 追捕方速度：`{metadata['v_drone_test']}` m/s",
        f"- 目标速度：`{metadata['v_prey_test']}` m/s",
        f"- 仿真步长：`{metadata['dt']}` s",
        f"- 单回合步长上限：`{metadata['episode_length']}`",
        f"- 位置归一化：`x,y / arena_size`，`z -> z/max_height*2-1`",
        "",
        "## 2. 网络结构",
        "- 直接复用仓库里的 `omni_drones.learning.TP_net`",
        f"- 输入：`[history_step, input_dim] = [{metadata['history_step']}, {metadata['input_dim']}]`",
        "- 主干：`1层 LSTM(hidden_dim=64)`",
        f"- 输出：`future_step * 3 = {metadata['future_predcition_step']} * 3`",
        "- 输出激活：`tanh`，对应归一化后的未来 5 步目标位置",
        "",
        "## 3. 损失函数",
        "- 主损失：归一化坐标上的 MSE",
        "- 形式：`Loss = mean((TP_net(TP_input) - TP_future.reshape(B, -1))^2)`",
        "- 评估指标：`val_loss`、`ADE(m)`、`FDE(m)`、每个预测步的平均位置误差",
        "",
        "## 4. 训练参数",
        f"- batch size：`{args.batch_size}`",
        f"- epochs：`{args.epochs}`",
        f"- learning rate：`{args.lr}`",
        f"- weight decay：`{args.weight_decay}`",
        f"- val_ratio：`{args.val_ratio}`",
        f"- num_workers：`{args.num_workers}`",
        f"- device：`{args.device}`",
        "",
        "## 5. 当前最佳结果",
        f"- best_epoch：`{best_metrics.get('best_epoch', 'n/a')}`",
        f"- val_loss：`{best_metrics.get('best_val_loss', float('nan')):.6f}`",
        f"- val_ADE：`{best_metrics.get('best_val_ade_m', float('nan')):.4f} m`",
        f"- val_FDE：`{best_metrics.get('best_val_fde_m', float('nan')):.4f} m`",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--save_tag", default="tp_supervised")
    parser.add_argument("--analysis_dir", default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(int(args.seed))

    chunk_files = sorted(glob.glob(os.path.join(args.dataset_dir, "tp_wave_*.pt")))
    if not chunk_files:
        raise FileNotFoundError(f"No TP chunks found in {args.dataset_dir}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = args.save_dir
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(os.getcwd(), save_dir)
    save_dir = os.path.join(save_dir, args.save_tag)
    os.makedirs(save_dir, exist_ok=True)

    analysis_dir = args.analysis_dir or os.path.join(os.getcwd(), "analysis", f"{args.save_tag}_{timestamp}")
    if not os.path.isabs(analysis_dir):
        analysis_dir = os.path.join(os.getcwd(), analysis_dir)
    os.makedirs(analysis_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(analysis_dir, "tensorboard"))

    chunk_inputs = []
    chunk_targets = []
    metadata = None
    total_samples = 0
    for chunk_path in chunk_files:
        chunk = torch.load(chunk_path, map_location="cpu")
        tp_input = chunk["TP_input"].contiguous()
        tp_future = chunk["TP_future"].contiguous()
        if tp_input.shape[0] <= 0:
            continue
        chunk_inputs.append(tp_input)
        chunk_targets.append(tp_future)
        total_samples += int(tp_input.shape[0])
        if metadata is None:
            metadata = {
                "history_step": int(chunk["history_step"]),
                "future_predcition_step": int(chunk["future_predcition_step"]),
                "window_step": int(chunk["window_step"]),
                "input_dim": int(chunk["input_dim"]),
                "arena_size": float(chunk["arena_size"]),
                "max_height": float(chunk["max_height"]),
                "batch_envs": int(chunk["batch_envs"]),
                "episode_length": int(chunk["episode_length"]),
                "dt": float(chunk["dt"]),
                "v_drone_test": float(chunk["v_drone_test"]),
                "v_prey_test": float(chunk["v_prey_test"]),
            }

    if not chunk_inputs:
        raise RuntimeError(f"TP dataset is empty: {args.dataset_dir}")

    tp_input = torch.cat(chunk_inputs, dim=0)
    tp_future = torch.cat(chunk_targets, dim=0)

    num_samples = int(tp_input.shape[0])
    perm = torch.randperm(num_samples)
    num_val = max(1, int(num_samples * float(args.val_ratio)))
    val_idx = perm[:num_val]
    train_idx = perm[num_val:]
    if train_idx.numel() <= 0:
        raise RuntimeError("No training samples left after validation split.")

    train_dataset = IndexedTPDataset(tp_input, tp_future, train_idx)
    val_dataset = IndexedTPDataset(tp_input, tp_future, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=max(1, int(args.num_workers) // 2),
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(args.device)
    model = TP_net(
        input_dim=int(metadata["input_dim"]),
        output_dim=int(metadata["future_predcition_step"]) * 3,
        future_predcition_step=int(metadata["future_predcition_step"]),
        window_step=int(metadata["window_step"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    csv_path = os.path.join(analysis_dir, "train_metrics.csv")
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_ade_m",
        "val_fde_m",
    ] + [f"val_h{i}_err_m" for i in range(1, int(metadata["future_predcition_step"]) + 1)]

    best_val_loss = float("inf")
    best_metrics = {}

    print(
        f"[TP train] samples={num_samples} train={len(train_dataset)} val={len(val_dataset)} "
        f"input={tuple(tp_input.shape[1:])} future={tuple(tp_future.shape[1:])} device={device}"
    )

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch_input, batch_future in train_loader:
            batch_input = batch_input.to(device=device, dtype=torch.float32, non_blocking=True)
            batch_future = batch_future.to(device=device, dtype=torch.float32, non_blocking=True)
            pred = model(batch_input).reshape(batch_future.shape[0], batch_future.shape[1], batch_future.shape[2])
            loss = F.mse_loss(pred, batch_future, reduction="mean")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = int(batch_input.shape[0])
            train_loss_sum += float(loss.item()) * batch_size
            train_count += batch_size

        train_loss = train_loss_sum / max(train_count, 1)
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            arena_size=float(metadata["arena_size"]),
            max_height=float(metadata["max_height"]),
        )

        writer.add_scalar("tp/train_loss", train_loss, epoch)
        writer.add_scalar("tp/val_loss", val_metrics["loss"], epoch)
        writer.add_scalar("tp/val_ade_m", val_metrics["ade_m"], epoch)
        writer.add_scalar("tp/val_fde_m", val_metrics["fde_m"], epoch)
        for i in range(1, int(metadata["future_predcition_step"]) + 1):
            key = f"h{i}_err_m"
            if key in val_metrics:
                writer.add_scalar(f"tp/val_{key}", val_metrics[key], epoch)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_ade_m": val_metrics["ade_m"],
            "val_fde_m": val_metrics["fde_m"],
        }
        for i in range(1, int(metadata["future_predcition_step"]) + 1):
            key = f"h{i}_err_m"
            row[f"val_h{i}_err_m"] = val_metrics.get(key, float("nan"))
        append_csv(csv_path, row, fieldnames)

        ckpt_path = os.path.join(save_dir, f"tp_epoch_{epoch:03d}.pt")
        torch.save(model.state_dict(), ckpt_path)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            best_metrics = {
                "best_epoch": epoch,
                "best_val_loss": best_val_loss,
                "best_val_ade_m": float(val_metrics["ade_m"]),
                "best_val_fde_m": float(val_metrics["fde_m"]),
            }
            best_path = os.path.join(save_dir, f"tp_only_{timestamp}.pt")
            torch.save(model.state_dict(), best_path)

        print(
            f"[TP train] epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_ADE={val_metrics['ade_m']:.4f}m "
            f"val_FDE={val_metrics['fde_m']:.4f}m"
        )

    final_path = os.path.join(save_dir, "tp_final.pt")
    torch.save(model.state_dict(), final_path)

    summary = {
        "dataset_dir": args.dataset_dir,
        "num_chunks": len(chunk_files),
        "num_samples": num_samples,
        "num_train": len(train_dataset),
        "num_val": len(val_dataset),
        "save_dir": save_dir,
        "analysis_dir": analysis_dir,
        **metadata,
        **best_metrics,
    }
    with open(os.path.join(analysis_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_summary_md(
        os.path.join(analysis_dir, "TP预测网络训练说明.md"),
        args,
        metadata,
        best_metrics,
        chunk_files,
    )
    writer.close()
    print(f"[TP train] best={best_metrics}")
    print(f"[TP train] summary={os.path.join(analysis_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
