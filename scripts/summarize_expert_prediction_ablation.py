#!/usr/bin/env python3
"""Parse expert prediction ablation logs and emit csv/json/markdown summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


def _maybe_float(value: str) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _parse_avg_line(line: str, prefix: str) -> dict[str, float]:
    payload = line.split(":", 1)[1].strip()
    result: dict[str, float] = {}
    for part in payload.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        key, raw = part.split("=", 1)
        value = _maybe_float(raw.strip())
        if value is not None:
            result[f"{prefix}_{key.strip()}"] = value
    return result


def parse_log(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    row: dict[str, Any] = {
        "log_path": str(log_path),
        "requested_mode": log_path.stem,
        "actual_mode": None,
        "episodes": None,
        "capture_count": None,
        "goal_count": None,
        "landed_count": None,
        "timeout_count": None,
        "capture_rate": None,
        "goal_rate": None,
        "landed_rate": None,
        "timeout_rate": None,
        "capture_steps_mean": None,
        "capture_steps_std": None,
        "capture_steps_min": None,
        "capture_steps_max": None,
        "expert_pred_pos_err": None,
        "expert_pred_next_err": None,
    }

    mode_match = re.search(r"pred_mode=([a-z_]+)", text)
    if mode_match:
        row["actual_mode"] = mode_match.group(1)

    episodes_match = re.search(r"Episodes\s*:\s*(\d+)", text)
    if episodes_match:
        row["episodes"] = int(episodes_match.group(1))

    for key, pattern in {
        "capture_count": r"Capture\s*:\s*\d+%\s*\((\d+)\)",
        "goal_count": r"Goal zone\s*:\s*\d+%\s*\((\d+)\)",
        "landed_count": r"Landed\s*:\s*\d+%\s*\((\d+)\)",
        "timeout_count": r"Timeout\s*:\s*\d+%\s*\((\d+)\)",
    }.items():
        match = re.search(pattern, text)
        if match:
            row[key] = int(match.group(1))

    pred_match = re.search(
        r"Expert pred\s*:\s*pos_err=([0-9.]+),\s*next_err=([0-9.]+)",
        text,
    )
    if pred_match:
        row["expert_pred_pos_err"] = float(pred_match.group(1))
        row["expert_pred_next_err"] = float(pred_match.group(2))

    capture_steps_match = re.search(
        r"Capture steps\s*:\s*mean=([0-9.]+),\s*std=([0-9.]+),\s*min=(\d+),\s*max=(\d+)",
        text,
    )
    if capture_steps_match:
        row["capture_steps_mean"] = float(capture_steps_match.group(1))
        row["capture_steps_std"] = float(capture_steps_match.group(2))
        row["capture_steps_min"] = int(capture_steps_match.group(3))
        row["capture_steps_max"] = int(capture_steps_match.group(4))

    for line in text.splitlines():
        if "Timeout avg" in line:
            row.update(_parse_avg_line(line, "timeout"))
        elif "Capture avg" in line:
            row.update(_parse_avg_line(line, "capture"))

    episodes = row.get("episodes")
    if episodes:
        for outcome in ("capture", "goal", "landed", "timeout"):
            count = row.get(f"{outcome}_count")
            if count is not None:
                row[f"{outcome}_rate"] = count / episodes

    return row


def _format_scalar(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        return f"{value:.{digits}f}"
    return str(value)


def _format_percent(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and math.isnan(value):
        return "-"
    return f"{100.0 * float(value):.2f}%"


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown(
    rows: list[dict[str, Any]],
    out_path: Path,
    *,
    v_prey: float,
    v_drone: float,
    episode_length: int,
    batch_envs: int,
    num_waves: int,
) -> None:
    noise_baseline = next((row for row in rows if row.get("requested_mode") == "noise"), None)
    noise_capture_rate = noise_baseline.get("capture_rate") if noise_baseline else None

    lines = [
        "# Expert Prediction Ablation",
        "",
        f"- `v_prey={v_prey}`",
        f"- `v_drone={v_drone}`",
        f"- `episode_length={episode_length}`",
        f"- `generic_batch_envs={batch_envs}`",
        f"- `num_waves={num_waves}`",
        "",
        "| requested | actual | episodes | capture | goal | landed | timeout | cap_steps_mean | pred_pos_err | pred_next_err | delta_vs_noise | log |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for row in rows:
        capture_rate = row.get("capture_rate")
        delta = None
        if noise_capture_rate is not None and capture_rate is not None:
            delta = capture_rate - noise_capture_rate
        lines.append(
            "| "
            + " | ".join(
                [
                    _format_scalar(row.get("requested_mode"), 0),
                    _format_scalar(row.get("actual_mode"), 0),
                    _format_scalar(row.get("episodes"), 0),
                    _format_percent(row.get("capture_rate")),
                    _format_percent(row.get("goal_rate")),
                    _format_percent(row.get("landed_rate")),
                    _format_percent(row.get("timeout_rate")),
                    _format_scalar(row.get("capture_steps_mean"), 1),
                    _format_scalar(row.get("expert_pred_pos_err"), 4),
                    _format_scalar(row.get("expert_pred_next_err"), 4),
                    ("-" if delta is None else f"{100.0 * delta:+.2f} pp"),
                    f"[log]({Path(row['log_path']).name})",
                ]
            )
            + " |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize expert prediction ablation logs.")
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--v_prey", type=float, required=True)
    parser.add_argument("--v_drone", type=float, required=True)
    parser.add_argument("--episode_length", type=int, required=True)
    parser.add_argument("--batch_envs", type=int, required=True)
    parser.add_argument("--num_waves", type=int, required=True)
    parser.add_argument("--modes", nargs="+", required=True)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for mode in args.modes:
        log_path = log_dir / f"{mode}.log"
        if not log_path.is_file():
            raise FileNotFoundError(f"Missing log for mode {mode}: {log_path}")
        rows.append(parse_log(log_path))

    csv_path = out_dir / "summary.csv"
    json_path = out_dir / "summary.json"
    md_path = out_dir / "summary.md"

    write_csv(rows, csv_path)
    write_json(rows, json_path)
    write_markdown(
        rows,
        md_path,
        v_prey=args.v_prey,
        v_drone=args.v_drone,
        episode_length=args.episode_length,
        batch_envs=args.batch_envs,
        num_waves=args.num_waves,
    )

    print(f"[summary] csv  : {csv_path}")
    print(f"[summary] json : {json_path}")
    print(f"[summary] md   : {md_path}")


if __name__ == "__main__":
    main()
