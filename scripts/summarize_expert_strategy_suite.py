#!/usr/bin/env python3
"""Summarize a curated set of expert-eval logs into csv/json/markdown tables."""

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


def parse_log(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    row: dict[str, Any] = {
        "log_path": str(log_path),
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
        "any_collision_rate": None,
        "drone_collision_rate": None,
    }

    patterns = {
        "episodes": r"Episodes\s*:\s*(\d+)",
        "capture_count": r"Capture\s*:\s*\d+%\s*\((\d+)\)",
        "goal_count": r"Goal zone\s*:\s*\d+%\s*\((\d+)\)",
        "landed_count": r"Landed\s*:\s*\d+%\s*\((\d+)\)",
        "timeout_count": r"Timeout\s*:\s*\d+%\s*\((\d+)\)",
        "any_collision_count": r"Any coll\s*:\s*\d+%\s*\((\d+)\)",
        "drone_collision_count": r"Drone coll\s*:\s*\d+%\s*\((\d+)\)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            row[key] = int(match.group(1))

    capture_steps_match = re.search(
        r"Capture steps\s*:\s*mean=([0-9.]+),\s*std=([0-9.]+),\s*min=(\d+),\s*max=(\d+)",
        text,
    )
    if capture_steps_match:
        row["capture_steps_mean"] = float(capture_steps_match.group(1))
        row["capture_steps_std"] = float(capture_steps_match.group(2))
        row["capture_steps_min"] = int(capture_steps_match.group(3))
        row["capture_steps_max"] = int(capture_steps_match.group(4))

    episodes = row.get("episodes")
    if episodes:
        for name in ("capture", "goal", "landed", "timeout"):
            count = row.get(f"{name}_count")
            if count is not None:
                row[f"{name}_rate"] = count / episodes
        for name in ("any_collision", "drone_collision"):
            count = row.get(f"{name}_count")
            if count is not None:
                row[f"{name}_rate"] = count / episodes
    return row


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        return f"{value * 100:.1f}%"
    return str(value)


def _fmt_scalar(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Format: label=/abs/path/to/log",
    )
    parser.add_argument("--title", default="Expert Strategy Summary")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for spec in args.case:
        if "=" not in spec:
            raise ValueError(f"Invalid --case '{spec}', expected label=/path/to/log")
        label, raw_path = spec.split("=", 1)
        row = parse_log(Path(raw_path))
        row["strategy"] = label
        rows.append(row)

    csv_path = out_dir / "summary.csv"
    json_path = out_dir / "summary.json"
    md_path = out_dir / "summary.md"

    fieldnames = [
        "strategy",
        "episodes",
        "capture_rate",
        "capture_count",
        "goal_rate",
        "goal_count",
        "landed_rate",
        "landed_count",
        "timeout_rate",
        "timeout_count",
        "any_collision_rate",
        "any_collision_count",
        "drone_collision_rate",
        "drone_collision_count",
        "capture_steps_mean",
        "capture_steps_std",
        "capture_steps_min",
        "capture_steps_max",
        "log_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [f"# {args.title}", ""]
    if args.notes:
        lines.append(args.notes)
        lines.append("")
    lines.append(
        "| strategy | episodes | capture | goal | landed | timeout | any coll | drone coll | first_capture_step_mean | log |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["strategy"]),
                    str(row.get("episodes") or "-"),
                    _fmt_pct(row.get("capture_rate")),
                    _fmt_pct(row.get("goal_rate")),
                    _fmt_pct(row.get("landed_rate")),
                    _fmt_pct(row.get("timeout_rate")),
                    _fmt_pct(row.get("any_collision_rate")),
                    _fmt_pct(row.get("drone_collision_rate")),
                    _fmt_scalar(row.get("capture_steps_mean"), 1),
                    str(row.get("log_path") or "-"),
                ]
            )
            + " |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
