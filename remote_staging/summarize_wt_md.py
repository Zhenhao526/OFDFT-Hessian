#!/usr/bin/env python3
"""Summarize in-progress ABACUS MD logs without modifying a run."""

from __future__ import annotations

import argparse
import math
import re
import statistics
import time
from pathlib import Path


STEP_RE = re.compile(r"STEP OF MOLECULAR DYNAMICS:\s*(\d+)")
TN_RE = re.compile(r"^\s*TN(\d+)\s")
ITER_RE = re.compile(r"^Iter(\d+):")


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.pstdev(values)


def slope(values: list[float]) -> float:
    if len(values) < 2:
        return math.nan
    x_mean = (len(values) - 1) / 2
    y_mean = statistics.fmean(values)
    denom = sum((i - x_mean) ** 2 for i in range(len(values)))
    return sum((i - x_mean) * (y - y_mean) for i, y in enumerate(values)) / denom


def parse_log(path: Path) -> tuple[list[dict[str, float]], int, dict[int, int]]:
    lines = path.read_text(errors="replace").splitlines()
    current_step = -1
    current_tn_max = -1
    records: list[dict[str, float]] = []
    tn_counts: dict[int, int] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        step_match = STEP_RE.search(line)
        if step_match:
            current_step = int(step_match.group(1))
            current_tn_max = -1
        tn_match = TN_RE.match(line)
        if tn_match and current_step >= 0:
            current_tn_max = max(current_tn_max, int(tn_match.group(1)))
        iter_match = ITER_RE.match(line)
        if iter_match and current_step >= 0:
            current_tn_max = max(current_tn_max, int(iter_match.group(1)))
        if "Temperature (K)" in line and i + 1 < len(lines):
            fields = lines[i + 1].split()
            if len(fields) >= 4:
                try:
                    records.append(
                        {
                            "step": float(current_step),
                            "temperature": float(fields[3]),
                            "pressure": float(fields[4]) if len(fields) >= 5 else math.nan,
                        }
                    )
                    if current_tn_max >= 0:
                        tn_counts[current_step] = current_tn_max + 1
                except ValueError:
                    pass
        i += 1
    return records, current_step, tn_counts


def md_steps(input_path: Path) -> int:
    for line in input_path.read_text(errors="replace").splitlines():
        fields = line.split()
        if fields and fields[0].lower() == "md_nstep" and len(fields) >= 2:
            return int(fields[1])
    return 0


def summarize(run_dir: Path, tail: int) -> None:
    logs = sorted(run_dir.glob("OUT.*/running_md.log"))
    if not logs:
        print(f"RUN {run_dir} status=no_log")
        return
    log = logs[0]
    records, current_step, tn_counts = parse_log(log)
    total = md_steps(run_dir / "INPUT")
    if not records:
        print(f"RUN {run_dir} status=initializing current_step={current_step} total={total}")
        return

    recent = records[-tail:]
    temperatures = [r["temperature"] for r in records]
    pressures = [r["pressure"] for r in records if math.isfinite(r["pressure"])]
    recent_t = [r["temperature"] for r in recent]
    recent_p = [r["pressure"] for r in recent if math.isfinite(r["pressure"])]
    t_mean, t_std = mean_std(temperatures)
    rt_mean, rt_std = mean_std(recent_t)
    p_mean, p_std = mean_std(pressures)
    rp_mean, rp_std = mean_std(recent_p)
    sampled_step = int(records[-1]["step"])
    progress = max(current_step, sampled_step + 1)

    input_info = next(iter(sorted(run_dir.glob("OUT.*/INPUT.info"))), log)
    start = input_info.stat().st_mtime
    elapsed = max(time.time() - start, 1.0)
    sec_per_step = elapsed / max(progress, 1)
    eta_seconds = max(total - progress, 0) * sec_per_step
    completed_tn = [tn_counts[int(r["step"])] for r in records if int(r["step"]) in tn_counts]
    tn_mean = statistics.fmean(completed_tn) if completed_tn else math.nan
    tn_max = max(completed_tn) if completed_tn else -1

    print(f"RUN {run_dir}")
    print(
        "  "
        f"latest_sampled_step={sampled_step} current_step={current_step} total={total} "
        f"records={len(records)} sec_per_step={sec_per_step:.2f} eta_min={eta_seconds / 60:.1f}"
    )
    print(
        "  "
        f"T_all={t_mean:.2f}+/-{t_std:.2f} T_last={temperatures[-1]:.2f} "
        f"T_recent{len(recent_t)}={rt_mean:.2f}+/-{rt_std:.2f} "
        f"T_recent_slope={slope(recent_t):.3f}K/record"
    )
    if pressures:
        print(
            "  "
            f"P_all={p_mean:.3f}+/-{p_std:.3f}kbar P_last={pressures[-1]:.3f}kbar "
            f"P_recent{len(recent_p)}={rp_mean:.3f}+/-{rp_std:.3f}kbar"
        )
    else:
        print("  P=not_recorded")
    print(f"  electronic_iterations_mean={tn_mean:.2f} max={tn_max}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--tail", type=int, default=10)
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        summarize(run_dir, args.tail)


if __name__ == "__main__":
    main()
