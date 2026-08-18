#!/usr/bin/env python3
"""Analyze paired-volume XWM single points into post-hoc pressures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


ETOT_RE = re.compile(r"!\s*FINAL_ETOT_IS\s+([-+0-9.eE]+)\s+eV")
KB_EV_K = 8.617333262145e-5
EV_A3_TO_KBAR = 1602.176634


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def block_se(values: list[float], block_size: int = 5) -> float:
    blocks = [values[index : index + block_size] for index in range(0, len(values), block_size)]
    means = [statistics.mean(block) for block in blocks if len(block) == block_size]
    if len(means) < 2:
        return math.nan
    return statistics.stdev(means) / math.sqrt(len(means))


def parse_energy(job: Path) -> tuple[float, Path]:
    logs = list(job.glob("OUT.*/running_scf.log"))
    if len(logs) != 1:
        raise ValueError(f"expected one running_scf.log in {job}, found {len(logs)}")
    matches = ETOT_RE.findall(logs[0].read_text(encoding="utf-8", errors="replace"))
    if len(matches) != 1:
        raise ValueError(f"expected one FINAL_ETOT_IS in {logs[0]}, found {len(matches)}")
    return float(matches[0]), logs[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.out or root / "analysis").resolve()
    out.mkdir(parents=True, exist_ok=True)

    expected = json.loads((root / "PREPARATION_MANIFEST.json").read_text())
    jobs = expected["jobs"]
    if len(jobs) != expected["singlepoint_jobs"]:
        raise ValueError("manifest job count mismatch")

    pairs: dict[tuple[str, str, int], dict[str, dict]] = defaultdict(dict)
    output_hashes: list[tuple[str, str]] = []
    for row in jobs:
        job = root / row["job"]
        if not (job / "sp.done").exists():
            raise ValueError(f"incomplete job: {job}")
        if (job / "sp.failed").exists():
            raise ValueError(f"failed marker present: {job}")
        exit_codes = (job / "exit_code.txt").read_text().split()
        if exit_codes != ["0"]:
            raise ValueError(f"nonzero or ambiguous exit code in {job}: {exit_codes}")
        metadata = json.loads((job / "metadata.json").read_text())
        energy_ev, log = parse_energy(job)
        output_hashes.append((sha256(log), str(log.relative_to(root))))
        key = (row["phase"], row["volume_label"], int(row["step"]))
        pairs[key][row["sign"]] = {"energy_ev": energy_ev, "metadata": metadata, "log": str(log.relative_to(root))}

    frame_rows: list[dict] = []
    for (phase, label, step), pair in sorted(pairs.items()):
        if set(pair) != {"minus", "plus"}:
            raise ValueError(f"incomplete finite-difference pair: {(phase, label, step)}")
        minus, plus = pair["minus"], pair["plus"]
        m0, m1 = minus["metadata"], plus["metadata"]
        for key in ("frame_audit_sha256", "source_temperature_k", "source_volume_a3", "epsilon"):
            if m0[key] != m1[key]:
                raise ValueError(f"pair metadata mismatch for {key}: {(phase, label, step)}")
        epsilon = float(m0["epsilon"])
        base_volume = float(m0["source_volume_a3"])
        temperature = float(m0["source_temperature_k"])
        natoms = int(m0["natoms"])
        d_u_d_v = (plus["energy_ev"] - minus["energy_ev"]) / (2.0 * epsilon * base_volume)
        p_conf = -d_u_d_v * EV_A3_TO_KBAR
        p_ideal = natoms * KB_EV_K * temperature / base_volume * EV_A3_TO_KBAR
        frame_rows.append(
            {
                "phase": phase,
                "volume_label": label,
                "step": step,
                "temperature_k": temperature,
                "volume_a3": base_volume,
                "volume_per_atom_a3": base_volume / natoms,
                "epsilon": epsilon,
                "energy_minus_ev": minus["energy_ev"],
                "energy_plus_ev": plus["energy_ev"],
                "p_conf_kbar": p_conf,
                "p_ideal_kbar": p_ideal,
                "p_total_kbar": p_conf + p_ideal,
            }
        )

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in frame_rows:
        grouped[(row["phase"], row["volume_label"])].append(row)
    points: list[dict] = []
    for (phase, label), rows in sorted(grouped.items()):
        rows.sort(key=lambda item: item["step"])
        pressures = [float(item["p_total_kbar"]) for item in rows]
        split = len(pressures) // 2
        points.append(
            {
                "phase": phase,
                "volume_label": label,
                "volume_per_atom_a3": statistics.mean(float(item["volume_per_atom_a3"]) for item in rows),
                "frames": len(rows),
                "temperature_mean_k": statistics.mean(float(item["temperature_k"]) for item in rows),
                "pressure_mean_kbar": statistics.mean(pressures),
                "pressure_sd_kbar": statistics.stdev(pressures),
                "pressure_block_se_kbar": block_se(pressures),
                "pressure_half_drift_kbar": abs(statistics.mean(pressures[:split]) - statistics.mean(pressures[split:])),
                "pressure_min_kbar": min(pressures),
                "pressure_max_kbar": max(pressures),
            }
        )

    roots: dict[str, dict | None] = {}
    for phase in ("solid", "liquid"):
        phase_points = sorted((point for point in points if point["phase"] == phase), key=lambda item: item["volume_per_atom_a3"])
        bracket = None
        for left, right in zip(phase_points, phase_points[1:]):
            p0, p1 = left["pressure_mean_kbar"], right["pressure_mean_kbar"]
            if p0 == 0 or p0 * p1 <= 0:
                v0, v1 = left["volume_per_atom_a3"], right["volume_per_atom_a3"]
                zero_volume = v0 - p0 * (v1 - v0) / (p1 - p0)
                bracket = {"volume_per_atom_a3": zero_volume, "left": left, "right": right}
                break
        roots[phase] = bracket

    with (out / "frame_pressures.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)
    report = {
        "schema": "xwm-numerical-virial-analysis-v1",
        "status": "zero_pressure_bracketed" if all(roots.values()) else "zero_pressure_not_bracketed",
        "jobs_verified": len(jobs),
        "pairs_verified": len(pairs),
        "points": points,
        "roots": roots,
        "input_manifest_sha256": sha256(root / "PREPARATION_MANIFEST.json"),
    }
    report_path = out / "numerical_pressure_summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_hashes.extend([(sha256(out / "frame_pressures.csv"), "analysis/frame_pressures.csv"), (sha256(report_path), "analysis/numerical_pressure_summary.json")])
    (out / "OUTPUT_SHA256SUMS").write_text("".join(f"{digest}  {path}\n" for digest, path in sorted(output_hashes, key=lambda item: item[1])), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
