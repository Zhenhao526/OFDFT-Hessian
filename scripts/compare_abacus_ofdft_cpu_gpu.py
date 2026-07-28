#!/usr/bin/env python3
"""Compare matched CPU and GPU ABACUS OFDFT validation runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from mpn_melting.trajectory import parse_md_dump

ENERGY_RE = re.compile(r"!FINAL_ETOT_IS\s+([-+0-9.eE]+)\s+eV")
REAL_TIME_RE = re.compile(r"^real\s+([-+0-9.eE]+)\s*$", re.MULTILINE)
TN_RE = re.compile(r"^\s*TN\d+\s+", re.MULTILINE)


def only_file(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern} below {root}, found {len(matches)}")
    return matches[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def max_vector_difference(left, right, field: str) -> float:
    left_values = getattr(left, field)
    right_values = getattr(right, field)
    if left_values is None or right_values is None:
        raise ValueError(f"missing {field} in compared frame")
    if len(left_values) != len(right_values):
        raise ValueError(f"different {field} lengths")
    return max(
        (
            abs(left_component - right_component)
            for left_vector, right_vector in zip(left_values, right_values)
            for left_component, right_component in zip(left_vector, right_vector)
        ),
        default=0.0,
    )


def max_lattice_difference(left, right) -> float:
    return max(
        abs(left_component - right_component)
        for left_vector, right_vector in zip(left.lattice, right.lattice)
        for left_component, right_component in zip(left_vector, right_vector)
    )


def read_run(root: Path) -> dict:
    log = only_file(root, "OUT.*/running_md.log")
    dump = only_file(root, "OUT.*/MD_dump")
    stdout = root / "run.stdout"
    log_text = log.read_text(encoding="utf-8", errors="ignore")
    stdout_text = stdout.read_text(encoding="utf-8", errors="ignore")
    energies = [float(value) for value in ENERGY_RE.findall(log_text)]
    runtimes = [float(value) for value in REAL_TIME_RE.findall(stdout_text)]
    frames = parse_md_dump(dump)
    if not energies or not runtimes or not frames:
        raise ValueError(f"incomplete validation output below {root}")
    return {
        "root": str(root.resolve()),
        "log_sha256": sha256(log),
        "dump_sha256": sha256(dump),
        "final_energy_eV": energies[-1],
        "real_time_s": runtimes[-1],
        "tn_iterations": len(TN_RE.findall(stdout_text)),
        "gpu_device_reported": "GPU / NVIDIA" in stdout_text,
        "frame": frames[-1],
    }


def compare_pair(label: str, cpu_root: Path, gpu_root: Path) -> dict:
    cpu = read_run(cpu_root)
    gpu = read_run(gpu_root)
    if cpu["frame"].step != gpu["frame"].step:
        raise ValueError(f"{label}: CPU/GPU final frame steps differ")
    energy_error = abs(cpu["final_energy_eV"] - gpu["final_energy_eV"])
    force_error = max_vector_difference(cpu["frame"], gpu["frame"], "forces")
    position_error = max_vector_difference(cpu["frame"], gpu["frame"], "positions")
    velocity_error = max_vector_difference(cpu["frame"], gpu["frame"], "velocities")
    lattice_error = max_lattice_difference(cpu["frame"], gpu["frame"])
    speedup = cpu["real_time_s"] / gpu["real_time_s"]
    checks = {
        "gpu_device_reported": gpu["gpu_device_reported"],
        "same_tn_iteration_count": cpu["tn_iterations"] == gpu["tn_iterations"],
        "energy_error_le_1e-8_eV_per_system": energy_error <= 1.0e-8,
        "force_error_le_1e-8_eV_per_A": force_error <= 1.0e-8,
        "position_error_le_1e-10_A": position_error <= 1.0e-10,
        "velocity_error_le_1e-10_A_per_fs": velocity_error <= 1.0e-10,
        "lattice_error_le_1e-12_A": lattice_error <= 1.0e-12,
    }
    return {
        "label": label,
        "cpu": {key: value for key, value in cpu.items() if key != "frame"},
        "gpu": {key: value for key, value in gpu.items() if key != "frame"},
        "absolute_differences": {
            "energy_eV_per_system": energy_error,
            "force_eV_per_A": force_error,
            "position_A": position_error,
            "velocity_A_per_fs": velocity_error,
            "lattice_A": lattice_error,
        },
        "cpu_over_gpu_speedup": speedup,
        "performance_improved_by_20_percent": speedup >= 1.2,
        "checks": checks,
        "status": "numerically_verified" if all(checks.values()) else "failed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        metavar=("LABEL", "CPU_RUN", "GPU_RUN"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    comparisons = [
        compare_pair(label, Path(cpu_root), Path(gpu_root))
        for label, cpu_root, gpu_root in args.pair
    ]
    result = {
        "schema": "abacus-ofdft-cpu-gpu-validation-v1",
        "comparisons": comparisons,
        "all_numerically_verified": all(
            row["status"] == "numerically_verified" for row in comparisons
        ),
        "all_performance_improved_by_20_percent": all(
            row["performance_improved_by_20_percent"] for row in comparisons
        ),
    }
    result["recommendation"] = (
        "eligible_for_production_benchmark"
        if result["all_numerically_verified"]
        and result["all_performance_improved_by_20_percent"]
        else "do_not_migrate_production"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
