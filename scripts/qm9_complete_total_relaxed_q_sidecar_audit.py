#!/usr/bin/env python3
"""Build strict relaxed scalar-curvature sidecars from existing train100 audits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_difference(first: float, second: float, floor: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), floor)


def assess_directional_q(
    branches: list[dict[str, Any]], gate: dict[str, Any]
) -> dict[str, Any]:
    if not branches:
        raise ValueError("at least one branch is required")
    floor = float(gate["curvature_floor_hartree_per_bohr2"])
    selected_step = float(gate["selected_force_secant_step_bohr"])
    small_steps = np.asarray(gate["small_steps_bohr"], dtype=np.float64)
    baseline_values = []
    pbe_values = []
    step_spreads = []
    implicit_differences = []
    closure_differences = []
    reference_direction = np.asarray(branches[0]["direction"], dtype=np.float64)
    direction_mismatch = 0.0
    for branch in branches:
        direction = np.asarray(branch["direction"], dtype=np.float64)
        direction_mismatch = max(
            direction_mismatch,
            float(np.max(np.abs(direction - reference_direction))),
        )
        steps = np.asarray(branch["curvature_steps_bohr"], dtype=np.float64)
        scan = np.asarray(branch["relaxed_hvp_step_scan"], dtype=np.float64)
        scalar_scan = np.einsum("ij,kij->k", direction, scan)
        small_indices = [
            int(np.argmin(np.abs(steps - requested))) for requested in small_steps
        ]
        if any(not math.isclose(float(steps[index]), float(requested), rel_tol=0.0, abs_tol=1e-12)
               for index, requested in zip(small_indices, small_steps, strict=True)):
            raise ValueError("required small curvature step is missing")
        selected_index = int(np.argmin(np.abs(steps - selected_step)))
        if not math.isclose(
            float(steps[selected_index]), selected_step, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("selected force-secant step is missing")
        selected_q = float(scalar_scan[selected_index])
        small_q = scalar_scan[small_indices]
        small_center = float(np.median(small_q))
        step_spreads.append(
            float(np.max(np.abs(small_q - small_center)))
            / max(abs(small_center), floor)
        )
        implicit_q = float(
            np.sum(direction * np.asarray(branch["implicit_hvp"], dtype=np.float64))
        )
        pbe_q = float(
            np.sum(direction * np.asarray(branch["pbe_hvp"], dtype=np.float64))
        )
        baseline_values.append(selected_q)
        pbe_values.append(pbe_q)
        implicit_differences.append(
            _relative_difference(selected_q, implicit_q, floor)
        )
        closure_differences.append(
            _relative_difference(
                float(branch["energy_curvature_closure"]),
                float(branch["force_curvature_closure"]),
                floor,
            )
        )
    baseline_mean = float(np.mean(baseline_values))
    pbe_mean = float(np.mean(pbe_values))
    branch_spread = (max(baseline_values) - min(baseline_values)) / max(
        abs(baseline_mean), floor
    )
    pbe_branch_spread = (max(pbe_values) - min(pbe_values)) / max(
        abs(pbe_mean), floor
    )
    checks = {
        "direction_match": direction_mismatch
        <= float(gate["maximum_direction_mismatch"]),
        "small_step_stability": max(step_spreads)
        <= float(gate["maximum_small_step_relative_spread"]),
        "branch_stability": branch_spread
        <= float(gate["maximum_branch_relative_spread"]),
        "implicit_relaxed_agreement": max(implicit_differences)
        <= float(gate["maximum_implicit_relaxed_relative_difference"]),
        "energy_force_closure": max(closure_differences)
        <= float(gate["maximum_energy_force_relative_difference"]),
        "pbe_branch_match": pbe_branch_spread <= 1.0e-10,
    }
    finite_values = [
        baseline_mean,
        pbe_mean,
        branch_spread,
        pbe_branch_spread,
        direction_mismatch,
        *step_spreads,
        *implicit_differences,
        *closure_differences,
    ]
    checks["finite"] = all(math.isfinite(value) for value in finite_values)
    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "baseline_q_hartree_per_bohr2": baseline_mean,
        "pbe_q_hartree_per_bohr2": pbe_mean,
        "correction_q_hartree_per_bohr2": pbe_mean - baseline_mean,
        "branch_relative_spread": branch_spread,
        "pbe_branch_relative_spread": pbe_branch_spread,
        "small_step_relative_spread_max": max(step_spreads),
        "implicit_relaxed_relative_difference_max": max(implicit_differences),
        "energy_force_relative_difference_max": max(closure_differences),
        "direction_mismatch_max": direction_mismatch,
    }


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = yaml.safe_load(path.read_text())
    if protocol.get("test100_access_allowed") is not False:
        raise ValueError("protocol does not freeze Test100")
    if int(protocol.get("test100_evaluations_used", -1)) != 0:
        raise ValueError("protocol Test100 count is not zero")
    for name, value in protocol["inputs"].items():
        if not name.endswith("_sha256"):
            continue
        source_name = name.removesuffix("_sha256")
        source = Path(protocol["inputs"][source_name])
        if _sha256(source) != str(value):
            raise ValueError(f"input hash drift: {source_name}")
    selection = json.loads(Path(protocol["inputs"]["selection_manifest"]).read_text())
    if selection.get("test_accessed_for_selection") is not False:
        raise ValueError("selection manifest accessed Test100")
    if selection.get("parent_overlap") != {
        "train_test": [],
        "train_val": [],
        "val_test": [],
    }:
        raise ValueError("selection parent grouping or disjointness drift")
    tasks = json.loads(Path(protocol["inputs"]["tasks_json"]).read_text())
    if tasks.get("test100_accessed") is not False or tasks.get("subset") not in {
        "file",
        "train",
    }:
        raise ValueError("task inventory is not frozen train-only data")
    if int(tasks.get("molecule_count", -1)) != 100 or int(tasks.get("task_count", -1)) != 1200:
        raise ValueError("unexpected train100 task inventory size")
    return protocol


def _read_parent_gates(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 100:
        raise ValueError("parent stability CSV must contain 100 parents")
    return {str(row["molecule_id"]): row for row in rows}


def _read_tasks(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 1200:
        raise ValueError("task TSV must contain 1200 tasks")
    return rows


def _closure_values(path: Path, requested_step: float) -> tuple[float, float]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [
        row
        for row in rows
        if math.isclose(
            float(row["step_bohr"]), requested_step, rel_tol=0.0, abs_tol=1e-12
        )
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one scalar closure row at {requested_step}: {path}")
    return float(matches[0]["energy_curvature"]), float(matches[0]["force_curvature"])


def _load_task_payload(
    task_root: Path, row: dict[str, Any], closure_step: float
) -> dict[str, Any]:
    index = int(row["task_index"])
    molecule_id = str(row["molecule_id"])
    direction_index = int(row["direction_index"])
    branch = str(row["branch"])
    task_dir = task_root / f"task_{index:04d}_{molecule_id}_d{direction_index}_{branch}"
    arrays = list(task_dir.glob("*_hvp_arrays.npz"))
    if len(arrays) != 1:
        raise ValueError(f"missing or ambiguous HVP arrays: {task_dir}")
    energy_q, force_q = _closure_values(task_dir / "curvature_scan.csv", closure_step)
    with np.load(arrays[0]) as payload:
        return {
            "molecule_id": molecule_id,
            "direction_index": direction_index,
            "branch": branch,
            "atomic_numbers": np.asarray(payload["atomic_numbers"], dtype=np.int64),
            "positions_bohr": np.asarray(payload["positions_bohr"], dtype=np.float64),
            "direction": np.asarray(payload["direction"], dtype=np.float64),
            "direction_kind": str(np.asarray(payload["direction_kind"]).item()),
            "curvature_steps_bohr": np.asarray(
                payload["curvature_steps_bohr"], dtype=np.float64
            ),
            "relaxed_hvp_step_scan": np.asarray(
                payload["relaxed_hvp_step_scan"], dtype=np.float64
            ),
            "implicit_hvp": np.asarray(payload["implicit_hvp"], dtype=np.float64),
            "pbe_hvp": np.asarray(payload["pbe_hvp"], dtype=np.float64),
            "energy_curvature_closure": energy_q,
            "force_curvature_closure": force_q,
        }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _load_protocol(args.protocol)
    inputs = protocol["inputs"]
    gate = protocol["numerical_gate"]
    expected_branches = list(gate["branches"])
    parent_gates = _read_parent_gates(Path(inputs["parent_stability_csv"]))
    tasks = _read_tasks(Path(inputs["tasks_tsv"]))
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    task_root = Path(inputs["task_root"])
    closure_step = float(gate["scalar_energy_closure_step_bohr"])
    for row in tasks:
        grouped[(str(row["molecule_id"]), int(row["direction_index"]))].append(
            _load_task_payload(task_root, row, closure_step)
        )
    if len(grouped) != 400:
        raise ValueError("expected 400 parent-direction groups")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sidecar_dir = args.output_dir / "sidecars"
    sidecar_dir.mkdir(exist_ok=True)
    direction_rows: list[dict[str, Any]] = []
    parent_payloads: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for (molecule_id, direction_index), branches in sorted(grouped.items()):
        branches.sort(key=lambda row: expected_branches.index(row["branch"]))
        if [row["branch"] for row in branches] != expected_branches:
            raise ValueError(f"branch inventory drift: {molecule_id} direction {direction_index}")
        parent_gate_failures = str(parent_gates[molecule_id]["parent_gate_failures"])
        parent_gate_passed = parent_gate_failures == ""
        metrics = assess_directional_q(branches, gate)
        eligible = parent_gate_passed and bool(metrics["eligible"])
        failed_checks = [
            name for name, passed in metrics["checks"].items() if not passed
        ]
        direction_rows.append(
            {
                "molecule_id": molecule_id,
                "direction_index": direction_index,
                "direction_kind": branches[0]["direction_kind"],
                "natoms": len(branches[0]["atomic_numbers"]),
                "parent_gate_passed": parent_gate_passed,
                "parent_gate_failures": parent_gate_failures,
                "eligible": eligible,
                "failed_checks": ";".join(failed_checks),
                **{key: value for key, value in metrics.items() if key not in {"checks", "eligible"}},
            }
        )
        parent_payloads[molecule_id].append((branches[0], {**metrics, "eligible": eligible}))

    minimum = int(gate["minimum_eligible_directions_per_parent"])
    parent_rows = []
    manifest_entries = []
    for molecule_id, rows in sorted(parent_payloads.items()):
        rows.sort(key=lambda item: int(item[0]["direction_index"]))
        mask = np.asarray([bool(item[1]["eligible"]) for item in rows], dtype=bool)
        parent_eligible = bool(np.count_nonzero(mask) >= minimum)
        parent_rows.append(
            {
                "molecule_id": molecule_id,
                "natoms": len(rows[0][0]["atomic_numbers"]),
                "parent_gate_passed": str(parent_gates[molecule_id]["parent_gate_failures"]) == "",
                "eligible_direction_count": int(np.count_nonzero(mask)),
                "minimum_eligible_directions": minimum,
                "parent_eligible": parent_eligible,
            }
        )
        if not parent_eligible:
            continue
        path = sidecar_dir / f"{molecule_id}.0000000.npz"
        np.savez_compressed(
            path,
            atomic_numbers=rows[0][0]["atomic_numbers"],
            positions_bohr=rows[0][0]["positions_bohr"],
            direction=np.stack([item[0]["direction"] for item in rows]),
            direction_kind=np.asarray([item[0]["direction_kind"] for item in rows]),
            direction_index=np.asarray([item[0]["direction_index"] for item in rows]),
            baseline_q_hartree_per_bohr2=np.asarray(
                [item[1]["baseline_q_hartree_per_bohr2"] for item in rows]
            ),
            pbe_q_hartree_per_bohr2=np.asarray(
                [item[1]["pbe_q_hartree_per_bohr2"] for item in rows]
            ),
            correction_q_hartree_per_bohr2=np.asarray(
                [item[1]["correction_q_hartree_per_bohr2"] for item in rows]
            ),
            eligibility_mask=mask,
            parent_eligibility=np.asarray(parent_eligible),
        )
        manifest_entries.append(
            {
                "molecule_id": molecule_id,
                "filename": path.name,
                "sha256": _sha256(path),
                "eligible_direction_count": int(np.count_nonzero(mask)),
                "eligibility_mask": mask.astype(int).tolist(),
            }
        )

    _write_csv(args.output_dir / "per_direction_q_stability.csv", direction_rows)
    _write_csv(args.output_dir / "per_parent_q_stability.csv", parent_rows)
    manifest = {
        "definition": protocol["definition"],
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "entries": manifest_entries,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    eligible_direction_count = sum(bool(row["eligible"]) for row in direction_rows)
    eligible_parent_count = len(manifest_entries)
    target_count = int(protocol["decision"]["train100_q_ready_parent_count"])
    result = {
        "definition": protocol["definition"],
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "source_task_count": len(tasks),
        "source_parent_count": len(parent_rows),
        "source_direction_count": len(direction_rows),
        "parent_density_branch_kkt_gate_pass_count": sum(
            bool(row["parent_gate_passed"]) for row in parent_rows
        ),
        "eligible_direction_count": eligible_direction_count,
        "eligible_parent_count": eligible_parent_count,
        "eligible_parent_target_count": target_count,
        "train100_q_ready": eligible_parent_count >= target_count,
        "manifest": manifest_path.resolve().as_posix(),
        "manifest_sha256": _sha256(manifest_path),
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    audit(parse_args())
