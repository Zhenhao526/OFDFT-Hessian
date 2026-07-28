#!/usr/bin/env python3
"""Apply the frozen multi-branch HVP stability gates without consulting Test100."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


BRANCHES = {
    ("configured", "base_continuation"): "sad_continuation",
    ("label_reference", "base_continuation"): "label_continuation",
    ("configured", "configured"): "sad_independent",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_diagnostics(
    output_dir: Path,
    direction_rows: list[dict[str, Any]],
    parent_rows: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for stable, color, label in (
        (True, "#0072B2", "passing direction"),
        (False, "#D55E00", "excluded direction"),
    ):
        selected = [row for row in direction_rows if bool(row["stable"]) is stable]
        if not selected:
            continue
        x = np.asarray(
            [float(row["hvp_step_stability_relative_max"]) for row in selected]
        )
        y = np.asarray(
            [float(row["energy_force_curvature_relative_max"]) for row in selected]
        )
        x = np.nan_to_num(x, nan=1e3, posinf=1e3, neginf=1e-12)
        y = np.nan_to_num(y, nan=1e3, posinf=1e3, neginf=1e-12)
        ax.scatter(np.maximum(x, 1e-12), np.maximum(y, 1e-12), s=24, alpha=0.7, color=color, label=label)
    ax.axvline(0.05, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0.05, color="black", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("HVP step instability (relative)")
    ax.set_ylabel("Energy-force closure mismatch (relative)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "direction_step_vs_closure.png", dpi=180)
    plt.close(fig)

    direction_failures: dict[str, int] = {}
    for row in direction_rows:
        for reason in str(row["failure_reasons"]).split(";"):
            if reason:
                direction_failures[reason] = direction_failures.get(reason, 0) + 1
    parent_failures: dict[str, int] = {}
    for row in parent_rows:
        for reason in str(row["parent_gate_failures"]).split(";"):
            if reason:
                parent_failures[reason] = parent_failures.get(reason, 0) + 1
    reasons = sorted(set(direction_failures) | set(parent_failures))
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    positions = np.arange(len(reasons))
    ax.bar(
        positions - 0.2,
        [direction_failures.get(reason, 0) for reason in reasons],
        0.4,
        color="#0072B2",
        label="direction masks",
    )
    ax.bar(
        positions + 0.2,
        [parent_failures.get(reason, 0) for reason in reasons],
        0.4,
        color="#D55E00",
        label="parent exclusions",
    )
    ax.set_xticks(positions, reasons, rotation=35, ha="right")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "stability_failure_reasons.png", dpi=180)
    plt.close(fig)


def _as_float(value: Any) -> float | None:
    if value in (None, "", "None", "nan"):
        return None
    return float(value)


def _point_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open() as handle:
        return {row["point"]: row for row in csv.DictReader(handle)}


def _branch_name(summary: dict[str, Any]) -> str:
    key = (summary.get("base_initialization"), summary.get("displaced_initialization"))
    if key not in BRANCHES:
        raise ValueError(f"Unregistered initialization branch {key}")
    return BRANCHES[key]


def _load_tasks(
    root: Path | list[Path],
) -> tuple[dict[tuple[str, int], dict[str, dict[str, Any]]], list[str]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    errors = []
    roots = [root] if isinstance(root, Path) else list(root)
    summary_paths = sorted(
        summary_path
        for task_root in roots
        for summary_path in task_root.rglob("summary.json")
    )
    for summary_path in summary_paths:
        try:
            payload = json.loads(summary_path.read_text())
            summaries = payload.get("summaries", [])
            if len(summaries) != 1:
                continue
            summary = summaries[0]
            molecule_id = str(summary["molecule_id"]).zfill(7)
            direction_index = int(summary["direction_index"])
            branch = _branch_name(summary)
            arrays = list(summary_path.parent.glob("*_hvp_arrays.npz"))
            if len(arrays) != 1:
                raise ValueError(f"Expected one HVP array file, found {len(arrays)}")
            points_path = summary_path.parent / "points.csv"
            record = {
                "summary": summary,
                "payload": payload,
                "arrays_path": arrays[0],
                "points": _point_rows(points_path),
                "summary_path": summary_path,
            }
            key = (molecule_id, direction_index)
            if branch in grouped.setdefault(key, {}):
                raise ValueError(f"Duplicate {key} branch {branch}")
            grouped[key][branch] = record
        except Exception as error:  # preserve malformed tasks as audit evidence
            errors.append(f"{summary_path}: {type(error).__name__}: {error}")
    return grouped, errors


def _load_direction_arrays(path: Path) -> dict[str, np.ndarray]:
    required = (
        "base_coulomb_matrix",
        "base_coefficients",
        "relaxed_plus_coefficients",
        "relaxed_minus_coefficients",
        "relaxed_plus_force",
        "relaxed_minus_force",
        "curvature_steps_bohr",
        "relaxed_hvp_step_scan",
        "relaxed_hvp",
        "implicit_hvp",
    )
    with np.load(path) as payload:
        return {key: np.asarray(payload[key]) for key in required}


def _coulomb_metrics(left: np.ndarray, right: np.ndarray, matrix: np.ndarray) -> tuple[float, float]:
    left_norm = np.sqrt(max(float(left @ matrix @ left), 0.0))
    right_norm = np.sqrt(max(float(right @ matrix @ right), 0.0))
    cross = float(left @ matrix @ right)
    cosine = cross / max(left_norm * right_norm, np.finfo(float).tiny)
    difference = left - right
    distance = np.sqrt(max(float(difference @ matrix @ difference), 0.0))
    relative_distance = distance / max(left_norm, right_norm, np.finfo(float).tiny)
    return cosine, relative_distance


def _max_pairwise(values: list[np.ndarray], metric) -> float:
    if len(values) < 2:
        return 0.0
    return max(metric(left, right) for left, right in itertools.combinations(values, 2))


def _gate_direction(
    molecule_id: str,
    direction_index: int,
    records: dict[str, dict[str, Any]],
    gates: dict[str, float],
    strict_hvp_steps: list[float],
    energy_force_curvature_min_step: float,
) -> dict[str, Any]:
    reasons = []
    missing = sorted(set(BRANCHES.values()).difference(records))
    if missing:
        reasons.append("missing_branches:" + ",".join(missing))
    usable = [records[key] for key in sorted(records)]
    if not usable:
        return {
            "molecule_id": molecule_id,
            "direction_index": direction_index,
            "stable": False,
            "failure_reasons": ";".join(reasons or ["no_records"]),
        }

    # Materialize only the three branches for this direction and close each NPZ
    # immediately. Keeping 1200 lazy NpzFile handles exceeds common fd limits.
    usable = [
        {**record, "arrays": _load_direction_arrays(record["arrays_path"])}
        for record in usable
    ]
    first_summary = usable[0]["summary"]
    row: dict[str, Any] = {
        "molecule_id": molecule_id,
        "natoms": int(first_summary["natoms"]),
        "direction_index": direction_index,
        "direction_kind": first_summary["direction_kind"],
        "branch_count": len(usable),
    }

    gradients = []
    constraints = []
    energies_by_point: dict[str, list[float]] = {key: [] for key in ("base", "plus", "minus")}
    coefficients_by_point: dict[str, list[np.ndarray]] = {
        key: [] for key in ("base", "plus", "minus")
    }
    forces_by_point: dict[str, list[np.ndarray]] = {key: [] for key in ("plus", "minus")}
    kkt_minima = []
    kkt_conditions = []
    response_relative = []
    response_stationarity = []
    response_constraint = []
    step_instabilities = []
    implicit_relaxed = []
    curvature_mismatches = []
    wall_times = []
    coulomb = np.asarray(usable[0]["arrays"]["base_coulomb_matrix"], dtype=np.float64)

    for record in usable:
        summary = record["summary"]
        arrays = record["arrays"]
        points = record["points"]
        wall_times.append(float(summary["wall_time_s"]))
        gradients.append(float(summary["base_projected_gradient_norm"]))
        constraints.append(float(summary.get("base_constraint_residual", np.inf)))
        for point_name in ("base", "relaxed_plus", "relaxed_minus"):
            point = points.get(point_name)
            if point is None:
                reasons.append(f"missing_point:{point_name}")
                continue
            gradient = _as_float(point.get("final_gradient_norm"))
            if gradient is not None:
                gradients.append(gradient)
            constraint = _as_float(point.get("constraint_residual"))
            if constraint is not None:
                constraints.append(abs(constraint))
            key = point_name.removeprefix("relaxed_")
            energy = _as_float(point.get("legacy_total_energy"))
            if energy is not None:
                energies_by_point[key].append(energy)
        coefficients_by_point["base"].append(np.asarray(arrays["base_coefficients"]))
        coefficients_by_point["plus"].append(
            np.asarray(arrays["relaxed_plus_coefficients"])
        )
        coefficients_by_point["minus"].append(
            np.asarray(arrays["relaxed_minus_coefficients"])
        )
        forces_by_point["plus"].append(np.asarray(arrays["relaxed_plus_force"]))
        forces_by_point["minus"].append(np.asarray(arrays["relaxed_minus_force"]))

        minimum = summary.get("dense_tangent_eigenvalue_min")
        condition = summary.get("dense_tangent_condition_number")
        if minimum is None or condition is None:
            reasons.append("missing_dense_kkt_spectrum")
        else:
            kkt_minima.append(float(minimum))
            kkt_conditions.append(float(condition))
        response_relative.append(float(summary["response_relative_residual"]))
        response_stationarity.append(float(summary["stationarity_direction_residual"]))
        response_constraint.append(float(summary["constraint_direction_residual"]))

        all_steps = np.asarray(arrays["curvature_steps_bohr"], dtype=np.float64)
        all_step_hvps = np.asarray(arrays["relaxed_hvp_step_scan"], dtype=np.float64)
        selected_indices = [
            index
            for index, step in enumerate(all_steps)
            if any(np.isclose(step, expected, rtol=0.0, atol=1e-15) for expected in strict_hvp_steps)
        ]
        if len(selected_indices) != len(strict_hvp_steps):
            reasons.append("missing_strict_hvp_steps")
        if selected_indices:
            step_hvps = all_step_hvps[selected_indices]
            primary_index = next(
                (
                    index
                    for index, step in enumerate(all_steps[selected_indices])
                    if np.isclose(
                        step,
                        float(summary["hvp_step_bohr"]),
                        rtol=0.0,
                        atol=1e-15,
                    )
                ),
                0,
            )
            primary = step_hvps[primary_index]
            primary_scale = max(
                float(np.sqrt(np.mean(primary**2))),
                gates["hvp_step_stability_component_floor_hartree_per_bohr2"],
            )
            step_instabilities.append(
                max(
                    float(np.sqrt(np.mean((candidate - primary) ** 2))) / primary_scale
                    for index, candidate in enumerate(step_hvps)
                    if index != primary_index
                )
                if len(step_hvps) > 1
                else 0.0
            )
        else:
            step_instabilities.append(np.inf)
        relaxed = np.asarray(arrays["relaxed_hvp"], dtype=np.float64)
        implicit = np.asarray(arrays["implicit_hvp"], dtype=np.float64)
        implicit_relaxed.append(
            float(np.sqrt(np.mean((implicit - relaxed) ** 2)))
            / max(
                float(np.sqrt(np.mean(relaxed**2))),
                gates["hvp_step_stability_component_floor_hartree_per_bohr2"],
            )
        )
        for curvature in summary["curvature_step_scan"]:
            if float(curvature["step_bohr"]) < energy_force_curvature_min_step:
                continue
            energy_curvature = float(curvature["energy_curvature"])
            force_curvature = float(curvature["force_curvature"])
            curvature_mismatches.append(
                abs(energy_curvature - force_curvature)
                / max(
                    abs(energy_curvature),
                    abs(force_curvature),
                    gates["curvature_absolute_floor_hartree_per_bohr2"],
                )
            )

    if not curvature_mismatches:
        reasons.append("missing_energy_force_curvature_step")
    energy_spread = max(
        (max(values) - min(values) for values in energies_by_point.values() if values),
        default=np.inf,
    )
    density_cosines = []
    density_distances = []
    for values in coefficients_by_point.values():
        for left, right in itertools.combinations(values, 2):
            cosine, distance = _coulomb_metrics(left, right, coulomb)
            density_cosines.append(cosine)
            density_distances.append(distance)
    force_branch_rmse = max(
        (
            _max_pairwise(
                values,
                lambda left, right: float(np.sqrt(np.mean((left - right) ** 2))),
            )
            for values in forces_by_point.values()
        ),
        default=np.inf,
    )

    row.update(
        {
            "max_projected_density_gradient": max(gradients, default=np.inf),
            "max_constraint_residual": max(constraints, default=np.inf),
            "branch_total_energy_spread_hartree": energy_spread,
            "branch_density_coulomb_cosine_min": min(density_cosines, default=-np.inf),
            "branch_density_relative_coulomb_distance_max": max(
                density_distances, default=np.inf
            ),
            "branch_force_rmse_max_hartree_per_bohr": force_branch_rmse,
            "tangent_min_eigenvalue_min": min(kkt_minima, default=-np.inf),
            "tangent_condition_number_max": max(kkt_conditions, default=np.inf),
            "response_relative_residual_max": max(response_relative, default=np.inf),
            "response_stationarity_residual_max": max(
                response_stationarity, default=np.inf
            ),
            "response_constraint_residual_max": max(response_constraint, default=np.inf),
            "hvp_step_stability_relative_max": max(step_instabilities, default=np.inf),
            "implicit_vs_relaxed_relative_max": max(implicit_relaxed, default=np.inf),
            "energy_force_curvature_relative_max": max(
                curvature_mismatches, default=np.inf
            ),
            "wall_time_s_sum": sum(wall_times),
        }
    )

    checks = {
        "density_gradient": row["max_projected_density_gradient"]
        <= gates["projected_density_gradient_max"],
        "constraint": row["max_constraint_residual"] <= gates["constraint_residual_max"],
        "branch_energy": row["branch_total_energy_spread_hartree"]
        <= gates["branch_total_energy_spread_max_hartree"],
        "branch_density_cosine": row["branch_density_coulomb_cosine_min"]
        >= gates["branch_density_coulomb_cosine_min"],
        "branch_density_distance": row["branch_density_relative_coulomb_distance_max"]
        <= gates["branch_density_relative_coulomb_distance_max"],
        "branch_force": row["branch_force_rmse_max_hartree_per_bohr"]
        <= gates["branch_force_rmse_max_hartree_per_bohr"],
        "kkt_positive": row["tangent_min_eigenvalue_min"]
        >= gates["tangent_min_eigenvalue_min"],
        "kkt_condition": row["tangent_condition_number_max"]
        <= gates["tangent_condition_number_max"],
        "response_relative": row["response_relative_residual_max"]
        <= gates["response_relative_residual_max"],
        "response_stationarity": row["response_stationarity_residual_max"]
        <= gates["response_stationarity_residual_max"],
        "response_constraint": row["response_constraint_residual_max"]
        <= gates["response_constraint_residual_max"],
        "step_stability": row["hvp_step_stability_relative_max"]
        <= gates["hvp_step_stability_relative_max"],
        "implicit_agreement": row["implicit_vs_relaxed_relative_max"]
        <= gates["implicit_vs_relaxed_relative_max"],
        "energy_force_curvature": row["energy_force_curvature_relative_max"]
        <= gates["energy_force_curvature_relative_max"],
    }
    reasons.extend(key for key, passed in checks.items() if not passed)
    row["stable"] = not reasons
    row["failure_reasons"] = ";".join(sorted(set(reasons)))
    return row


def _write_filtered_sidecars(
    input_dir: Path,
    output_dir: Path,
    direction_rows: list[dict[str, Any]],
    parent_rows: list[dict[str, Any]],
    expected_directions: int,
) -> Path:
    masks: dict[str, np.ndarray] = {}
    for row in direction_rows:
        masks.setdefault(
            row["molecule_id"], np.zeros(expected_directions, dtype=np.bool_)
        )[int(row["direction_index"])] = bool(row["stable"])
    eligible = {
        row["molecule_id"] for row in parent_rows if bool(row["parent_stable"])
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for molecule_id, mask in masks.items():
        if molecule_id not in eligible:
            continue
        source = input_dir / f"{molecule_id}.0000000.npz"
        if not source.exists():
            raise FileNotFoundError(source)
        with np.load(source) as source_payload:
            payload = {key: np.asarray(source_payload[key]) for key in source_payload.files}
        payload["stability_mask"] = mask
        payload["parent_stability_eligible"] = np.asarray(True, dtype=np.bool_)
        destination = output_dir / source.name
        np.savez_compressed(destination, **payload)
        entries.append(
            {
                "molecule_id": molecule_id,
                "filename": destination.name,
                "sha256": _sha256(destination),
                "stability_mask": mask.astype(int).tolist(),
                "stable_direction_count": int(mask.sum()),
            }
        )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "definition": "Frozen stable-only HVP sidecars; manifest entries are authoritative.",
                "source_dir": input_dir.resolve().as_posix(),
                "sidecar_count": len(entries),
                "entries": sorted(entries, key=lambda row: row["molecule_id"]),
                "test100_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return manifest_path


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol.get("test100_access_allowed") is not False:
        raise ValueError("Protocol must explicitly forbid Test100 access")
    gates = {key: float(value) for key, value in protocol["gates"].items()}
    strict_hvp_steps = [
        float(value) for value in protocol["finite_difference"]["strict_hvp_steps_bohr"]
    ]
    energy_force_curvature_min_step = float(
        protocol["finite_difference"]["energy_force_curvature_min_step_bohr"]
    )
    grouped, load_errors = _load_tasks(args.task_root)
    expected_molecules = (
        [line.strip().zfill(7) for line in args.expected_molecules.read_text().splitlines() if line.strip()]
        if args.expected_molecules is not None
        else sorted({key[0] for key in grouped})
    )
    rows = []
    for molecule_id in expected_molecules:
        for direction_index in range(args.expected_directions):
            rows.append(
                _gate_direction(
                    molecule_id,
                    direction_index,
                    grouped.get((molecule_id, direction_index), {}),
                    gates,
                    strict_hvp_steps,
                    energy_force_curvature_min_step,
                )
            )
    parent_rows = []
    aggregation = protocol.get("aggregation", {})
    parent_exclusion_gates = set(aggregation.get("parent_exclusion_gates", []))
    minimum_stable_directions = int(
        aggregation.get("minimum_stable_directions", args.expected_directions)
    )
    if not 1 <= minimum_stable_directions <= args.expected_directions:
        raise ValueError("minimum_stable_directions must be within expected directions")
    for molecule_id in expected_molecules:
        molecule_rows = [row for row in rows if row["molecule_id"] == molecule_id]
        stable_count = sum(bool(row["stable"]) for row in molecule_rows)
        all_reasons = {
            reason
            for row in molecule_rows
            for reason in str(row["failure_reasons"]).split(";")
            if reason
        }
        parent_gate_failures = sorted(all_reasons & parent_exclusion_gates)
        parent_stable = (
            stable_count >= minimum_stable_directions and not parent_gate_failures
        )
        parent_rows.append(
            {
                "molecule_id": molecule_id,
                "direction_count": len(molecule_rows),
                "stable_direction_count": stable_count,
                "minimum_stable_directions": minimum_stable_directions,
                "parent_stable": parent_stable,
                "parent_gate_failures": ";".join(parent_gate_failures),
                "failure_reasons": ";".join(sorted(all_reasons)),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "direction_stability.csv", rows)
    _write_csv(args.output_dir / "parent_stability.csv", parent_rows)
    _plot_diagnostics(args.output_dir, rows, parent_rows)
    stable_ids = [row["molecule_id"] for row in parent_rows if row["parent_stable"]]
    excluded_ids = [row["molecule_id"] for row in parent_rows if not row["parent_stable"]]
    (args.output_dir / "stable_parent_ids.txt").write_text("\n".join(stable_ids) + ("\n" if stable_ids else ""))
    (args.output_dir / "excluded_parent_ids.txt").write_text(
        "\n".join(excluded_ids) + ("\n" if excluded_ids else "")
    )
    stable_sidecar_manifest = None
    if args.input_sidecar_dir is not None:
        if args.output_sidecar_dir is None:
            raise ValueError("--output-sidecar-dir is required with --input-sidecar-dir")
        stable_sidecar_manifest = _write_filtered_sidecars(
            args.input_sidecar_dir,
            args.output_sidecar_dir,
            rows,
            parent_rows,
            args.expected_directions,
        )
    result = {
        "definition": __doc__,
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": _sha256(args.protocol),
        "task_roots": [str(path.resolve()) for path in args.task_root],
        "molecule_count": len(expected_molecules),
        "direction_count": len(rows),
        "stable_direction_count": sum(bool(row["stable"]) for row in rows),
        "stable_parent_count": len(stable_ids),
        "excluded_parent_count": len(excluded_ids),
        "stable_parent_ids": stable_ids,
        "excluded_parent_ids": excluded_ids,
        "minimum_stable_directions": minimum_stable_directions,
        "parent_exclusion_gates": sorted(parent_exclusion_gates),
        "load_errors": load_errors,
        "stable_sidecar_manifest": (
            stable_sidecar_manifest.resolve().as_posix()
            if stable_sidecar_manifest is not None
            else None
        ),
        "test100_accessed": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-molecules", type=Path)
    parser.add_argument("--expected-directions", type=int, default=4)
    parser.add_argument("--input-sidecar-dir", type=Path)
    parser.add_argument("--output-sidecar-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
