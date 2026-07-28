#!/usr/bin/env python3
"""Freeze and evaluate post-fit directions that can never return to training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import zarr

try:
    from scripts.prepare_qm9_complete_total_direction_generalization import (
        build_direction_bank,
    )
except ModuleNotFoundError:
    from prepare_qm9_complete_total_direction_generalization import build_direction_bank


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _jitter_structured_directions(
    directions: np.ndarray,
    kinds: np.ndarray,
    external_basis: np.ndarray,
    *,
    fraction: float,
    seed: int,
) -> np.ndarray:
    if not 0.0 < fraction < 1.0:
        raise ValueError("structured_jitter_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    output = np.asarray(directions, dtype=np.float64).copy()
    for index, kind in enumerate(np.asarray(kinds).astype(str)):
        if kind == "random_internal":
            continue
        base = output[index]
        for _ in range(100):
            noise = rng.normal(size=base.size)
            noise -= external_basis @ (external_basis.T @ noise)
            noise -= np.dot(base, noise) * base
            norm = np.linalg.norm(noise)
            if norm > 1.0e-10:
                noise /= norm
                output[index] = (base + fraction * noise) / np.sqrt(
                    1.0 + fraction**2
                )
                break
        else:
            raise RuntimeError("could not jitter structured confirmation direction")
    return output


def prepare_confirmation(args: argparse.Namespace) -> dict[str, Any]:
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol.get("test100_access_allowed") is not False:
        raise ValueError("confirmation protocol must freeze Test100")
    expected_parent_hash = protocol["source"]["selected_parent_manifest_sha256"]
    expected_direction_hash = protocol["source"]["trained_direction_manifest_sha256"]
    if _sha256(args.parent_manifest) != expected_parent_hash:
        raise ValueError("selected parent manifest hash mismatch")
    if _sha256(args.trained_direction_manifest) != expected_direction_hash:
        raise ValueError("trained direction manifest hash mismatch")
    parents = json.loads(args.parent_manifest.read_text())["parents"]
    trained_manifest = json.loads(args.trained_direction_manifest.read_text())
    trained_lookup = {
        row["molecule_id"]: row for row in trained_manifest["directions"]
    }
    config = protocol["directions"]
    direction_dir = args.output_dir / "directions"
    direction_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for order, parent in enumerate(parents):
        molecule_id = str(parent["molecule_id"])
        root = zarr.open(parent["label_path"], mode="r")
        atomic_numbers = np.asarray(root["geometry/atomic_numbers"], dtype=np.int64)
        positions = np.asarray(root["geometry/atom_pos"], dtype=np.float64)
        with np.load(parent["pbe_hessian_path"]) as payload:
            pbe_hessian = np.asarray(payload["pbe_hessian"], dtype=np.float64)
        bank = build_direction_bank(
            atomic_numbers,
            positions,
            pbe_hessian,
            seed=int(config["seed"]) + order,
            maximum=int(config["maximum_per_parent"]),
            structured_per_kind=int(config["structured_per_kind"]),
            heldout_fraction=0.2,
            minimum_random_fraction=float(config["minimum_random_fraction"]),
        )
        confirmation_directions = _jitter_structured_directions(
            bank["directions"],
            bank["kinds"],
            bank["external_basis"],
            fraction=float(config["structured_jitter_fraction"]),
            seed=int(config["seed"]) + 100000 + order,
        )
        trained_path = Path(trained_lookup[molecule_id]["direction_path"])
        with np.load(trained_path) as trained_payload:
            trained_directions = np.asarray(trained_payload["directions"])
        maximum_overlap = float(
            np.max(np.abs(confirmation_directions @ trained_directions.T))
        )
        if maximum_overlap >= 1.0 - 1.0e-10:
            raise ValueError(f"confirmation direction duplicates training bank: {molecule_id}")
        output_path = direction_dir / f"{molecule_id}.npz"
        np.savez_compressed(
            output_path,
            directions=confirmation_directions,
            kinds=bank["kinds"],
            roles=np.full(len(confirmation_directions), "confirm", dtype="U7"),
            external_basis=bank["external_basis"],
            pbe_hvp=np.einsum(
                "ij,dj->di", pbe_hessian, confirmation_directions
            ),
        )
        rows.append(
            {
                "molecule_id": molecule_id,
                "natoms": int(len(atomic_numbers)),
                "direction_path": output_path.resolve().as_posix(),
                "direction_sha256": _sha256(output_path),
                "direction_count": int(len(confirmation_directions)),
                "random_direction_count": int(
                    np.sum(bank["kinds"] == "random_internal")
                ),
                "maximum_abs_overlap_with_trained_bank": maximum_overlap,
                "direction_pairwise_max_abs_cosine": float(
                    np.max(
                        np.abs(
                            confirmation_directions @ confirmation_directions.T
                            - np.eye(len(confirmation_directions))
                        )
                    )
                ),
                "external_overlap_max_abs": float(
                    np.max(np.abs(confirmation_directions @ bank["external_basis"]))
                ),
            }
        )
    result = {
        "protocol_id": protocol["protocol_id"],
        "protocol_path": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "source_parent_manifest": args.parent_manifest.resolve().as_posix(),
        "source_parent_manifest_sha256": _sha256(args.parent_manifest),
        "source_trained_direction_manifest": args.trained_direction_manifest.resolve().as_posix(),
        "source_trained_direction_manifest_sha256": _sha256(
            args.trained_direction_manifest
        ),
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "confirmation_results_may_not_return_to_training": True,
        "parents": rows,
    }
    output = args.output_dir / "confirmation_manifest.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".json.sha256").write_text(
        f"{_sha256(output)}  {output.name}\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def evaluate_confirmation(args: argparse.Namespace) -> dict[str, Any]:
    protocol = yaml.safe_load(args.protocol.read_text())
    manifest = json.loads(args.confirmation_manifest.read_text())
    if manifest.get("test100_accessed") is not False:
        raise ValueError("confirmation manifest does not freeze Test100")
    if manifest.get("confirmation_results_may_not_return_to_training") is not True:
        raise ValueError("confirmation manifest lacks no-training certificate")
    floor = float(protocol["metrics"]["hvp_reference_floor"])
    per_direction = []
    per_parent = []
    summaries = []
    for item in args.run:
        name, run_text = item.split("=", maxsplit=1)
        run_dir = Path(run_text)
        run_summary = json.loads((run_dir / "summary.json").read_text())
        if run_summary.get("test100_accessed") is not False:
            raise ValueError(f"{name} does not freeze Test100")
        local_parents = []
        for parent in manifest["parents"]:
            molecule_id = parent["molecule_id"]
            with np.load(run_dir / f"{molecule_id}_result.npz") as payload:
                predicted = np.asarray(payload["predicted_hessian"])
                reference_hessian = np.asarray(payload["pbe_hessian"])
            with np.load(parent["direction_path"]) as payload:
                directions = np.asarray(payload["directions"])
                kinds = np.asarray(payload["kinds"]).astype(str)
            difference = predicted - reference_hessian
            error_hvp = np.einsum("ij,dj->di", difference, directions)
            reference_hvp = np.einsum(
                "ij,dj->di", reference_hessian, directions
            )
            error_norm = np.linalg.norm(error_hvp, axis=1)
            reference_norm = np.linalg.norm(reference_hvp, axis=1)
            floored_relative = error_norm / np.maximum(reference_norm, floor)
            for index, kind in enumerate(kinds):
                per_direction.append(
                    {
                        "run": name,
                        "molecule_id": molecule_id,
                        "direction_index": index,
                        "kind": kind,
                        "hvp_mae": float(np.mean(np.abs(error_hvp[index]))),
                        "hvp_rmse": float(np.sqrt(np.mean(error_hvp[index] ** 2))),
                        "hvp_relative_l2": float(
                            error_norm[index]
                            / max(reference_norm[index], np.finfo(float).tiny)
                        ),
                        "hvp_floored_relative_l2": float(floored_relative[index]),
                        "reference_l2": float(reference_norm[index]),
                    }
                )
            row = {
                "run": name,
                "molecule_id": molecule_id,
                "natoms": parent["natoms"],
                "hvp_mae": float(np.mean(np.abs(error_hvp))),
                "hvp_rmse": float(np.sqrt(np.mean(error_hvp**2))),
                "hvp_relative_l2": float(
                    np.linalg.norm(error_hvp)
                    / max(np.linalg.norm(reference_hvp), np.finfo(float).tiny)
                ),
                "hvp_floored_relative_rms": float(
                    np.sqrt(np.mean(floored_relative**2))
                ),
            }
            per_parent.append(row)
            local_parents.append(row)
        relative = np.asarray(
            [row["hvp_relative_l2"] for row in local_parents], dtype=float
        )
        gates = protocol["gates"]
        summary = {
            "run": name,
            "parent_count": len(local_parents),
            "median_parent_hvp_relative_l2": float(np.median(relative)),
            "p80_parent_hvp_relative_l2": float(np.quantile(relative, 0.8)),
            "p90_parent_hvp_relative_l2": float(np.quantile(relative, 0.9)),
            "max_parent_hvp_relative_l2": float(np.max(relative)),
            "fraction_parent_hvp_relative_l2_below_0p10": float(
                np.mean(relative <= 0.10)
            ),
            "fraction_parent_hvp_relative_l2_below_0p15": float(
                np.mean(relative <= 0.15)
            ),
            "fraction_parent_hvp_relative_l2_below_0p20": float(
                np.mean(relative <= 0.20)
            ),
            "mean_hvp_mae": float(
                np.mean([row["hvp_mae"] for row in local_parents])
            ),
            "mean_hvp_rmse": float(
                np.mean([row["hvp_rmse"] for row in local_parents])
            ),
            "median_hvp_floored_relative_rms": float(
                np.median(
                    [row["hvp_floored_relative_rms"] for row in local_parents]
                )
            ),
        }
        summary["confirmation_gate_passed"] = bool(
            summary["median_parent_hvp_relative_l2"]
            <= float(gates["median_parent_hvp_relative_l2_max"])
            and summary["p90_parent_hvp_relative_l2"]
            <= float(gates["p90_parent_hvp_relative_l2_max"])
            and summary["fraction_parent_hvp_relative_l2_below_0p15"]
            >= float(gates["fraction_parent_hvp_relative_l2_below_0p15_min"])
        )
        summaries.append(summary)
    kind_summaries = []
    for run in dict.fromkeys(row["run"] for row in per_direction):
        for kind in sorted(
            {row["kind"] for row in per_direction if row["run"] == run}
        ):
            local = [
                row
                for row in per_direction
                if row["run"] == run and row["kind"] == kind
            ]
            relative = np.asarray([row["hvp_relative_l2"] for row in local])
            kind_summaries.append(
                {
                    "run": run,
                    "kind": kind,
                    "direction_count": len(local),
                    "median_hvp_relative_l2": float(np.median(relative)),
                    "p90_hvp_relative_l2": float(np.quantile(relative, 0.9)),
                    "mean_hvp_mae": float(np.mean([row["hvp_mae"] for row in local])),
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "per_direction.csv", per_direction)
    _write_csv(args.output_dir / "per_parent.csv", per_parent)
    _write_csv(args.output_dir / "run_summary.csv", summaries)
    _write_csv(args.output_dir / "kind_summary.csv", kind_summaries)
    result = {
        "definition": "post-fit frozen confirmation HVPs; results are prohibited from returning to training",
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "confirmation_results_may_not_return_to_training": True,
        "summaries": summaries,
        "kind_summaries": kind_summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--protocol", type=Path, required=True)
    prepare.add_argument("--parent-manifest", type=Path, required=True)
    prepare.add_argument("--trained-direction-manifest", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--protocol", type=Path, required=True)
    evaluate.add_argument("--confirmation-manifest", type=Path, required=True)
    evaluate.add_argument("--run", action="append", required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.command == "prepare":
        prepare_confirmation(parsed)
    else:
        evaluate_confirmation(parsed)
