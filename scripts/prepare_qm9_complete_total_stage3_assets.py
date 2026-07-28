#!/usr/bin/env python3
"""Freeze train-only replay/HVP and independent validation assets for Stage 3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

try:
    from scripts.qm9_complete_total_stage2_select_baselines import (
        _load_provenance_audit,
        _run_parent,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from qm9_complete_total_stage2_select_baselines import (
        _load_provenance_audit,
        _run_parent,
    )


LABEL_PATTERN = re.compile(r"^(?P<parent>\d{7})\.(?P<sample>\d{7})\.zarr\.zip$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _validation_capacity_manifest(
    validation_rows: list[dict[str, Any]],
    *,
    protocol: Path | None,
    validation_stability_summary: Path,
) -> dict[str, Any]:
    """Build the frozen manifest consumed by the strict capacity evaluator."""
    return {
        "definition": (
            "Independent validation-parent inputs for original-A strict "
            "complete-total density-relaxed full-Hessian baselines."
        ),
        "protocol": None if protocol is None else protocol.resolve().as_posix(),
        "protocol_sha256": None if protocol is None else _sha256(protocol),
        "validation_stability_summary": (
            validation_stability_summary.resolve().as_posix()
        ),
        "validation_stability_summary_sha256": _sha256(
            validation_stability_summary
        ),
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "parent_count": len(validation_rows),
        "parents": validation_rows,
    }


def _parent_ids_from_difficulty(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    parent_ids = [str(row["parent_id"]) for row in rows]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("train800 difficulty CSV contains duplicate parent IDs")
    return parent_ids


def _replay_rows(
    dataset_dir: Path, parent_ids: list[str], expected_sample_ids: set[int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for parent_id in sorted(parent_ids):
        local = []
        for path in sorted((dataset_dir / "labels").glob(f"{parent_id}.*.zarr.zip")):
            match = LABEL_PATTERN.match(path.name)
            if match is None or match.group("parent") != parent_id:
                continue
            sample_id = int(match.group("sample"))
            if sample_id in expected_sample_ids:
                local.append((sample_id, path.resolve()))
        actual = {sample_id for sample_id, _ in local}
        if actual != expected_sample_ids or len(local) != len(expected_sample_ids):
            raise ValueError(
                f"{parent_id} replay samples {sorted(actual)} != "
                f"{sorted(expected_sample_ids)}"
            )
        for sample_id, path in local:
            rows.append(
                {
                    "molecule_id": parent_id,
                    "sample_id": sample_id,
                    "label_path": path.as_posix(),
                    "label_sha256": _sha256(path),
                }
            )
    return rows


def _latest_baseline_rows(
    candidate_manifest: dict[str, Any], baseline_run_root: Path
) -> dict[str, dict[str, Any]]:
    candidates = {
        str(row["molecule_id"]): row for row in candidate_manifest["candidates"]
    }
    available: dict[str, dict[str, Any]] = {}
    for run_dir in sorted(baseline_run_root.glob("*_job*")):
        metric_path = run_dir / "full_hessian_metrics.csv"
        summary_path = run_dir / "summary.json"
        if not metric_path.is_file() or not summary_path.is_file():
            continue
        with metric_path.open(newline="") as handle:
            metric_rows = list(csv.DictReader(handle))
        if len(metric_rows) != 1:
            continue
        molecule_id = str(metric_rows[0]["molecule_id"])
        if molecule_id not in candidates:
            raise ValueError(f"non-frozen baseline parent in {run_dir}: {molecule_id}")
        available[molecule_id] = _run_parent(run_dir, candidates[molecule_id])
    return available


def _pbe_hessian_lookup(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_json(path)
    if not isinstance(rows, list):
        raise ValueError("PBE Hessian manifest must be a list")
    lookup = {str(row["molecule_id"]): row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("duplicate molecule IDs in PBE Hessian manifest")
    return lookup


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    selection = _read_json(args.train100_selection_manifest)
    if selection.get("test_accessed_for_selection") is not False:
        raise ValueError("train100 selection does not certify frozen Test100")
    train100_rows = _read_json(args.train100_selection)
    train100_ids = {str(row["parent_id"]) for row in train100_rows}
    train800_ids = set(_parent_ids_from_difficulty(args.train800_difficulty_csv))
    if len(train800_ids) != args.expected_train_parent_count:
        raise ValueError(
            f"expected {args.expected_train_parent_count} train parents, "
            f"found {len(train800_ids)}"
        )
    if len(train100_ids) != args.expected_hvp_inventory_parent_count:
        raise ValueError(
            f"expected {args.expected_hvp_inventory_parent_count} selected HVP parents, "
            f"found {len(train100_ids)}"
        )
    if not train100_ids <= train800_ids:
        raise ValueError("train100 is not a subset of train800")

    train_stability = _read_json(args.train_stability_summary)
    if train_stability.get("test100_accessed") is not False:
        raise ValueError("train stability summary does not certify frozen Test100")
    train_stable_ids = {str(value) for value in train_stability["stable_parent_ids"]}
    stable_sidecars = _read_json(args.train_stable_sidecar_manifest)
    if stable_sidecars.get("test100_accessed") is not False:
        raise ValueError("stable sidecars do not certify frozen Test100")
    sidecar_lookup = {
        str(row["molecule_id"]): row for row in stable_sidecars["entries"]
    }
    if train_stable_ids != set(sidecar_lookup):
        raise ValueError("stable parent IDs and sidecar manifest differ")

    candidate_manifest = _read_json(args.stage2_candidate_manifest)
    if candidate_manifest.get("test100_accessed") is not False:
        raise ValueError("Stage-2 candidate manifest does not certify frozen Test100")
    baseline_rows: dict[str, dict[str, Any]] = {}
    for path in args.known_baseline_manifest:
        known = _read_json(path)
        if known.get("test100_accessed") is not False:
            raise ValueError(f"{path} does not certify frozen Test100")
        for row in known["parents"]:
            baseline_rows[str(row["molecule_id"])] = row
    baseline_rows.update(
        _latest_baseline_rows(candidate_manifest, args.stage2_baseline_run_root)
    )
    density_gate = float(
        candidate_manifest["numerical_gate"]["strict_density_gradient_max"]
    )
    curl_gate = float(
        candidate_manifest["numerical_gate"][
            "baseline_asym_over_pbe_frobenius_max"
        ]
    )
    provenance_path = getattr(args, "baseline_provenance_audit", None)
    provenance = (
        _load_provenance_audit(provenance_path)
        if provenance_path is not None
        else None
    )
    baseline_audit: list[dict[str, Any]] = []
    eligible_baselines: dict[str, dict[str, Any]] = {}
    for molecule_id in sorted(train_stable_ids):
        row = baseline_rows.get(molecule_id)
        reasons = []
        if row is None:
            reasons.append("missing_complete_total_baseline")
        else:
            if not row["full_hessian_complete"]:
                reasons.append("incomplete_hessian")
            if row["baseline_max_density_gradient_norm"] > density_gate:
                reasons.append("density_gradient")
            if row["baseline_asym_over_pbe_frobenius"] > curl_gate:
                reasons.append("pbe_normalized_curl")
            if provenance is not None:
                run_dir = str(Path(row.get("baseline_run_dir", "")).resolve())
                provenance_row = provenance.get(run_dir)
                if provenance_row is None:
                    reasons.append("missing_checkpoint_provenance")
                elif not (
                    provenance_row.get("status") == "pass"
                    and provenance_row.get("base_state_exact_match") is True
                ):
                    reasons.append("checkpoint_provenance")
        baseline_audit.append(
            {
                "molecule_id": molecule_id,
                "eligible": not reasons,
                "failure_reasons": ";".join(reasons),
                "baseline_max_density_gradient_norm": (
                    None if row is None else row["baseline_max_density_gradient_norm"]
                ),
                "baseline_asym_over_pbe_frobenius": (
                    None if row is None else row["baseline_asym_over_pbe_frobenius"]
                ),
            }
        )
        if not reasons:
            eligible_baselines[molecule_id] = row

    pbe_lookup = _pbe_hessian_lookup(args.pbe_hessian_manifest)
    hvp_rows = []
    for molecule_id, baseline in sorted(eligible_baselines.items()):
        sidecar = sidecar_lookup[molecule_id]
        pbe = pbe_lookup.get(molecule_id)
        if pbe is None or not pbe.get("success") or not pbe.get("finite"):
            raise ValueError(f"missing finite PBE Hessian for {molecule_id}")
        sidecar_path = (
            args.train_stable_sidecar_manifest.parent / sidecar["filename"]
        ).resolve()
        if _sha256(sidecar_path) != sidecar["sha256"]:
            raise ValueError(f"stable sidecar hash mismatch for {molecule_id}")
        hvp_rows.append(
            {
                "molecule_id": molecule_id,
                "natoms": baseline["natoms"],
                "stable_direction_count": int(sum(sidecar["stability_mask"])),
                "stability_mask": "".join(str(int(x)) for x in sidecar["stability_mask"]),
                "stable_sidecar_path": sidecar_path.as_posix(),
                "stable_sidecar_sha256": sidecar["sha256"],
                "pbe_hessian_path": pbe["cache_path"],
                "pbe_hessian_sha256": _sha256(Path(pbe["cache_path"])),
                "baseline_capacity_array": baseline["capacity_array"],
                "baseline_capacity_array_sha256": baseline["capacity_array_sha256"],
                "baseline_asym_over_pbe_frobenius": baseline[
                    "baseline_asym_over_pbe_frobenius"
                ],
            }
        )

    validation_stability = _read_json(args.validation_stability_summary)
    if validation_stability.get("test100_accessed") is not False:
        raise ValueError("validation stability summary does not certify frozen Test100")
    validation_ids = {
        str(value) for value in validation_stability["stable_parent_ids"]
    }
    if train800_ids & validation_ids:
        raise ValueError("train800/validation parent leakage")
    validation_rows = []
    for molecule_id in sorted(validation_ids):
        pbe = pbe_lookup.get(molecule_id)
        if pbe is None or not pbe.get("success") or not pbe.get("finite"):
            raise ValueError(f"missing finite validation PBE Hessian for {molecule_id}")
        label_path = (
            args.dataset_dir / "labels" / f"{molecule_id}.0000000.zarr.zip"
        ).resolve()
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        validation_rows.append(
            {
                "molecule_id": molecule_id,
                "natoms": pbe["natoms"],
                "label_path": label_path.as_posix(),
                "label_sha256": _sha256(label_path),
                "pbe_hessian_path": pbe["cache_path"],
                "pbe_hessian_sha256": _sha256(Path(pbe["cache_path"])),
                "complete_total_baseline_status": "missing",
            }
        )

    replay_rows = _replay_rows(
        args.dataset_dir, sorted(train800_ids), set(args.replay_sample_ids)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    replay_csv = args.output_dir / "train800_replay_labels.csv"
    hvp_csv = args.output_dir / "train100_stable_hvp_parents.csv"
    validation_csv = args.output_dir / "validation_stable_candidates.csv"
    validation_capacity_manifest_path = (
        args.output_dir / "validation_capacity_manifest.json"
    )
    baseline_csv = args.output_dir / "train100_complete_total_baseline_audit.csv"
    _write_csv(replay_csv, replay_rows)
    _write_csv(hvp_csv, hvp_rows)
    _write_csv(validation_csv, validation_rows)
    _write_csv(baseline_csv, baseline_audit)
    validation_capacity_manifest = _validation_capacity_manifest(
        validation_rows,
        protocol=args.protocol,
        validation_stability_summary=args.validation_stability_summary,
    )
    validation_capacity_manifest_path.write_text(
        json.dumps(validation_capacity_manifest, indent=2, sort_keys=True) + "\n"
    )

    input_paths = {
        "train800_difficulty_csv": args.train800_difficulty_csv,
        "train100_selection_manifest": args.train100_selection_manifest,
        "train100_selection": args.train100_selection,
        "train_stability_summary": args.train_stability_summary,
        "train_stable_sidecar_manifest": args.train_stable_sidecar_manifest,
        "stage2_candidate_manifest": args.stage2_candidate_manifest,
        "validation_stability_summary": args.validation_stability_summary,
        "pbe_hessian_manifest": args.pbe_hessian_manifest,
    }
    if provenance_path is not None:
        input_paths["baseline_provenance_audit"] = provenance_path
    input_paths.update(
        {
            f"known_baseline_manifest_{index}": path
            for index, path in enumerate(args.known_baseline_manifest)
        }
    )
    result = {
        "definition": (
            "Frozen Stage-3 train800 E/F replay, stability-masked train100 HVP, "
            "and independent validation assets."
        ),
        "protocol": None if args.protocol is None else args.protocol.resolve().as_posix(),
        "protocol_sha256": None if args.protocol is None else _sha256(args.protocol),
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "input_artifacts": {
            name: {"path": path.resolve().as_posix(), "sha256": _sha256(path)}
            for name, path in input_paths.items()
        },
        "counts": {
            "train800_parents": len(train800_ids),
            "train800_replay_labels": len(replay_rows),
            "train100_inventory_parents": len(train100_ids),
            "train100_v3_stable_parents": len(train_stable_ids),
            "train100_complete_total_eligible_parents": len(hvp_rows),
            "train100_complete_total_excluded_parents": len(train_stable_ids)
            - len(hvp_rows),
            "validation_v3_stable_candidates": len(validation_rows),
            "validation_complete_total_baselines_ready": 0,
        },
        "gates": {
            "strict_density_gradient_max": density_gate,
            "baseline_asym_over_pbe_frobenius_max": curl_gate,
        },
        "output_artifacts": {
            "train800_replay_labels": {
                "path": replay_csv.resolve().as_posix(),
                "sha256": _sha256(replay_csv),
            },
            "train100_stable_hvp_parents": {
                "path": hvp_csv.resolve().as_posix(),
                "sha256": _sha256(hvp_csv),
            },
            "validation_stable_candidates": {
                "path": validation_csv.resolve().as_posix(),
                "sha256": _sha256(validation_csv),
            },
            "validation_capacity_manifest": {
                "path": validation_capacity_manifest_path.resolve().as_posix(),
                "sha256": _sha256(validation_capacity_manifest_path),
            },
            "train100_complete_total_baseline_audit": {
                "path": baseline_csv.resolve().as_posix(),
                "sha256": _sha256(baseline_csv),
            },
        },
        "eligible_hvp_parent_ids": [row["molecule_id"] for row in hvp_rows],
        "validation_candidate_ids": [row["molecule_id"] for row in validation_rows],
        "next_required_asset": (
            "strict complete-total density-relaxed E/F baselines for 3200 train replay "
            "geometries and full-Hessian baselines for seven validation candidates"
        ),
    }
    manifest_path = args.output_dir / "stage3_asset_manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--train800-difficulty-csv", type=Path, required=True)
    parser.add_argument("--train100-selection-manifest", type=Path, required=True)
    parser.add_argument("--train100-selection", type=Path, required=True)
    parser.add_argument("--train-stability-summary", type=Path, required=True)
    parser.add_argument("--train-stable-sidecar-manifest", type=Path, required=True)
    parser.add_argument("--stage2-candidate-manifest", type=Path, required=True)
    parser.add_argument("--stage2-baseline-run-root", type=Path, required=True)
    parser.add_argument("--baseline-provenance-audit", type=Path)
    parser.add_argument(
        "--known-baseline-manifest", type=Path, action="append", default=[]
    )
    parser.add_argument("--validation-stability-summary", type=Path, required=True)
    parser.add_argument("--pbe-hessian-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-train-parent-count", type=int, default=800)
    parser.add_argument("--expected-hvp-inventory-parent-count", type=int, default=100)
    parser.add_argument("--replay-sample-ids", type=int, nargs="+", default=[0, 1, 2, 3])
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
