#!/usr/bin/env python3
"""Plan and verify deterministic 8-GPU Hessian shards for QM9 v11 candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import zarr


EXPECTED_CANDIDATE_MANIFEST_SHA256 = (
    "f1a6a16be453bfbf9898831922b6e482d5581c268ecaaf77d60dff18a97eb0e8"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def label_inventory(dataset_dir: Path) -> dict[str, dict[str, object]]:
    labels = sorted((dataset_dir / "labels").glob("*.zarr.zip"))
    inventory: dict[str, dict[str, object]] = {}
    for label_path in labels:
        molecule_id, sample_text, *_ = label_path.name.removesuffix(".zarr.zip").split(".")
        if int(sample_text) != 0:
            raise RuntimeError(f"unexpected non-reference sample: {label_path}")
        if molecule_id in inventory:
            raise RuntimeError(f"duplicate molecule id in labels: {molecule_id}")
        root = zarr.open(label_path, mode="r")
        natoms = int(root["geometry/atomic_numbers"].shape[0])
        n_scf_steps = int(root["of_labels/n_scf_steps"][()])
        inventory[molecule_id] = {
            "label_path": str(label_path),
            "label_sha256": sha256(label_path),
            "natoms": natoms,
            "n_scf_steps": n_scf_steps,
        }
    return inventory


def checkpoint_inventory(dataset_dir: Path) -> dict[str, dict[str, object]]:
    inventory: dict[str, dict[str, object]] = {}
    for chk_path in sorted((dataset_dir / "kohn_sham").glob("*.chk")):
        stem = chk_path.name.removesuffix(".chk")
        molecule_id, sample_text = stem.rsplit("_", 1)[-1].split(".")
        if int(sample_text) != 0:
            raise RuntimeError(f"unexpected non-reference checkpoint: {chk_path}")
        if molecule_id in inventory:
            raise RuntimeError(f"duplicate molecule id in checkpoints: {molecule_id}")
        inventory[molecule_id] = {
            "chk_path": str(chk_path),
            "chk_sha256": sha256(chk_path),
        }
    return inventory


def load_candidates(path: Path) -> dict[str, dict[str, object]]:
    actual = sha256(path)
    if actual != EXPECTED_CANDIDATE_MANIFEST_SHA256:
        raise RuntimeError(f"candidate manifest SHA256 mismatch: {actual}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    molecules = payload["molecules"]
    result = {row["molecule_id"]: row for row in molecules}
    if len(molecules) != 120 or len(result) != 120:
        raise RuntimeError("candidate manifest must contain 120 unique molecules")
    return result


def plan(args: argparse.Namespace) -> None:
    candidates = load_candidates(args.candidate_manifest)
    labels = label_inventory(args.dataset_dir)
    checkpoints = checkpoint_inventory(args.dataset_dir)
    expected_ids = set(candidates)
    if set(labels) != expected_ids:
        raise RuntimeError(
            f"label IDs mismatch: missing={sorted(expected_ids - set(labels))} "
            f"extra={sorted(set(labels) - expected_ids)}"
        )
    if set(checkpoints) != expected_ids:
        raise RuntimeError(
            f"checkpoint IDs mismatch: missing={sorted(expected_ids - set(checkpoints))} "
            f"extra={sorted(set(checkpoints) - expected_ids)}"
        )

    rows: list[dict[str, object]] = []
    for molecule_id in sorted(expected_ids):
        candidate = candidates[molecule_id]
        label = labels[molecule_id]
        checkpoint = checkpoints[molecule_id]
        if int(candidate["atom_count"]) != int(label["natoms"]):
            raise RuntimeError(f"atom count mismatch for molecule {molecule_id}")
        # Analytic DFT Hessians are dominated by basis/atom count. SCF steps are
        # included as a mild multiplier to avoid putting all difficult cases on
        # one GPU, while preserving the stronger size weighting.
        cost = (int(label["natoms"]) ** 4) * max(int(label["n_scf_steps"]), 1)
        rows.append(
            {
                "molecule_id": molecule_id,
                "natoms": int(label["natoms"]),
                "n_scf_steps": int(label["n_scf_steps"]),
                "estimated_cost": int(cost),
                **checkpoint,
                "label_path": label["label_path"],
                "label_sha256": label["label_sha256"],
            }
        )

    # Longest-processing-time greedy assignment is deterministic and minimizes
    # the largest estimated shard load well for heterogeneous molecule sizes.
    shards = [{"rank": rank, "estimated_cost": 0, "molecules": []} for rank in range(args.shards)]
    for row in sorted(rows, key=lambda item: (-int(item["estimated_cost"]), item["molecule_id"])):
        shard = min(shards, key=lambda item: (int(item["estimated_cost"]), int(item["rank"])))
        shard["molecules"].append(row)
        shard["estimated_cost"] = int(shard["estimated_cost"]) + int(row["estimated_cost"])

    payload = {
        "purpose": "candidate120_full_pbe_hessian_label_generation_not_model_selection",
        "candidate_manifest": str(args.candidate_manifest),
        "candidate_manifest_sha256": EXPECTED_CANDIDATE_MANIFEST_SHA256,
        "dataset_dir": str(args.dataset_dir),
        "molecule_count": len(rows),
        "shard_count": args.shards,
        "cost_model": "natoms**4 * max(n_scf_steps, 1)",
        "shards": shards,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({
        "output": str(args.output),
        "sha256": sha256(args.output),
        "molecule_count": len(rows),
        "shard_sizes": [len(shard["molecules"]) for shard in shards],
        "shard_costs": [shard["estimated_cost"] for shard in shards],
    }, sort_keys=True))


def verify(args: argparse.Namespace) -> None:
    plan_payload = json.loads(args.plan.read_text(encoding="utf-8"))
    planned_rows = [row for shard in plan_payload["shards"] for row in shard["molecules"]]
    planned = {row["molecule_id"]: row for row in planned_rows}
    if len(planned_rows) != 120 or len(planned) != 120:
        raise RuntimeError("plan does not contain exactly 120 unique molecules")

    records: list[dict[str, object]] = []
    for manifest_path in sorted(args.manifest_dir.glob("rank*/manifest.json")):
        rows = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in rows:
            row = dict(row)
            row["rank_manifest"] = str(manifest_path)
            records.append(row)
    by_id = {row["molecule_id"]: row for row in records}
    if len(records) != 120 or len(by_id) != 120 or set(by_id) != set(planned):
        raise RuntimeError(
            f"Hessian record mismatch: records={len(records)} unique={len(by_id)} "
            f"missing={sorted(set(planned) - set(by_id))} extra={sorted(set(by_id) - set(planned))}"
        )

    failures: list[str] = []
    final_rows: list[dict[str, object]] = []
    for molecule_id in sorted(planned):
        expected = planned[molecule_id]
        record = by_id[molecule_id]
        cache_path = Path(str(record["cache_path"]))
        problems: list[str] = []
        if not record.get("success"):
            problems.append("success_false")
        if not record.get("finite"):
            problems.append("nonfinite_manifest")
        if not record.get("chk_unchanged"):
            problems.append("checkpoint_changed_manifest")
        if record.get("chk_sha256_before") != expected["chk_sha256"]:
            problems.append("checkpoint_sha_before_mismatch")
        if record.get("chk_sha256_after") != expected["chk_sha256"]:
            problems.append("checkpoint_sha_after_mismatch")
        if sha256(Path(str(expected["chk_path"]))) != expected["chk_sha256"]:
            problems.append("checkpoint_current_sha_mismatch")
        if not cache_path.is_file():
            problems.append("missing_npz")
            hessian_shape = None
            symmetry_error = None
            finite = False
            output_sha = None
        else:
            hessian = np.asarray(np.load(cache_path)["pbe_hessian"], dtype=np.float64)
            wanted_shape = (3 * int(expected["natoms"]),) * 2
            hessian_shape = list(hessian.shape)
            symmetry_error = float(np.max(np.abs(hessian - hessian.T)))
            finite = bool(np.isfinite(hessian).all())
            output_sha = sha256(cache_path)
            if hessian.shape != wanted_shape:
                problems.append("shape_mismatch")
            if not finite:
                problems.append("nonfinite_npz")
            if symmetry_error > args.symmetry_tolerance:
                problems.append("symmetry_tolerance_exceeded")
        if problems:
            failures.append(f"{molecule_id}:{','.join(problems)}")
        final_rows.append(
            {
                **record,
                "expected_natoms": expected["natoms"],
                "verified_shape": hessian_shape,
                "verified_finite": finite,
                "verified_symmetry_max_abs_error": symmetry_error,
                "output_sha256": output_sha,
                "verification_problems": problems,
            }
        )

    atomic_write_json(args.output, {
        "success": not failures,
        "molecule_count": len(final_rows),
        "symmetry_tolerance": args.symmetry_tolerance,
        "failures": failures,
        "records": final_rows,
    })
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "molecule_id", "natoms", "elapsed_s", "verified_finite",
            "verified_symmetry_max_abs_error", "max_abs", "output_sha256",
        ], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final_rows)
    print(json.dumps({
        "success": not failures,
        "molecule_count": len(final_rows),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "csv": str(csv_path),
        "csv_sha256": sha256(csv_path),
        "failures": failures,
    }, sort_keys=True))
    if failures:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    p_plan = subparsers.add_parser("plan")
    p_plan.add_argument("--dataset-dir", type=Path, required=True)
    p_plan.add_argument("--candidate-manifest", type=Path, required=True)
    p_plan.add_argument("--shards", type=int, default=8)
    p_plan.add_argument("--output", type=Path, required=True)
    p_verify = subparsers.add_parser("verify")
    p_verify.add_argument("--plan", type=Path, required=True)
    p_verify.add_argument("--manifest-dir", type=Path, required=True)
    p_verify.add_argument("--symmetry-tolerance", type=float, default=1e-10)
    p_verify.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "plan":
        plan(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
