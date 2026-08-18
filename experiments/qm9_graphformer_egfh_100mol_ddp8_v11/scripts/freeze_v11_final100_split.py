#!/usr/bin/env python3
"""Freeze a target-scale-stratified, scaffold-aware QM9 v11 80/10/10 split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import zarr
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


ALGORITHM_ID = "qm9_v11_final100_scaffold_target_scale_maxmin_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def robust_standardize(features: np.ndarray) -> np.ndarray:
    median = np.median(features, axis=0)
    q25, q75 = np.percentile(features, [25, 75], axis=0)
    scale = np.maximum(q75 - q25, 1.0e-12)
    return (features - median) / scale


def target_record(row: dict[str, object], dataset_dir: Path, hessian_dir: Path) -> dict[str, object]:
    molecule_id = str(row["molecule_id"])
    label_matches = sorted((dataset_dir / "labels").glob(f"{molecule_id}.0000000.zarr.zip"))
    if len(label_matches) != 1:
        raise RuntimeError(f"expected one label for {molecule_id}, found {len(label_matches)}")
    label_path = label_matches[0]
    hessian_path = hessian_dir / f"pbe_hessian_{molecule_id}_0000000.npz"
    if not hessian_path.is_file():
        raise RuntimeError(f"missing Hessian for {molecule_id}")
    root = zarr.open(label_path, mode="r")
    force = np.asarray(root["metadata/pbe_derivatives/forces"], dtype=np.float64)
    hessian = np.asarray(np.load(hessian_path)["pbe_hessian"], dtype=np.float64)
    natoms = int(row["atom_count"])
    if force.shape != (natoms, 3) or hessian.shape != (3 * natoms, 3 * natoms):
        raise RuntimeError(f"target shape mismatch for {molecule_id}")
    if not np.isfinite(force).all() or not np.isfinite(hessian).all():
        raise RuntimeError(f"nonfinite target for {molecule_id}")
    symmetry = float(np.max(np.abs(hessian - hessian.T)))
    if symmetry > 1.0e-10:
        raise RuntimeError(f"Hessian symmetry failure for {molecule_id}: {symmetry}")
    mol = Chem.MolFromSmiles(str(row["canonical_smiles"]))
    if mol is None:
        raise RuntimeError(f"cannot parse canonical SMILES for {molecule_id}")
    fingerprint = AllChem.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    ).GetFingerprint(mol)
    return {
        **row,
        "label_path": str(label_path),
        "label_sha256": sha256(label_path),
        "pbe_hessian_path": str(hessian_path),
        "pbe_hessian_sha256": sha256(hessian_path),
        "force_mae_target_scale": float(np.mean(np.abs(force))),
        "force_rms_target_scale": float(np.sqrt(np.mean(force * force))),
        "hessian_mae_target_scale": float(np.mean(np.abs(hessian))),
        "hessian_rms_target_scale": float(np.sqrt(np.mean(hessian * hessian))),
        "hessian_frobenius_target_scale": float(np.linalg.norm(hessian)),
        "hessian_symmetry_max_abs": symmetry,
        "_fingerprint": fingerprint,
    }


def numerical_features(rows: list[dict[str, object]]) -> np.ndarray:
    values = []
    for row in rows:
        values.append([
            float(row["atom_count"]),
            float(row["heavy_atom_count"]),
            float(row["radius_of_gyration_angstrom"]),
            float(row["max_pair_distance_angstrom"]),
            math.log10(max(float(row["force_rms_target_scale"]), 1.0e-16)),
            math.log10(max(float(row["hessian_rms_target_scale"]), 1.0e-16)),
            math.log10(max(float(row["hessian_frobenius_target_scale"]), 1.0e-16)),
        ])
    return robust_standardize(np.asarray(values, dtype=np.float64))


def choose_final100(rows: list[dict[str, object]], seed: int) -> list[int]:
    if len(rows) != 120:
        raise RuntimeError(f"expected 120 candidates, found {len(rows)}")
    rng = random.Random(seed)
    features = numerical_features(rows)
    selected: list[int] = []
    selected_set: set[int] = set()
    used_scaffolds: set[str] = set()
    stratum_population = Counter(str(row["stratum"]) for row in rows)
    stratum_counts: Counter[str] = Counter()

    # Seed all chemistry/size strata with a target-scale-central representative.
    for stratum in sorted(stratum_population):
        indices = [
            i for i, row in enumerate(rows)
            if row["stratum"] == stratum
            and str(row["scaffold"]) not in used_scaffolds
        ]
        if not indices:
            # A scaffold may occur in more than one size/composition stratum.
            # Preserve the hard scaffold-uniqueness gate and let max-min
            # selection cover this stratum later when a distinct scaffold is
            # available.
            continue
        center = np.median(features[indices], axis=0)
        indices.sort(key=lambda i: (
            float(np.linalg.norm(features[i] - center)),
            str(rows[i]["molecule_id"]),
        ))
        index = indices[0]
        selected.append(index)
        selected_set.add(index)
        used_scaffolds.add(str(rows[index]["scaffold"]))
        stratum_counts[stratum] += 1

    maximum_similarity = np.zeros(len(rows), dtype=np.float64)
    minimum_numeric_distance = np.full(len(rows), np.inf, dtype=np.float64)
    for index in selected:
        similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(
            rows[index]["_fingerprint"], [row["_fingerprint"] for row in rows]
        ))
        maximum_similarity = np.maximum(maximum_similarity, similarities)
        distances = np.linalg.norm(features - features[index], axis=1)
        minimum_numeric_distance = np.minimum(minimum_numeric_distance, distances)

    while len(selected) < 100:
        best: tuple[float, float, str] | None = None
        best_index: int | None = None
        for index, row in enumerate(rows):
            if index in selected_set or str(row["scaffold"]) in used_scaffolds:
                continue
            desired = 100.0 * stratum_population[str(row["stratum"])] / len(rows)
            underfill = max(0.0, desired - stratum_counts[str(row["stratum"])])
            numeric_diversity = math.tanh(float(minimum_numeric_distance[index]) / 3.0)
            score = (
                0.60 * (1.0 - float(maximum_similarity[index]))
                + 0.30 * numeric_diversity
                + 0.10 * min(underfill, 1.0)
            )
            tie = rng.random() * 1.0e-12
            key = (score + tie, -float(maximum_similarity[index]), str(row["molecule_id"]))
            if best is None or key > best:
                best = key
                best_index = index
        if best_index is None:
            raise RuntimeError("could not choose 100 candidates with unique scaffolds")
        selected.append(best_index)
        selected_set.add(best_index)
        used_scaffolds.add(str(rows[best_index]["scaffold"]))
        stratum_counts[str(rows[best_index]["stratum"])] += 1
        similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(
            rows[best_index]["_fingerprint"], [row["_fingerprint"] for row in rows]
        ))
        maximum_similarity = np.maximum(maximum_similarity, similarities)
        distances = np.linalg.norm(features - features[best_index], axis=1)
        minimum_numeric_distance = np.minimum(minimum_numeric_distance, distances)
    return sorted(selected, key=lambda i: str(rows[i]["molecule_id"]))


def partition(rows: list[dict[str, object]], seed: int) -> dict[str, list[int]]:
    features = numerical_features(rows)
    fingerprints = [row["_fingerprint"] for row in rows]
    rng = random.Random(seed + 1)
    # Choose 20 mutually diverse, target-scale-covering held-out molecules.
    center = np.median(features, axis=0)
    first = max(range(len(rows)), key=lambda i: (
        float(np.linalg.norm(features[i] - center)), str(rows[i]["molecule_id"])
    ))
    heldout = [first]
    heldout_set = {first}
    max_sim = np.asarray(DataStructs.BulkTanimotoSimilarity(fingerprints[first], fingerprints))
    min_dist = np.linalg.norm(features - features[first], axis=1)
    strata = Counter(str(rows[first]["stratum"]) for _ in [0])
    while len(heldout) < 20:
        best = None
        best_index = None
        for index, row in enumerate(rows):
            if index in heldout_set:
                continue
            rarity = 1.0 / (1.0 + strata[str(row["stratum"])])
            score = 0.60 * (1.0 - float(max_sim[index])) + 0.30 * math.tanh(float(min_dist[index]) / 3.0) + 0.10 * rarity
            key = (score + rng.random() * 1.0e-12, str(row["molecule_id"]))
            if best is None or key > best:
                best = key
                best_index = index
        assert best_index is not None
        heldout.append(best_index)
        heldout_set.add(best_index)
        strata[str(rows[best_index]["stratum"])] += 1
        max_sim = np.maximum(max_sim, np.asarray(DataStructs.BulkTanimotoSimilarity(
            fingerprints[best_index], fingerprints
        )))
        min_dist = np.minimum(min_dist, np.linalg.norm(features - features[best_index], axis=1))

    # Deterministically split the 20 held-out molecules into two balanced sets.
    heldout_sorted = sorted(heldout, key=lambda i: (
        str(rows[i]["stratum"]),
        float(rows[i]["hessian_rms_target_scale"]),
        str(rows[i]["molecule_id"]),
    ))
    validation: list[int] = []
    test: list[int] = []
    val_feature_sum = np.zeros(features.shape[1])
    test_feature_sum = np.zeros(features.shape[1])
    for index in heldout_sorted:
        if len(validation) == 10:
            target = test
        elif len(test) == 10:
            target = validation
        else:
            val_candidate = np.linalg.norm((val_feature_sum + features[index]) / (len(validation) + 1))
            test_candidate = np.linalg.norm((test_feature_sum + features[index]) / (len(test) + 1))
            target = validation if val_candidate <= test_candidate else test
        target.append(index)
        if target is validation:
            val_feature_sum += features[index]
        else:
            test_feature_sum += features[index]
    train = sorted(set(range(len(rows))) - set(validation) - set(test))
    return {
        "train": sorted(train, key=lambda i: str(rows[i]["molecule_id"])),
        "validation": sorted(validation, key=lambda i: str(rows[i]["molecule_id"])),
        "test": sorted(test, key=lambda i: str(rows[i]["molecule_id"])),
    }


def public(row: dict[str, object], role: str) -> dict[str, object]:
    return {
        key: value for key, value in row.items() if not key.startswith("_")
    } | {"role": role}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--hessian-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    source_rows = source["molecules"]
    rows = [target_record(row, args.dataset_dir, args.hessian_dir) for row in source_rows]
    chosen = choose_final100(rows, args.seed)
    final_rows = [rows[index] for index in chosen]
    roles = partition(final_rows, args.seed)
    if {key: len(value) for key, value in roles.items()} != {
        "train": 80, "validation": 10, "test": 10
    }:
        raise RuntimeError("split cardinality failure")

    role_rows = {
        role: [public(final_rows[index], role) for index in indices]
        for role, indices in roles.items()
    }
    all_public = [row for role in ("train", "validation", "test") for row in role_rows[role]]
    ids = [row["molecule_id"] for row in all_public]
    smiles = [row["canonical_smiles"] for row in all_public]
    scaffolds = [row["scaffold"] for row in all_public]
    if len(set(ids)) != 100 or len(set(smiles)) != 100 or len(set(scaffolds)) != 100:
        raise RuntimeError("final split IDs, canonical SMILES, and scaffolds must all be unique")

    # Audit the strongest cross-role fingerprint similarities without using any
    # trained-model prediction or validation/test metric.
    by_id = {str(row["molecule_id"]): row for row in final_rows}
    cross_role_max: dict[str, float] = {}
    for left_role, right_role in (("train", "validation"), ("train", "test"), ("validation", "test")):
        maximum = 0.0
        for left in role_rows[left_role]:
            left_fp = by_id[str(left["molecule_id"])]["_fingerprint"]
            right_fps = [by_id[str(right["molecule_id"])]["_fingerprint"] for right in role_rows[right_role]]
            maximum = max(maximum, max(DataStructs.BulkTanimotoSimilarity(left_fp, right_fps)))
        cross_role_max[f"{left_role}_vs_{right_role}"] = float(maximum)

    payload = {
        "algorithm_id": ALGORITHM_ID,
        "seed": args.seed,
        "candidate_manifest": str(args.candidate_manifest),
        "candidate_manifest_sha256": sha256(args.candidate_manifest),
        "dataset_dir": str(args.dataset_dir),
        "hessian_dir": str(args.hessian_dir),
        "selection_uses_model_metrics": False,
        "roles_frozen_before_training": True,
        "test_access_policy": "IDs_and_frozen_label_hashes_only_until_one_final_evaluation",
        "counts": {role: len(values) for role, values in role_rows.items()},
        "unique_ids": len(set(ids)),
        "unique_canonical_smiles": len(set(smiles)),
        "unique_scaffolds": len(set(scaffolds)),
        "cross_role_max_morgan_tanimoto": cross_role_max,
        "splits": role_rows,
    }
    atomic_json(args.output, payload)
    print(json.dumps({
        "output": str(args.output),
        "sha256": sha256(args.output),
        "counts": payload["counts"],
        "unique_scaffolds": payload["unique_scaffolds"],
        "cross_role_max_morgan_tanimoto": cross_role_max,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
