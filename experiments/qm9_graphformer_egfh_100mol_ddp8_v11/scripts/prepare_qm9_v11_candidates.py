#!/usr/bin/env python3
"""Freeze a structurally diverse QM9 candidate pool without using model metrics.

The script reads the canonical 3-D structures from DeepChem's mirror of QM9,
filters malformed/duplicate molecules, computes chemistry and size descriptors,
and deterministically chooses an oversubscribed candidate set.  E/G/F/H target
scales are deliberately absent at this stage; they are incorporated only after
fresh label generation and before the final 100-molecule split is frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


ALLOWED_ELEMENTS = (1, 6, 7, 8, 9)
ALGORITHM_ID = "qm9_v11_structural_maxmin_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def heavy_bin(count: int) -> str:
    if count <= 3:
        return "01-03"
    if count <= 5:
        return "04-05"
    if count <= 7:
        return "06-07"
    return "08-09"


def composition_key(counts: Counter[int]) -> str:
    return "".join(f"{z}:{counts.get(z, 0)};" for z in ALLOWED_ELEMENTS)


def hetero_mask(counts: Counter[int]) -> str:
    present = "".join(symbol for z, symbol in ((7, "N"), (8, "O"), (9, "F")) if counts.get(z, 0))
    return present or "none"


def size_descriptors(mol: Chem.Mol) -> tuple[float, float]:
    positions = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
    centered = positions - positions.mean(axis=0, keepdims=True)
    radius_of_gyration = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    delta = positions[:, None, :] - positions[None, :, :]
    maximum_distance = float(np.sqrt(np.max(np.sum(delta * delta, axis=-1))))
    return radius_of_gyration, maximum_distance


@dataclass
class Candidate:
    molecule_id: int
    canonical_smiles: str
    scaffold: str
    atom_count: int
    heavy_atom_count: int
    composition: str
    hetero_mask: str
    heavy_bin: str
    radius_of_gyration_angstrom: float
    max_pair_distance_angstrom: float
    fingerprint: DataStructs.ExplicitBitVect

    @property
    def stratum(self) -> str:
        return f"{self.heavy_bin}|{self.hetero_mask}"

    def public_dict(self) -> dict[str, object]:
        return {
            "molecule_id": f"{self.molecule_id:07d}",
            "canonical_smiles": self.canonical_smiles,
            "scaffold": self.scaffold,
            "atom_count": self.atom_count,
            "heavy_atom_count": self.heavy_atom_count,
            "composition": self.composition,
            "hetero_mask": self.hetero_mask,
            "heavy_bin": self.heavy_bin,
            "stratum": self.stratum,
            "radius_of_gyration_angstrom": self.radius_of_gyration_angstrom,
            "max_pair_distance_angstrom": self.max_pair_distance_angstrom,
        }


def molecule_id(mol: Chem.Mol) -> int:
    name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
    token = name.rsplit("_", 1)[-1]
    return int(token)


def describe(mol: Chem.Mol) -> Candidate:
    mid = molecule_id(mol)
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    counts = Counter(atomic_numbers)
    if any(z not in ALLOWED_ELEMENTS for z in counts):
        raise ValueError("unsupported_element")
    if mol.GetNumConformers() != 1:
        raise ValueError("missing_or_multiple_conformer")
    positions = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
    if positions.shape != (len(atomic_numbers), 3) or not np.isfinite(positions).all():
        raise ValueError("invalid_coordinates")
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    if not smiles:
        raise ValueError("empty_smiles")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
    if not scaffold:
        degrees = "-".join(str(value) for value in sorted(atom.GetDegree() for atom in mol.GetAtoms()))
        scaffold = f"ACYCLIC:{sum(z > 1 for z in atomic_numbers)}:{hetero_mask(counts)}:{degrees}"
    rg, maximum_distance = size_descriptors(mol)
    fingerprint = AllChem.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True).GetFingerprint(mol)
    return Candidate(
        molecule_id=mid,
        canonical_smiles=smiles,
        scaffold=scaffold,
        atom_count=len(atomic_numbers),
        heavy_atom_count=sum(z > 1 for z in atomic_numbers),
        composition=composition_key(counts),
        hetero_mask=hetero_mask(counts),
        heavy_bin=heavy_bin(sum(z > 1 for z in atomic_numbers)),
        radius_of_gyration_angstrom=rg,
        max_pair_distance_angstrom=maximum_distance,
        fingerprint=fingerprint,
    )


def load_candidates(sdf_path: Path) -> tuple[list[Candidate], dict[str, int]]:
    RDLogger.DisableLog("rdApp.*")
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    candidates: list[Candidate] = []
    failures: Counter[str] = Counter()
    seen_smiles: set[str] = set()
    seen_ids: set[int] = set()
    for index, mol in enumerate(supplier):
        if mol is None:
            failures["rdkit_parse_failure"] += 1
            continue
        try:
            item = describe(mol)
        except Exception as exc:  # fail closed and count the exact reason
            failures[str(exc)] += 1
            continue
        if item.molecule_id in seen_ids:
            failures["duplicate_molecule_id"] += 1
            continue
        if item.canonical_smiles in seen_smiles:
            failures["duplicate_canonical_smiles"] += 1
            continue
        seen_ids.add(item.molecule_id)
        seen_smiles.add(item.canonical_smiles)
        candidates.append(item)
        if (index + 1) % 10000 == 0:
            print(json.dumps({"parsed": index + 1, "eligible": len(candidates)}, sort_keys=True), flush=True)
    return candidates, dict(sorted(failures.items()))


def choose(candidates: list[Candidate], count: int, seed: int) -> list[Candidate]:
    if count <= 0 or count > len(candidates):
        raise ValueError("selection count must be within candidate population")
    rng = random.Random(seed)
    by_stratum: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(candidates):
        by_stratum[item.stratum].append(index)

    selected: list[int] = []
    selected_set: set[int] = set()
    selected_scaffolds: Counter[str] = Counter()
    # Seed every populated chemistry/size stratum with a deterministic molecule
    # nearest that stratum's median geometric size.
    for stratum in sorted(by_stratum):
        indices = by_stratum[stratum]
        median_rg = float(np.median([candidates[i].radius_of_gyration_angstrom for i in indices]))
        ranked = sorted(
            indices,
            key=lambda i: (
                abs(candidates[i].radius_of_gyration_angstrom - median_rg),
                rng.random(),
                candidates[i].molecule_id,
            ),
        )
        index = ranked[0]
        selected.append(index)
        selected_set.add(index)
        selected_scaffolds[candidates[index].scaffold] += 1
        if len(selected) == count:
            break

    maximum_similarity = np.zeros(len(candidates), dtype=np.float64)
    for index in selected:
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(candidates[index].fingerprint, [c.fingerprint for c in candidates]),
            dtype=np.float64,
        )
        np.maximum(maximum_similarity, similarities, out=maximum_similarity)

    stratum_counts = Counter(candidates[i].stratum for i in selected)
    population = Counter(item.stratum for item in candidates)
    target_share = {key: math.sqrt(value) for key, value in population.items()}
    normalizer = sum(target_share.values())
    target_share = {key: value / normalizer for key, value in target_share.items()}

    while len(selected) < count:
        best_index = None
        best_score = None
        for index, item in enumerate(candidates):
            if index in selected_set:
                continue
            diversity = 1.0 - maximum_similarity[index]
            desired = target_share[item.stratum] * (len(selected) + 1)
            underfill = max(0.0, desired - stratum_counts[item.stratum])
            scaffold_bonus = 0.12 if selected_scaffolds[item.scaffold] == 0 else 0.0
            score = diversity + 0.08 * underfill + scaffold_bonus
            key = (score, -item.molecule_id)
            if best_score is None or key > best_score:
                best_score = key
                best_index = index
        assert best_index is not None
        selected.append(best_index)
        selected_set.add(best_index)
        item = candidates[best_index]
        stratum_counts[item.stratum] += 1
        selected_scaffolds[item.scaffold] += 1
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(item.fingerprint, [c.fingerprint for c in candidates]),
            dtype=np.float64,
        )
        np.maximum(maximum_similarity, similarities, out=maximum_similarity)
        print(json.dumps({"selected": len(selected), "molecule_id": item.molecule_id, "score": best_score[0]}), flush=True)
    return [candidates[index] for index in selected]


def write_xyz_subset(sdf_path: Path, selected: list[Candidate], raw_dir: Path) -> list[dict[str, object]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    selected_by_id = {item.molecule_id: item for item in selected}
    written: list[dict[str, object]] = []
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    for mol in supplier:
        if mol is None:
            continue
        mid = molecule_id(mol)
        item = selected_by_id.get(mid)
        if item is None:
            continue
        path = raw_dir / f"dsgdb9nsd_{mid:06d}.xyz"
        positions = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
        lines = [str(mol.GetNumAtoms()), f"gdb {mid} {item.canonical_smiles}"]
        for atom, xyz in zip(mol.GetAtoms(), positions, strict=True):
            lines.append(f"{atom.GetSymbol():2s} {xyz[0]: .12f} {xyz[1]: .12f} {xyz[2]: .12f}")
        path.write_text("\n".join(lines) + "\n")
        written.append({"molecule_id": f"{mid:07d}", "raw_xyz": str(path), "raw_xyz_sha256": sha256(path)})
    if len(written) != len(selected):
        missing = sorted(set(selected_by_id).difference(int(row["molecule_id"]) for row in written))
        raise RuntimeError(f"failed to write selected structures: {missing}")
    return sorted(written, key=lambda row: row["molecule_id"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf", type=Path, required=True)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260817)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if str(output_root).startswith("/scratch") or "/scratch/" in str(output_root):
        raise SystemExit("v11 local-only policy forbids /scratch")
    output_root.mkdir(parents=True, exist_ok=True)
    candidates, failures = load_candidates(args.sdf)
    selected = choose(candidates, args.count, args.seed)
    raw_records = write_xyz_subset(args.sdf, selected, output_root / "raw")
    raw_by_id = {row["molecule_id"]: row for row in raw_records}
    records = []
    for item in selected:
        row = item.public_dict()
        row.update(raw_by_id[row["molecule_id"]])
        records.append(row)
    records.sort(key=lambda row: row["molecule_id"])
    manifest = {
        "schema_version": 1,
        "algorithm_id": ALGORITHM_ID,
        "seed": args.seed,
        "selection_count": args.count,
        "eligible_count": len(candidates),
        "source_zip": str(args.source_zip.resolve()),
        "source_zip_sha256": sha256(args.source_zip),
        "source_sdf": str(args.sdf.resolve()),
        "source_sdf_sha256": sha256(args.sdf),
        "allowed_atomic_numbers": list(ALLOWED_ELEMENTS),
        "selection_uses_model_metrics": False,
        "selection_uses_validation_or_test_metrics": False,
        "failures": failures,
        "molecules": records,
    }
    manifest_path = output_root / "candidate120_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    csv_path = output_root / "candidate120_manifest.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    registration = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "csv": str(csv_path),
        "csv_sha256": sha256(csv_path),
    }
    (output_root / "registration.json").write_text(json.dumps(registration, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**registration, "eligible_count": len(candidates), "selection_count": len(records)}, indent=2))


if __name__ == "__main__":
    main()
