"""Select deterministic representative samples from group-safe legacy splits."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path


PARTITIONS = ("train", "val", "test")
DATASETS = ("QM9_perturbed_fock", "QMUGSBin0_perturbed_fock")
SOURCES = ("QM9_perturbed_fock", "QMUGS_perturbed_fock", "QMUGS")


def _load_assignments(split_path: Path) -> tuple[dict, dict]:
    with split_path.open("rb") as handle:
        split = pickle.load(handle)
    assignments = {}
    iterations = {}
    for partition in PARTITIONS:
        for source, filename, count in split[partition]:
            label = (str(source), str(filename))
            assignments[label] = partition
            iterations[label] = int(count)
    return assignments, iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--safe-split-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=32)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    assignments = {}
    iterations = {}
    needed = set()
    for dataset in DATASETS:
        split_path = args.safe_split_root / dataset / "split.pkl"
        assignments[dataset], iterations[dataset] = _load_assignments(split_path)
        needed.update(assignments[dataset])

    metadata = {}
    for source in SOURCES:
        for shard in range(args.num_shards):
            path = args.audit_root / "rows" / f"{source}_{shard}.jsonl.gz"
            with gzip.open(path, "rt") as handle:
                for line in handle:
                    row = json.loads(line)
                    label = (source, row["filename"])
                    if label in needed:
                        metadata[label] = {
                            "coefficients": row["coefficients"],
                            "composition": row["composition"],
                            "n_atoms": row["n_atoms"],
                            "scf_steps": row["scf_steps"],
                        }
    missing = sorted(needed - set(metadata))
    if missing:
        raise RuntimeError(f"Missing audited metadata for {len(missing)} split labels")

    report = {
        "datasets": {},
        "policy": (
            "per split minimum/maximum atom labels plus deterministic validation-only "
            "coverage for any otherwise missing element"
        ),
    }
    for dataset in DATASETS:
        selected_labels = []
        by_partition = {}

        def make_record(label, partition, role):
            item = metadata[label]
            steps = sorted({0, max(1, item["scf_steps"] // 2), item["scf_steps"] - 1})
            step_roles = {
                0: "initial_guess",
                max(1, item["scf_steps"] // 2): "intermediate_perturbed_fock",
                item["scf_steps"] - 1: "ground_state_final",
            }
            return {
                **item,
                "filename": label[1],
                "partition": partition,
                "selection_role": role,
                "source": label[0],
                "steps": [
                    {"role": step_roles[step], "scf_iteration": step} for step in steps
                ],
            }

        for partition in PARTITIONS:
            candidates = [
                label for label, value in assignments[dataset].items() if value == partition
            ]
            ordered = sorted(
                candidates,
                key=lambda label: (
                    metadata[label]["n_atoms"],
                    metadata[label]["coefficients"],
                    label,
                ),
            )
            choices = [ordered[0], ordered[-1]] if len(ordered) > 1 else ordered
            by_partition[partition] = []
            for role, label in zip(("minimum_atoms", "maximum_atoms"), choices):
                by_partition[partition].append(make_record(label, partition, role))
                selected_labels.append(label)

        all_elements = sorted(
            {
                int(element)
                for label in assignments[dataset]
                for element in metadata[label]["composition"]
            }
        )
        elements = {
            int(element)
            for label in selected_labels
            for element in metadata[label]["composition"]
        }
        for element in sorted(set(all_elements) - elements):
            candidates = sorted(
                label
                for label, partition in assignments[dataset].items()
                if partition == "val"
                and str(element) in metadata[label]["composition"]
                and label not in selected_labels
            )
            if not candidates:
                raise RuntimeError(f"No validation coverage candidate for element {element}")
            label = candidates[0]
            by_partition["val"].append(
                make_record(label, "val", f"element_coverage_Z{element}")
            )
            selected_labels.append(label)
            elements.add(element)
        elements = sorted(
            {
                int(element)
                for label in selected_labels
                for element in metadata[label]["composition"]
            }
        )
        report["datasets"][dataset] = {
            "coefficient_range": [
                min(metadata[label]["coefficients"] for label in selected_labels),
                max(metadata[label]["coefficients"] for label in selected_labels),
            ],
            "elements": elements,
            "labels": len(selected_labels),
            "n_atom_range": [
                min(metadata[label]["n_atoms"] for label in selected_labels),
                max(metadata[label]["n_atoms"] for label in selected_labels),
            ],
            "partitions": by_partition,
            "sample_instances": sum(
                len(record["steps"])
                for records in by_partition.values()
                for record in records
            ),
            "split_pickle": str(
                (args.safe_split_root / dataset / "split.pkl").resolve()
            ),
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
