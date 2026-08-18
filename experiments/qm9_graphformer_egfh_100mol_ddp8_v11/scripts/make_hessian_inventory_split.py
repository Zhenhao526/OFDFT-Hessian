#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import zarr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labels = sorted((args.dataset_dir / "labels").glob("*.zarr.zip"))
    rows = []
    for label_path in labels:
        root = zarr.open(label_path, mode="r")
        n_scf_steps = int(root["of_labels/n_scf_steps"][()])
        rows.append([args.dataset_name, label_path.name, n_scf_steps])

    split = {
        "sizes": {"train": len(rows), "val": 0, "test": 0},
        "train": rows,
        "val": [],
        "test": [],
        "purpose": "hessian_inventory_only_not_model_selection",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    with temp.open("wb") as handle:
        pickle.dump(split, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, args.output)
    print(f"wrote={args.output} labels={len(rows)}")


if __name__ == "__main__":
    main()
