#!/usr/bin/env python3
"""Inspect HORM-derived STRUCTURES25 label zarr files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr


DEFAULT_LABEL_GLOB = (
    "/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/"
    "horm_pbe_labels/**/labels/*.zarr.zip"
)


def array_has_nan(array: zarr.Array) -> bool:
    if not np.issubdtype(array.dtype, np.number):
        return False
    data = np.asarray(array)
    return bool(np.isnan(data).any())


def numeric_range(array: zarr.Array) -> tuple[float, float] | None:
    if not np.issubdtype(array.dtype, np.number):
        return None
    data = np.asarray(array)
    if data.size == 0:
        return None
    return float(np.nanmin(data)), float(np.nanmax(data))


def walk_arrays(group: zarr.Group, prefix: str = "") -> list[tuple[str, zarr.Array]]:
    arrays = []
    for key, value in sorted(group.items()):
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, zarr.Array):
            arrays.append((path, value))
        else:
            arrays.extend(walk_arrays(value, path))
    return arrays


def inspect_label(path: Path) -> None:
    root = zarr.open(path, mode="r")
    arrays = walk_arrays(root)
    print(f"\nLABEL_FILE {path}")
    print("ALL_KEYS")
    for key, array in arrays:
        print(
            f"  {key}: shape={array.shape}, dtype={array.dtype}, "
            f"nan={array_has_nan(array)}"
        )

    if "of_labels/spatial/coeffs" in root:
        coeffs = root["of_labels/spatial/coeffs"]
        print(f"DENSITY_COEFFICIENTS shape={coeffs.shape}, nan={array_has_nan(coeffs)}")
    else:
        print("DENSITY_COEFFICIENTS missing")

    print("ENERGY_RANGES")
    for group_name in ("ks_labels/energies", "of_labels/energies"):
        if group_name not in root:
            continue
        for key, array in sorted(root[group_name].items()):
            value_range = numeric_range(array)
            if value_range is not None:
                print(f"  {group_name}/{key}: min={value_range[0]:.12g}, max={value_range[1]:.12g}")

    print("GRADIENT_LABELS")
    spatial = root.get("of_labels/spatial")
    if spatial is None:
        print("  missing of_labels/spatial")
    else:
        for key, array in sorted(spatial.items()):
            if key.startswith("grad_"):
                print(f"  of_labels/spatial/{key}: shape={array.shape}, nan={array_has_nan(array)}")

    metadata = root.get("metadata/reference")
    has_reference = metadata is not None
    has_forces = has_reference and "forces" in metadata
    has_hessian = has_reference and "hessian" in metadata
    print(f"REFERENCE_METADATA present={has_reference}, forces={has_forces}, hessian={has_hessian}")
    if has_reference:
        for key in ("source_dataset", "source_key", "source_dft_level", "energy"):
            if key in metadata:
                value = metadata[key][()]
                print(f"  metadata/reference/{key}: {value}")

    pbe_derivatives = root.get("metadata/pbe_derivatives")
    has_pbe_derivatives = pbe_derivatives is not None
    has_pbe_forces = has_pbe_derivatives and "forces" in pbe_derivatives
    has_pbe_hessian = has_pbe_derivatives and "hessian_matrix" in pbe_derivatives
    print(
        "PBE_DERIVATIVES "
        f"present={has_pbe_derivatives}, forces={has_pbe_forces}, hessian={has_pbe_hessian}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "labels",
        nargs="*",
        type=Path,
        help="Label zarr.zip files or directories containing label files.",
    )
    return parser.parse_args()


def expand_paths(paths: list[Path]) -> list[Path]:
    if not paths:
        return sorted(Path("/").glob(DEFAULT_LABEL_GLOB.lstrip("/")))
    expanded = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.zarr.zip")))
        else:
            expanded.append(path)
    return expanded


def main() -> None:
    args = parse_args()
    paths = expand_paths(args.labels)
    if not paths:
        raise SystemExit(f"No label files found. Default glob: {DEFAULT_LABEL_GLOB}")
    for path in paths:
        inspect_label(path)


if __name__ == "__main__":
    main()
