#!/usr/bin/env python3
"""Quantify the rigid-mode floor imposed by a zero-force anchored scalar correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.qm9_complete_total_local_scalar_full_hessian_capacity import (
    _load_protocol,
    _load_selected_parents,
    _sha256,
)


def _rigid_basis(positions: np.ndarray) -> np.ndarray:
    centered = positions - positions.mean(axis=0, keepdims=True)
    atom_count = positions.shape[0]
    columns = []
    for axis in range(3):
        translation = np.zeros((atom_count, 3), dtype=np.float64)
        translation[:, axis] = 1.0
        columns.append(translation.reshape(-1))
    generators = (
        np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
        np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    for generator in generators:
        columns.append((centered @ generator.T).reshape(-1))
    matrix = np.stack(columns, axis=1)
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.sum(singular > 1e-10 * singular[0]))
    return u[:, :rank]


def run(args: argparse.Namespace) -> dict[str, object]:
    protocol, _ = _load_protocol(args.protocol, args.arm_id, "smoke")
    parents, provenance = _load_selected_parents(protocol, torch.device("cpu"))
    rows = []
    for parent in parents:
        positions = parent.positions.detach().cpu().numpy()
        basis = _rigid_basis(positions)
        projector = np.eye(parent.pbe_hessian.shape[0]) - basis @ basis.T
        target = parent.pbe_hessian - parent.source_hessian_symmetric
        admissible = projector @ target @ projector
        irreducible = target - admissible
        pbe_norm = max(np.linalg.norm(parent.pbe_hessian), np.finfo(float).tiny)
        rows.append(
            {
                "molecule_id": parent.molecule_id,
                "natoms": parent.natoms,
                "rigid_rank": int(basis.shape[1]),
                "zero_force_anchored_minimum_relative_frobenius": float(
                    np.linalg.norm(irreducible) / pbe_norm
                ),
                "target_rigid_response_relative_frobenius": float(
                    np.linalg.norm(target @ basis) / pbe_norm
                ),
                "force_correction_norm_hartree_per_bohr": float(
                    np.linalg.norm(parent.pbe_force - parent.source_force)
                ),
            }
        )
    floors = np.asarray(
        [row["zero_force_anchored_minimum_relative_frobenius"] for row in rows]
    )
    result = {
        "definition": "Orthogonal projection lower bound for any symmetric scalar Hessian correction whose value and gradient are zero at R0 and whose scalar is translation/rotation invariant.",
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "per_parent": rows,
        "minimum_relative_frobenius_floor": {
            "median": float(np.median(floors)),
            "max": float(np.max(floors)),
        },
        "stable5_gate_mathematically_possible": bool(np.max(floors) <= 0.05),
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        **provenance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm-id", default="L1_invariant_message_passing")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

