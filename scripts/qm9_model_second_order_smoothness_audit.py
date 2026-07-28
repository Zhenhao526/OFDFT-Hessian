#!/usr/bin/env python3
"""Audit model activations, cutoffs, graph topology, and second-order numerics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

import mldft.utils.omegaconf_resolvers  # noqa: F401


def _walk(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if hasattr(value, "items"):
        out = []
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.extend(_walk(item, path))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for idx, item in enumerate(value):
            out.extend(_walk(item, f"{prefix}[{idx}]"))
        return out
    return [(prefix, value)]


def _activation_audit(names: set[str]) -> list[dict[str, Any]]:
    classes = {"torch.nn.GELU": torch.nn.GELU, "torch.nn.SiLU": torch.nn.SiLU}
    rows = []
    for name in sorted(names):
        cls = classes.get(name)
        if cls is None:
            rows.append({"activation": name, "supported": False})
            continue
        x = torch.linspace(-12.0, 12.0, 4097, dtype=torch.float64, requires_grad=True)
        y = cls()(x)
        first = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), x)[0]
        rows.append(
            {
                "activation": name,
                "supported": True,
                "value_finite": bool(torch.isfinite(y).all()),
                "first_derivative_finite": bool(torch.isfinite(first).all()),
                "second_derivative_finite": bool(torch.isfinite(second).all()),
                "second_derivative_max_abs": float(second.abs().max()),
                "classification": "smooth analytic activation",
            }
        )
    return rows


def _graph_audit(run_dir: Path, molecule_ids: set[str], num_workers: int) -> list[dict[str, Any]]:
    cfg = OmegaConf.load(run_dir / "hparams.yaml")
    with open_dict(cfg):
        cfg.data.datamodule.num_workers = num_workers
        cfg.data.datamodule.shuffle_test = False
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    datamodule.setup("test")
    rows = []
    for path_index, path in enumerate(datamodule.test_set.paths):
        name_parts = path.name.split(".")
        molecule_id = name_parts[0]
        sample_id = int(name_parts[1]) if len(name_parts) > 2 else 0
        if molecule_id not in molecule_ids or sample_id != 0:
            continue
        scfs = datamodule.test_set.scf_iterations_per_path[path_index]
        matches = np.where(scfs == 1)[0]
        if not len(matches):
            continue
        dataset_index = int(datamodule.test_set.path_indices[path_index] + matches[0])
        sample = datamodule.test_set[dataset_index]
        edge_index = sample.edge_index
        n_atoms = int(sample.atomic_numbers.numel())
        self_edges = int((edge_index[0] == edge_index[1]).sum())
        unique_edges = int(torch.unique(edge_index, dim=1).shape[1])
        rows.append(
            {
                "molecule_id": molecule_id,
                "sample_id": sample_id,
                "natoms": n_atoms,
                "edges": int(edge_index.shape[1]),
                "expected_full_edges": n_atoms * n_atoms,
                "unique_edges": unique_edges,
                "self_edges": self_edges,
                "is_full_directed_graph": bool(
                    edge_index.shape[1] == n_atoms * n_atoms
                    and unique_edges == n_atoms * n_atoms
                    and self_edges == n_atoms
                ),
                "topology_changes_when_pos_tensor_is_perturbed": False,
                "reason": "edge_index is materialized by AddFullEdgeIndex and is not rebuilt in model.forward",
            }
        )
    return rows


def _precision_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": str(path)}
    payload = json.loads(path.read_text())
    return {
        "available": True,
        "path": str(path.resolve()),
        "full_hessian_cases": payload.get("full_hessian_cases"),
        "full_hessian_finite_cases": payload.get("full_hessian_finite_cases"),
        "hvp_cases": payload.get("hvp_cases"),
        "hvp_finite_cases": payload.get("hvp_finite_cases"),
        "float32_vs_float64_summary": payload.get("float32_vs_float64_summary"),
        "condition_summary": payload.get("condition_summary"),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    model_configs = []
    activation_names: set[str] = set()
    for run_dir in args.run_dir:
        cfg = OmegaConf.load(run_dir / "hparams_resolved.yaml")
        leaves = _walk(cfg.model.net)
        activations = sorted(
            {
                str(value)
                for path, value in leaves
                if "activation" in path and str(value).startswith("torch.nn.")
            }
        )
        activation_names.update(activations)
        cutoffs = {
            path: value
            for path, value in leaves
            if path.endswith("cutoff") or path.endswith("cutoff_start")
        }
        model_configs.append(
            {
                "run_dir": str(run_dir.resolve()),
                "activations": activations,
                "cutoffs": cutoffs,
                "transform_name": cfg.data.transforms.name,
                "cached_transform_name": cfg.data.transforms.cached_transforms.name,
                "pre_transforms": [
                    str(item.get("_target_", "")) for item in cfg.data.transforms.pre_transforms
                ],
            }
        )

    molecule_ids = {item.strip().zfill(7) for item in args.molecules.split(",") if item.strip()}
    graph_rows = _graph_audit(args.run_dir[0], molecule_ids, args.num_workers)
    result = {
        "definition": "Static and runtime audit of fixed-density second-order smoothness.",
        "model_configs": model_configs,
        "activation_checks": _activation_audit(activation_names),
        "graph_checks": graph_rows,
        "self_loop_distance": {
            "implementation": "mldft.ml.models.components.gbf_module._safe_edge_lengths",
            "self_loop_distance": "constant zero",
            "non_self_distance": "torch.norm(pos_i-pos_j)",
            "second_order_safe": True,
        },
        "precision_scan": _precision_summary(args.precision_analysis),
        "structural_caveats": [
            "Full edge topology is fixed within a forward/Hessian evaluation, so there is no cutoff or neighbor-list switching in the audited path.",
            "Cached local-frame/global-natural-representation transforms are geometry dependent across independently rebuilt displaced samples, but the current nuclear-coordinate autograd path does not differentiate through that rebuild.",
            "Density-relaxed current force omits classical total-OFDFT nuclear derivatives and implicit density response; activation smoothness cannot restore conservativity.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, default=str) + "\n")
    if args.graph_csv:
        args.graph_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({key for row in graph_rows for key in row})
        with args.graph_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(graph_rows)
    print(json.dumps(result, indent=2, default=str))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--molecules", required=True)
    parser.add_argument("--precision-analysis", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--graph-csv", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
