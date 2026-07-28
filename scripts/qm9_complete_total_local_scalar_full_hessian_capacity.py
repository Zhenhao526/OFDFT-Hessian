#!/usr/bin/env python3
"""Fit preregistered stable5 local scalar models to complete-total full Hessians."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
import zarr

from mldft.ml.models.components.local_body_order_residual import (
    BodyOrderTopology,
    LocalBodyOrderResidual,
    build_body_order_topology,
)
from mldft.ml.models.components.local_angular_scalar_residual import (
    LocalAngularScalarResidual,
)
from mldft.ml.models.components.local_message_passing_residual import (
    LocalMessagePassingResidual,
)
from mldft.ml.models.components.local_mace_scalar_residual import LocalMACEScalarResidual
from mldft.ml.models.components.local_mace_invariant_readout import (
    LocalMACEInvariantReadoutResidual,
)
from mldft.ml.models.components.local_equivariant_scalar_residual import (
    LocalEquivariantScalarResidual,
)
from mldft.ml.models.components.local_equivariant_quadratic_scalar_residual import (
    LocalEquivariantQuadraticScalarResidual,
)
from mldft.ml.models.components.local_tensor_equivariant_scalar_residual import (
    LocalTensorEquivariantScalarResidual,
)
from mldft.ofdft.complete_total_training import (
    assign_multi_task_pcgrad,
    assign_two_task_pcgrad,
)

try:
    from scripts.qm9_complete_total_geometry_three_body_capacity import hessian_metrics
except ModuleNotFoundError:
    from qm9_complete_total_geometry_three_body_capacity import hessian_metrics


@dataclass
class CapacityParent:
    molecule_id: str
    natoms: int
    atomic_numbers: torch.Tensor
    positions: torch.Tensor
    topology: BodyOrderTopology
    pbe_energy: float
    pbe_force: np.ndarray
    pbe_hessian: np.ndarray
    source_energy: float
    source_force: np.ndarray
    source_hessian_raw: np.ndarray
    source_hessian_symmetric: np.ndarray


ScalarModel = (
    LocalBodyOrderResidual
    | LocalMessagePassingResidual
    | LocalAngularScalarResidual
    | LocalEquivariantScalarResidual
    | LocalEquivariantQuadraticScalarResidual
    | LocalMACEScalarResidual
    | LocalMACEInvariantReadoutResidual
)


def _build_single_group_optimizer(
    parameters: list[torch.nn.Parameter], arm: dict[str, Any]
) -> torch.optim.Optimizer:
    optimizer_name = str(arm.get("optimizer", "adam")).lower()
    learning_rate = float(arm["learning_rate"])
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    if optimizer_name == "sgd":
        momentum = float(arm.get("momentum", 0.0))
        if momentum != 0.0:
            raise ValueError("PCGrad-compatible SGD requires zero momentum")
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=0.0)
    raise ValueError(f"unsupported optimizer: {optimizer_name}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_protocol(path: Path, arm_id: str, run_mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = yaml.safe_load(path.read_text())
    if protocol.get("test100_access_allowed") is not False:
        raise ValueError("protocol does not freeze Test100")
    if int(protocol.get("test100_evaluations_used", -1)) != 0:
        raise ValueError("protocol Test100 count is not zero")
    if protocol.get("stage") not in {
        "stable5_fit_only_local_scalar_capacity",
        "stable5_fit_only_joint_scalar_capacity",
        "stable5_fit_only_three_task_gradient_conflict_audit",
        "stable5_fit_only_angular_scalar_capacity",
        "stable5_fit_only_nonlinear_local_scalar_capacity",
        "stable5_fit_only_bounded_equivariant_scalar_capacity",
        "stable5_fit_only_mature_higher_body_scalar_capacity",
        "stable5_fit_only_force_secant_gradient_conflict_intervention",
    }:
        raise ValueError("unexpected protocol stage")
    if run_mode not in {"smoke", "formal"}:
        raise ValueError("run_mode must be smoke or formal")
    arms = {str(row["id"]): row for row in protocol["arms"]}
    if arm_id not in arms:
        raise ValueError(f"arm is not preregistered: {arm_id}")
    inputs = protocol["inputs"]
    baseline = Path(inputs["baseline_manifest"])
    stable5_protocol = Path(inputs["stable5_protocol"])
    if _sha256(baseline) != str(inputs["baseline_manifest_sha256"]):
        raise ValueError("baseline manifest hash drift")
    if _sha256(stable5_protocol) != str(inputs["stable5_protocol_sha256"]):
        raise ValueError("stable5 protocol hash drift")
    gradient_audit_value = inputs.get("gradient_audit")
    if gradient_audit_value is not None:
        gradient_audit = Path(gradient_audit_value)
        if _sha256(gradient_audit) != str(inputs["gradient_audit_sha256"]):
            raise ValueError("gradient audit hash drift")
    diagnostic_path_value = inputs.get("post_v4_diagnostic_summary")
    if diagnostic_path_value is not None:
        diagnostic_path = Path(diagnostic_path_value)
        if _sha256(diagnostic_path) != str(inputs["post_v4_diagnostic_summary_sha256"]):
            raise ValueError("post-v4 diagnostic summary hash drift")
        diagnostic = json.loads(diagnostic_path.read_text())
        if diagnostic.get("diagnosis") != inputs.get("required_post_v4_diagnosis"):
            raise ValueError("post-v4 diagnosis does not authorize representation change")
        if diagnostic.get("parent_cv_design_authorized") is not False:
            raise ValueError("post-v4 diagnostic unexpectedly authorizes parent CV")
        if diagnostic.get("test100_accessed") is not False:
            raise ValueError("post-v4 diagnostic does not certify frozen Test100")
    return protocol, arms[arm_id]


def _load_selected_parents(
    protocol: dict[str, Any], device: torch.device
) -> tuple[list[CapacityParent], dict[str, Any]]:
    inputs = protocol["inputs"]
    requested = [str(value) for value in inputs["parents"]]
    if len(requested) != 5 or len(set(requested)) != len(requested):
        raise ValueError("capacity protocol must contain five unique parents")
    manifest_path = Path(inputs["baseline_manifest"])
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("test100_accessed") is not False:
        raise ValueError("baseline manifest does not certify frozen Test100")
    if manifest.get("source_split_sha256") != inputs["source_split_sha256"]:
        raise ValueError("source split hash drift")
    rows = {str(row["molecule_id"]): row for row in manifest["parents"]}
    missing = sorted(set(requested) - set(rows))
    if missing:
        raise ValueError(f"stable5 parents missing from baseline: {missing}")

    parents = []
    opened_label_paths = []
    opened_capacity_arrays = []
    for molecule_id in requested:
        row = rows[molecule_id]
        label_path = Path(row["label_path"])
        capacity_array = Path(row["capacity_array"])
        label = zarr.open(label_path.as_posix(), mode="r")
        atomic_numbers = np.asarray(label["geometry/atomic_numbers"], dtype=np.int64)
        positions = np.asarray(label["geometry/atom_pos"], dtype=np.float64)
        pbe_force = np.asarray(
            label["metadata/pbe_derivatives/forces"], dtype=np.float64
        )
        energy_trace = np.asarray(label["ks_labels/energies/e_tot"], dtype=np.float64)
        has_energy = np.asarray(
            label["ks_labels/energies/has_energy_label"], dtype=np.bool_
        )
        with np.load(capacity_array) as payload:
            source_hessian = np.asarray(payload["predicted_hessian"], dtype=np.float64)
            pbe_hessian = np.asarray(payload["pbe_hessian"], dtype=np.float64)
            source_force = np.asarray(payload["predicted_base_force"], dtype=np.float64)
        symmetric_source = 0.5 * (source_hessian + source_hessian.T)
        atomic_tensor = torch.as_tensor(atomic_numbers, dtype=torch.long, device=device)
        position_tensor = torch.as_tensor(positions, dtype=torch.float64, device=device)
        parents.append(
            CapacityParent(
                molecule_id=molecule_id,
                natoms=int(atomic_numbers.size),
                atomic_numbers=atomic_tensor,
                positions=position_tensor,
                topology=build_body_order_topology(
                    torch.as_tensor(atomic_numbers, dtype=torch.long),
                    torch.as_tensor(positions, dtype=torch.float64),
                    bond_scale=1.25,
                ),
                pbe_energy=float(energy_trace[has_energy][-1]),
                pbe_force=pbe_force,
                pbe_hessian=pbe_hessian,
                source_energy=float(row["baseline_total_energy_hartree"]),
                source_force=source_force,
                source_hessian_raw=source_hessian,
                source_hessian_symmetric=symmetric_source,
            )
        )
        opened_label_paths.append(label_path.resolve().as_posix())
        opened_capacity_arrays.append(capacity_array.resolve().as_posix())
    return parents, {
        "selected_parent_ids": requested,
        "opened_label_paths": opened_label_paths,
        "opened_capacity_arrays": opened_capacity_arrays,
        "unselected_parent_artifacts_opened": 0,
        "baseline_manifest": manifest_path.resolve().as_posix(),
        "baseline_manifest_sha256": _sha256(manifest_path),
        "source_split_sha256": manifest["source_split_sha256"],
    }


def _build_model_and_optimizer(
    arm: dict[str, Any], device: torch.device
) -> tuple[ScalarModel, torch.optim.Optimizer, dict[str, Any]]:
    architecture = str(arm["architecture"])
    metadata_extra: dict[str, Any] = {}
    if architecture == "typed_pair_triplet_torsion":
        model: ScalarModel = LocalBodyOrderResidual(
            pair_hidden_size=int(arm["pair_hidden_size"]),
            triplet_hidden_size=int(arm["triplet_hidden_size"]),
            torsion_hidden_size=int(arm["torsion_hidden_size"]),
            seed=int(arm["seed"]),
            canceling_output=float(arm["canceling_output"]),
            cutoff_bohr=float(arm["cutoff_bohr"]),
        ).to(device)
        optimizer = torch.optim.Adam(
            [
                {
                    "params": [*model.pair.parameters(), *model.triplet.parameters()],
                    "lr": float(arm["learning_rate"]),
                },
                {
                    "params": model.torsion.parameters(),
                    "lr": float(arm["torsion_learning_rate"]),
                },
            ]
        )
    elif architecture == "invariant_distance_message_passing":
        model = LocalMessagePassingResidual(
            hidden_size=int(arm["hidden_size"]),
            radial_size=int(arm["radial_size"]),
            message_layers=int(arm["message_layers"]),
            cutoff_bohr=float(arm["cutoff_bohr"]),
            seed=int(arm["seed"]),
            activation=str(arm.get("activation", "softplus")),
            softplus_beta=float(arm.get("softplus_beta", 1.0)),
        ).to(device)
        optimizer = torch.optim.Adam(
            model.trainable_parameters(), lr=float(arm["learning_rate"])
        )
    elif architecture == "local_angular_symmetry_network":
        model = LocalAngularScalarResidual(
            hidden_size=int(arm["hidden_size"]),
            radial_size=int(arm["radial_size"]),
            angular_order=int(arm["angular_order"]),
            cutoff_bohr=float(arm["cutoff_bohr"]),
            seed=int(arm["seed"]),
            activation=str(arm.get("activation", "tanh")),
            softplus_beta=float(arm.get("softplus_beta", 1.0)),
            radial_feature_scale=float(arm.get("radial_feature_scale", 4.0)),
            angular_feature_scale=float(arm.get("angular_feature_scale", 16.0)),
        ).to(device)
        optimizer = torch.optim.Adam(
            model.trainable_parameters(), lr=float(arm["learning_rate"])
        )
    elif architecture == "bounded_equivariant_vector_message_passing":
        model = LocalEquivariantScalarResidual(
            hidden_size=int(arm["hidden_size"]),
            radial_size=int(arm["radial_size"]),
            interaction_layers=int(arm["interaction_layers"]),
            cutoff_bohr=float(arm["cutoff_bohr"]),
            seed=int(arm["seed"]),
        ).to(device)
        optimizer = torch.optim.Adam(
            model.trainable_parameters(), lr=float(arm["learning_rate"])
        )
    elif architecture == "tensor_product_quadrupole_message_passing":
        model = LocalTensorEquivariantScalarResidual(
            scalar_channels=int(arm["scalar_channels"]),
            vector_channels=int(arm["vector_channels"]),
            tensor_channels=int(arm["tensor_channels"]),
            radial_size=int(arm["radial_size"]),
            radial_hidden_size=int(arm["radial_hidden_size"]),
            interaction_layers=int(arm["interaction_layers"]),
            cutoff_bohr=float(arm["cutoff_bohr"]),
            output_scale=float(arm.get("output_scale", 1.0)),
            seed=int(arm["seed"]),
        ).to(device)
        optimizer = torch.optim.Adam(
            model.trainable_parameters(), lr=float(arm["learning_rate"])
        )
    elif architecture == "reference_local_equivariant_quadratic_scalar":
        model = LocalEquivariantQuadraticScalarResidual(
            hidden_size=int(arm["hidden_size"]),
            radial_size=int(arm["radial_size"]),
            cutoff_bohr=float(arm["cutoff_bohr"]),
            coefficient_scale=float(arm.get("coefficient_scale", 1.0)),
            seed=int(arm["seed"]),
        ).to(device)
        optimizer = torch.optim.Adam(
            model.trainable_parameters(), lr=float(arm["learning_rate"])
        )
    elif architecture == "mace_higher_body_scalar":
        model = LocalMACEScalarResidual(
            hidden_channels=int(arm["hidden_channels"]),
            mlp_channels=int(arm["mlp_channels"]),
            max_ell=int(arm["max_ell"]),
            correlation=int(arm["correlation"]),
            num_interactions=int(arm["num_interactions"]),
            num_bessel=int(arm["num_bessel"]),
            cutoff_polynomial_order=int(arm["cutoff_polynomial_order"]),
            cutoff_bohr=float(arm["cutoff_bohr"]),
            avg_num_neighbors=float(arm["avg_num_neighbors"]),
            output_scale=float(arm.get("output_scale", 1.0)),
            seed=int(arm["seed"]),
        ).to(device)
        optimizer = _build_single_group_optimizer(model.trainable_parameters(), arm)
    elif architecture == "mace_frozen_invariant_random_readout_scalar":
        model = LocalMACEInvariantReadoutResidual(
            hidden_channels=int(arm["hidden_channels"]),
            mlp_channels=int(arm["mlp_channels"]),
            max_ell=int(arm["max_ell"]),
            correlation=int(arm["correlation"]),
            num_interactions=int(arm["num_interactions"]),
            num_bessel=int(arm["num_bessel"]),
            cutoff_polynomial_order=int(arm["cutoff_polynomial_order"]),
            cutoff_bohr=float(arm["cutoff_bohr"]),
            avg_num_neighbors=float(arm["avg_num_neighbors"]),
            feature_mode=str(arm.get("feature_mode", "scalar")),
            random_feature_width=int(arm["random_feature_width"]),
            random_feature_input_scale=float(
                arm.get("random_feature_input_scale", 1.0)
            ),
            output_scale=float(arm.get("output_scale", 1.0)),
            seed=int(arm["seed"]),
        ).to(device)
        backbone_checkpoint = Path(str(arm["backbone_checkpoint"]))
        expected_backbone_sha256 = str(arm["backbone_checkpoint_sha256"])
        if _sha256(backbone_checkpoint) != expected_backbone_sha256:
            raise ValueError("MACE invariant-readout backbone checkpoint hash drift")
        source = torch.load(backbone_checkpoint, map_location=device, weights_only=False)
        if source.get("test100_accessed") is not False:
            raise ValueError("MACE invariant-readout backbone accessed Test100")
        if source.get("arm_id") != str(arm["backbone_arm_id"]):
            raise ValueError("MACE invariant-readout backbone arm mismatch")
        expected_parent_ids = [str(value) for value in arm["backbone_parent_ids"]]
        source_parent_ids = [
            str(value) for value in source.get("selected_parent_ids", [])
        ]
        if source_parent_ids != expected_parent_ids:
            raise ValueError("MACE invariant-readout backbone parent set mismatch")
        model.load_backbone_state_dict(source["state_dict"])
        optimizer = torch.optim.Adam(
            model.trainable_parameters(), lr=float(arm["learning_rate"])
        )
        metadata_extra = {
            "backbone_checkpoint": backbone_checkpoint.resolve().as_posix(),
            "backbone_checkpoint_sha256": expected_backbone_sha256,
            "backbone_step": int(source["step"]),
            "backbone_arm_id": source["arm_id"],
            "backbone_parent_ids": source_parent_ids,
        }
    else:
        raise ValueError(f"unsupported architecture: {architecture}")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    metadata = {
        "architecture": architecture,
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in trainable)),
        "total_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        **metadata_extra,
    }
    return model, optimizer, metadata


def _correction(
    model: ScalarModel,
    parent: CapacityParent,
    *,
    create_parameter_graph: bool,
    anchor_energy_force: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reference = parent.positions if anchor_energy_force else None
    if isinstance(
        model,
        (
            LocalMessagePassingResidual,
            LocalAngularScalarResidual,
            LocalEquivariantScalarResidual,
            LocalTensorEquivariantScalarResidual,
            LocalEquivariantQuadraticScalarResidual,
            LocalMACEScalarResidual,
            LocalMACEInvariantReadoutResidual,
        ),
    ):
        return model.energy_force_hessian(
            parent.positions,
            parent.atomic_numbers,
            parent.topology,
            create_parameter_graph=create_parameter_graph,
            reference_positions_bohr=reference,
        )
    return model.energy_force_hessian(
        parent.positions,
        parent.topology,
        create_parameter_graph=create_parameter_graph,
        reference_positions_bohr=reference,
    )


def _full_hessian_loss(
    predicted: torch.Tensor,
    reference: torch.Tensor,
    *,
    absolute_scale: float,
    relative_fraction: float,
    reference_floor: float,
) -> torch.Tensor:
    error = predicted - reference
    absolute = torch.mean((error / absolute_scale) ** 2)
    relative = torch.sum(error * error) / torch.clamp(
        torch.sum(reference * reference), min=reference_floor**2
    )
    return (1.0 - relative_fraction) * absolute + relative_fraction * relative


def _task_gradient_norm(
    loss: torch.Tensor, parameters: list[torch.nn.Parameter]
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    norm_squared = loss.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            norm_squared = norm_squared + torch.sum(gradient * gradient)
    return torch.sqrt(norm_squared.clamp_min(torch.finfo(loss.dtype).tiny))


def _separate_gradnorm_target_multiplier(
    task_name: str,
    *,
    hessian_warmup: float,
    preserve_hessian_warmup: bool,
) -> float:
    if not 0.0 <= hessian_warmup <= 1.0:
        raise ValueError("Hessian warm-up must be in [0, 1]")
    if task_name == "hessian" and preserve_hessian_warmup:
        return hessian_warmup
    return 1.0


def _parent_loss(
    model: ScalarModel,
    parent: CapacityParent,
    training: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    correction_energy, correction_force, correction_hessian = _correction(
        model,
        parent,
        create_parameter_graph=True,
        anchor_energy_force=bool(training.get("anchor_energy_force", True)),
    )
    source_force = torch.as_tensor(
        parent.source_force, dtype=torch.float64, device=parent.positions.device
    )
    source_hessian = torch.as_tensor(
        parent.source_hessian_symmetric,
        dtype=torch.float64,
        device=parent.positions.device,
    )
    pbe_force = torch.as_tensor(
        parent.pbe_force, dtype=torch.float64, device=parent.positions.device
    )
    pbe_hessian = torch.as_tensor(
        parent.pbe_hessian, dtype=torch.float64, device=parent.positions.device
    )
    predicted_energy = correction_energy + parent.source_energy
    predicted_force = correction_force + source_force
    predicted_hessian = correction_hessian + source_hessian
    energy_loss = (
        (predicted_energy - parent.pbe_energy) / float(training["energy_scale"])
    ) ** 2
    force_loss = torch.mean(
        ((predicted_force - pbe_force) / float(training["force_scale"])) ** 2
    )
    hessian_loss = _full_hessian_loss(
        predicted_hessian,
        pbe_hessian,
        absolute_scale=float(training["absolute_hessian_scale"]),
        relative_fraction=float(training["relative_loss_fraction"]),
        reference_floor=float(training["hessian_reference_floor"]),
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameter_loss = torch.mean(
        torch.stack([torch.mean(parameter * parameter) for parameter in trainable])
    )
    total = (
        float(training["lambda_energy"]) * energy_loss
        + float(training["lambda_force"]) * force_loss
        + float(training["lambda_hessian"]) * hessian_loss
        + float(training["lambda_parameter"]) * parameter_loss
    )
    return total, {
        "energy_loss": energy_loss,
        "force_loss": force_loss,
        "hessian_loss": hessian_loss,
        "parameter_loss": parameter_loss,
    }


def _distribution(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(np.max(values)),
    }


def _evaluate(
    model: ScalarModel,
    parents: list[CapacityParent],
    gate: dict[str, Any],
    *,
    anchor_energy_force: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    rows = []
    arrays = {}
    for parent in parents:
        correction_energy, correction_force, correction_hessian = _correction(
            model,
            parent,
            create_parameter_graph=False,
            anchor_energy_force=anchor_energy_force,
        )
        correction_energy_value = float(correction_energy.detach().cpu())
        correction_force_array = correction_force.detach().cpu().numpy()
        correction_hessian_array = correction_hessian.detach().cpu().numpy()
        predicted_energy = parent.source_energy + correction_energy_value
        predicted_force = parent.source_force + correction_force_array
        predicted_hessian = parent.source_hessian_symmetric + correction_hessian_array
        source_energy_error = abs(parent.source_energy - parent.pbe_energy)
        source_force_error = float(np.mean(np.abs(parent.source_force - parent.pbe_force)))
        row = {
            "molecule_id": parent.molecule_id,
            "natoms": parent.natoms,
            "energy_abs_error_hartree": abs(predicted_energy - parent.pbe_energy),
            "source_energy_abs_error_hartree": source_energy_error,
            "force_mae_hartree_per_bohr": float(
                np.mean(np.abs(predicted_force - parent.pbe_force))
            ),
            "source_force_mae_hartree_per_bohr": source_force_error,
            "correction_energy_abs_hartree": abs(correction_energy_value),
            "correction_force_max_abs_hartree_per_bohr": float(
                np.max(np.abs(correction_force_array))
            ),
            "source_raw_antisymmetric_over_symmetric_frobenius": float(
                np.linalg.norm(
                    0.5 * (parent.source_hessian_raw - parent.source_hessian_raw.T)
                )
                / max(
                    np.linalg.norm(parent.source_hessian_symmetric),
                    np.finfo(float).tiny,
                )
            ),
            **hessian_metrics(predicted_hessian, parent.pbe_hessian),
        }
        source_hessian = hessian_metrics(
            parent.source_hessian_symmetric, parent.pbe_hessian
        )
        row["source_relative_frobenius"] = source_hessian["relative_frobenius"]
        rows.append(row)
        arrays[parent.molecule_id] = {
            "predicted_force": predicted_force,
            "predicted_hessian": predicted_hessian,
            "source_hessian_symmetric": parent.source_hessian_symmetric,
            "source_hessian_raw": parent.source_hessian_raw,
            "pbe_hessian": parent.pbe_hessian,
        }

    hessian_distribution = _distribution(rows, "relative_frobenius")
    source_distribution = _distribution(rows, "source_relative_frobenius")
    energy_distribution = _distribution(rows, "energy_abs_error_hartree")
    source_energy_distribution = _distribution(rows, "source_energy_abs_error_hartree")
    force_distribution = _distribution(rows, "force_mae_hartree_per_bohr")
    source_force_distribution = _distribution(rows, "source_force_mae_hartree_per_bohr")
    energy_ratio = energy_distribution["median"] / max(
        source_energy_distribution["median"], np.finfo(float).tiny
    )
    force_ratio = force_distribution["median"] / max(
        source_force_distribution["median"], np.finfo(float).tiny
    )
    max_asymmetry = max(
        float(row["antisymmetric_over_symmetric_frobenius"]) for row in rows
    )
    checks = {
        "training_median_relative_frobenius": hessian_distribution["median"]
        <= float(gate["training_median_relative_frobenius_max"]),
        "training_all_parent_relative_frobenius": hessian_distribution["max"]
        <= float(gate["training_all_parent_relative_frobenius_max"]),
        "energy_median_ratio_to_source": energy_ratio
        <= float(gate["energy_median_ratio_to_source_max"]),
        "force_median_ratio_to_source": force_ratio
        <= float(gate["force_median_ratio_to_source_max"]),
        "antisymmetric_over_symmetric_frobenius": max_asymmetry
        <= float(gate["antisymmetric_over_symmetric_frobenius_max"]),
    }
    summary = {
        "hessian_relative_frobenius": hessian_distribution,
        "source_hessian_relative_frobenius": source_distribution,
        "energy_abs_error_hartree": energy_distribution,
        "source_energy_abs_error_hartree": source_energy_distribution,
        "force_mae_hartree_per_bohr": force_distribution,
        "source_force_mae_hartree_per_bohr": source_force_distribution,
        "energy_median_ratio_to_source": energy_ratio,
        "force_median_ratio_to_source": force_ratio,
        "max_antisymmetric_over_symmetric_frobenius": max_asymmetry,
        "gate": {**checks, "passed": all(checks.values())},
        "selection_score": (
            hessian_distribution["median"]
            + hessian_distribution["max"]
            + max(
                0.0,
                energy_ratio - float(gate["energy_median_ratio_to_source_max"]),
            )
            + max(
                0.0,
                force_ratio - float(gate["force_median_ratio_to_source_max"]),
            )
        ),
    }
    return summary, rows, arrays


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _periodic_evaluation_metrics(final: dict[str, Any]) -> dict[str, float | bool]:
    """Flatten gate-relevant full-dataset metrics for live JSONL monitoring."""
    return {
        "median_relative_frobenius": final["hessian_relative_frobenius"]["median"],
        "p90_relative_frobenius": final["hessian_relative_frobenius"]["p90"],
        "max_relative_frobenius": final["hessian_relative_frobenius"]["max"],
        "median_energy_abs_error_hartree": final["energy_abs_error_hartree"]["median"],
        "median_force_mae_hartree_per_bohr": final["force_mae_hartree_per_bohr"]["median"],
        "energy_median_ratio_to_source": final["energy_median_ratio_to_source"],
        "force_median_ratio_to_source": final["force_median_ratio_to_source"],
        "max_antisymmetric_over_symmetric_frobenius": final[
            "max_antisymmetric_over_symmetric_frobenius"
        ],
        "selection_score": final["selection_score"],
        "capacity_gate_passed": final["gate"]["passed"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    protocol, arm = _load_protocol(args.protocol, args.arm_id, args.run_mode)
    training = protocol["training"]
    steps = int(training[f"{args.run_mode}_steps"])
    log_interval = int(training[f"log_interval_{args.run_mode}"])
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(arm["seed"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parents, provenance = _load_selected_parents(protocol, device)
    model, optimizer, architecture = _build_model_and_optimizer(arm, device)
    generator = torch.Generator(device="cpu").manual_seed(int(arm["seed"]) + 104729)
    best_score = math.inf
    best_step = -1
    metrics_path = args.output_dir / "training_metrics.jsonl"
    with metrics_path.open("w") as handle:
        for step in range(steps + 1):
            sampled_terms = {
                "energy_loss": math.nan,
                "force_loss": math.nan,
                "hessian_loss": math.nan,
                "parameter_loss": math.nan,
            }
            gradient_norm = math.nan
            balance: dict[str, Any] = {}
            gradnorm: dict[str, float] = {}
            if step > 0:
                optimizer.zero_grad(set_to_none=True)
                batch_size = int(training.get("parent_batch_size", 1))
                if batch_size <= 0:
                    raise ValueError("parent_batch_size must be positive")
                if batch_size == 1:
                    parent_indices = [
                        int(torch.randint(len(parents), (1,), generator=generator))
                    ]
                else:
                    parent_indices = [
                        int(index)
                        for index in torch.randperm(len(parents), generator=generator)[
                            : min(batch_size, len(parents))
                        ]
                    ]
                losses = [
                    _parent_loss(model, parents[parent_index], training)
                    for parent_index in parent_indices
                ]
                terms = {
                    key: torch.mean(torch.stack([row[1][key] for row in losses]))
                    for key in losses[0][1]
                }
                warmup_steps = int(training.get("hessian_warmup_steps", 0))
                warmup = min(1.0, step / warmup_steps) if warmup_steps else 1.0
                energy_task = (
                    float(training["lambda_energy"]) * terms["energy_loss"]
                    + float(training["lambda_parameter"]) * terms["parameter_loss"]
                )
                force_task = float(training["lambda_force"]) * terms["force_loss"]
                energy_force_task = energy_task + force_task
                hessian_task = (
                    warmup
                    * float(training["lambda_hessian"])
                    * terms["hessian_loss"]
                )
                if bool(training.get("gradnorm_balance", False)):
                    trainable = [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ]
                    energy_force_gradient_norm = _task_gradient_norm(
                        energy_force_task, trainable
                    )
                    raw_hessian_gradient_norm = _task_gradient_norm(
                        hessian_task, trainable
                    )
                    requested_scale = (
                        float(training.get("gradnorm_target_ratio", 1.0))
                        * energy_force_gradient_norm
                        / raw_hessian_gradient_norm
                    )
                    gradnorm_scale = torch.clamp(
                        requested_scale,
                        min=float(training.get("gradnorm_min_scale", 1e-3)),
                        max=float(training.get("gradnorm_max_scale", 1e4)),
                    ).detach()
                    hessian_task = gradnorm_scale * hessian_task
                    gradnorm = {
                        "gradnorm/energy_force_gradient_norm": float(
                            energy_force_gradient_norm.detach().cpu()
                        ),
                        "gradnorm/raw_hessian_gradient_norm": float(
                            raw_hessian_gradient_norm.detach().cpu()
                        ),
                        "gradnorm/hessian_scale": float(gradnorm_scale.cpu()),
                    }
                task_partition = str(
                    training.get("pcgrad_task_partition", "energy_force_hessian")
                )
                if task_partition not in (
                    "energy_force_hessian",
                    "energy_force_hessian_separate",
                ):
                    raise ValueError(f"unsupported PCGrad task partition: {task_partition}")
                total = energy_force_task + hessian_task
                if task_partition == "energy_force_hessian_separate":
                    if not bool(training.get("pcgrad", False)):
                        raise ValueError("separate E/F/H task partition requires PCGrad")
                    trainable = [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ]
                    separate_tasks = {
                        "energy": energy_task,
                        "force": force_task,
                        "hessian": hessian_task,
                    }
                    if bool(training.get("gradnorm_balance_separate_tasks", False)):
                        raw_norms = {
                            name: _task_gradient_norm(loss, trainable)
                            for name, loss in separate_tasks.items()
                        }
                        reference_name = str(
                            training.get("gradnorm_reference_task", "force")
                        )
                        if reference_name not in raw_norms:
                            raise ValueError(
                                f"unknown GradNorm reference task: {reference_name}"
                            )
                        target_norm = raw_norms[reference_name].detach()
                        balanced_tasks = {}
                        for name, loss in separate_tasks.items():
                            target_multiplier = _separate_gradnorm_target_multiplier(
                                name,
                                hessian_warmup=warmup,
                                preserve_hessian_warmup=bool(
                                    training.get(
                                        "gradnorm_preserve_hessian_warmup", False
                                    )
                                ),
                            )
                            requested = (
                                target_multiplier * target_norm / raw_norms[name]
                            )
                            scale = torch.clamp(
                                requested,
                                min=float(training.get("gradnorm_min_scale", 1e-3)),
                                max=float(training.get("gradnorm_max_scale", 1e4)),
                            ).detach()
                            balanced_tasks[name] = scale * loss
                            gradnorm[f"gradnorm/{name}_raw_gradient_norm"] = float(
                                raw_norms[name].detach().cpu()
                            )
                            gradnorm[f"gradnorm/{name}_scale"] = float(scale.cpu())
                            gradnorm[f"gradnorm/{name}_target_multiplier"] = float(
                                target_multiplier
                            )
                        separate_tasks = balanced_tasks
                    balance = assign_multi_task_pcgrad(separate_tasks, trainable)
                elif bool(training.get("pcgrad", False)):
                    balance = assign_two_task_pcgrad(
                        energy_force_task,
                        hessian_task,
                        [
                            parameter
                            for parameter in model.parameters()
                            if parameter.requires_grad
                        ],
                        max_second_to_first_norm_ratio=float(
                            training.get(
                                "pcgrad_max_hessian_to_energy_force_norm_ratio",
                                math.inf,
                            )
                        ),
                    )
                else:
                    total.backward()
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        [
                            parameter
                            for parameter in model.parameters()
                            if parameter.requires_grad
                        ],
                        float(training["gradient_clip"]),
                    )
                    .detach()
                    .cpu()
                )
                optimizer.step()
                sampled_terms = {
                    key: float(value.detach().cpu()) for key, value in terms.items()
                }
            if step % log_interval == 0 or step == steps:
                final, rows, arrays = _evaluate(
                    model,
                    parents,
                    protocol["capacity_gate"],
                    anchor_energy_force=bool(
                        training.get("anchor_energy_force", True)
                    ),
                )
                elapsed = time.perf_counter() - started
                row = {
                    "step": step,
                    "wall_time_s": elapsed,
                    "gradient_norm": gradient_norm,
                    **{f"sampled_{key}": value for key, value in sampled_terms.items()},
                    **_periodic_evaluation_metrics(final),
                    **balance,
                    **gradnorm,
                    "gpu_peak_memory_mb": (
                        torch.cuda.max_memory_allocated(device) / 1024**2
                        if device.type == "cuda"
                        else 0.0
                    ),
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                print(json.dumps(row, sort_keys=True), flush=True)
                checkpoint = {
                    "step": step,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "protocol": args.protocol.resolve().as_posix(),
                    "protocol_sha256": _sha256(args.protocol),
                    "arm_id": args.arm_id,
                    "run_mode": args.run_mode,
                    "test100_accessed": False,
                    **provenance,
                }
                torch.save(checkpoint, args.output_dir / "last.ckpt")
                if float(final["selection_score"]) < best_score:
                    best_score = float(final["selection_score"])
                    best_step = step
                    torch.save(checkpoint, args.output_dir / "best.ckpt")
                    _write_rows(args.output_dir / "per_parent_metrics.csv", rows)
                    for molecule_id, payload in arrays.items():
                        np.savez_compressed(
                            args.output_dir / f"{molecule_id}_result.npz", **payload
                        )

    best = torch.load(args.output_dir / "best.ckpt", map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"])
    final, rows, arrays = _evaluate(
        model,
        parents,
        protocol["capacity_gate"],
        anchor_energy_force=bool(training.get("anchor_energy_force", True)),
    )
    _write_rows(args.output_dir / "per_parent_metrics.csv", rows)
    for molecule_id, payload in arrays.items():
        np.savez_compressed(args.output_dir / f"{molecule_id}_result.npz", **payload)
    result = {
        "definition": protocol["definitions"],
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "arm_id": args.arm_id,
        "run_mode": args.run_mode,
        "arm": arm,
        "architecture": architecture,
        "steps": steps,
        "best_step": best_step,
        "best_selection_score": best_score,
        "final": final,
        "per_parent": rows,
        "wall_time_s": time.perf_counter() - started,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "gpu_peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        ),
        "parent_cv_authorized": bool(final["gate"]["passed"] and args.run_mode == "formal"),
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        **provenance,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--run-mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
