import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.qm9_graphformer_analytic_relaxed_hvp_audit import _common_args
from scripts.qm9_graphformer_full39_capacity_only import (
    _align_direction_tensors,
    _load_density_rescue,
    _parameter_scope_names,
    capacity_passed,
    configure_capacity_numerics,
    damped_cgls,
    implementation_provenance,
)


def _graphformer_like_names() -> list[tuple[str, torch.Tensor]]:
    names = [
        "energy_mlp.0.weight",
        "energy_mlp.0.bias",
        "energy_mlp.3.weight",
    ]
    for block in range(4):
        names.extend(
            [
                f"gnn_module.g3d_layers.{block}.self_attn.weight",
                f"gnn_module.g3d_layers.{block}.ffn.weight",
            ]
        )
    names.append("gbf.weight")
    return [(name, torch.zeros(1)) for name in names]


def test_parameter_scopes_are_nested_and_include_readout():
    named = _graphformer_like_names()
    readout = set(_parameter_scope_names(named, "energy_readout"))
    last1 = set(_parameter_scope_names(named, "readout_last1"))
    last2 = set(_parameter_scope_names(named, "readout_last2"))
    full = set(_parameter_scope_names(named, "full_graphformer"))

    assert readout < last1 < last2 < full
    assert all(name.startswith("energy_mlp.") for name in readout)
    assert any("g3d_layers.3." in name for name in last1)
    assert not any("g3d_layers.2." in name for name in last1)
    assert any("g3d_layers.2." in name for name in last2)
    assert not any("g3d_layers.1." in name for name in last2)


def test_damped_cgls_matches_small_dense_least_squares():
    matrix = torch.tensor(
        [[2.0, -1.0], [0.5, 3.0], [-1.0, 0.25]], dtype=torch.float64
    )
    residual = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    damping = 0.2
    step, diagnostics = damped_cgls(
        residual,
        lambda vector: matrix @ vector,
        lambda vector: matrix.T @ vector,
        parameter_count=2,
        damping=damping,
        max_iterations=10,
        relative_tolerance=1.0e-12,
    )
    expected = torch.linalg.solve(
        matrix.T @ matrix + damping * torch.eye(2, dtype=torch.float64),
        -(matrix.T @ residual),
    )

    torch.testing.assert_close(step, expected, rtol=1.0e-10, atol=1.0e-12)
    assert diagnostics["converged"]
    assert diagnostics["linearized_minimum_claimed"]
    assert diagnostics["linearized_residual_upper_bound"] < np.linalg.norm(
        residual.numpy()
    )


def test_capacity_gate_is_only_full39_relative_frobenius():
    assert capacity_passed(0.05)
    assert capacity_passed(0.049999)
    assert not capacity_passed(0.050001)
    assert not capacity_passed(float("nan"))


def test_direction_tensors_follow_hvp_device_and_dtype():
    basis = torch.eye(2, dtype=torch.float64)
    projector = torch.eye(2, dtype=torch.float64)
    target = torch.ones((2, 1), dtype=torch.float64)
    like = torch.zeros(2, dtype=torch.float32)
    aligned = _align_direction_tensors(
        basis, projector, target, like
    )
    assert all(item.device == like.device for item in aligned)
    assert all(item.dtype == like.dtype for item in aligned)


def test_capacity_numeric_mode_is_registered_float64():
    previous = torch.get_default_dtype()
    try:
        configure_capacity_numerics({"numerics": {"dtype": "float64"}})
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(previous)


def test_capacity_numeric_mode_rejects_non_float64():
    try:
        configure_capacity_numerics({"numerics": {"dtype": "float32"}})
    except ValueError as error:
        assert "requires float64" in str(error)
    else:
        raise AssertionError("float32 capacity protocol was accepted")


def test_capacity_implementation_provenance_hashes_existing_sources():
    provenance = implementation_provenance()
    assert "scripts/qm9_graphformer_full39_capacity_only.py" in provenance
    assert "mldft/ofdft/implicit_response.py" in provenance
    assert all(
        len(item["sha256"]) == 64 and Path(item["path"]).is_file()
        for item in provenance.values()
    )


def test_capacity_v2_registers_extended_strict_newton_budget(tmp_path):
    protocol_path = (
        Path(__file__).resolve().parents[1]
        / "configs/audit/qm9_graphformer_0028399_full39_capacity_only_v2.yaml"
    )
    protocol = yaml.safe_load(protocol_path.read_text())
    args = _common_args(tmp_path, protocol, "cpu")

    assert args.density_strict_threshold == 1.0e-8
    assert args.newton_max_iterations == 20


def test_density_rescue_is_bound_to_capacity_v2():
    root = Path(__file__).resolve().parents[1]
    protocol_path = (
        root
        / "configs/audit/qm9_graphformer_0028399_full39_capacity_only_v2.yaml"
    )
    rescue_path = (
        root
        / "configs/audit/qm9_graphformer_0028399_density_solver_rescue_v1.yaml"
    )
    protocol = yaml.safe_load(protocol_path.read_text())
    rescue, rescue_hash = _load_density_rescue(
        rescue_path, protocol, protocol_path
    )

    assert rescue["decision"]["lbfgs_refine"] is False
    assert rescue["decision"]["newton_refine"] is True
    assert len(rescue_hash) == 64
