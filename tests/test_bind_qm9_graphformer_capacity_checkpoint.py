import argparse
import hashlib
import json

import torch

from scripts.bind_qm9_graphformer_capacity_checkpoint import _load_checkpoint, bind


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bind_capacity_checkpoint_adds_strict_original_provenance(tmp_path):
    root_hash = "a" * 64
    checkpoint = tmp_path / "step.ckpt"
    torch.save(
        {
            "state_dict": {"weight": torch.ones(1)},
            "complete_total_capacity": {
                "step": 20,
                "optimizer_state_dict": {"state": {}},
            },
        },
        checkpoint,
    )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "final_step": 20,
                "source_checkpoint_sha256": root_hash,
                "protocol_id": "strict-v1",
                "direction_manifest_sha256": "b" * 64,
                "direction_role": "all",
                "molecules": ["0028399"],
                "validation_accessed": False,
                "test100_accessed": False,
                "proxy_hvp_fallback_allowed": False,
                "complete_total_relaxed_hvp_graph_required": True,
                "strict_active_density_refresh": True,
                "implicit_density_parameter_response": True,
                "loss_weights": {
                    "lambda_E": 1.0,
                    "lambda_F": 1.0,
                    "lambda_rho": 0.1,
                    "lambda_H": 100.0,
                    "lambda_Q": 0.0,
                    "lambda_spec": 0.0,
                },
                "initial_full_hessian_metrics": [
                    {"molecule_id": "0028399", "relative_frobenius": 2.7}
                ],
            }
        )
    )
    output = tmp_path / "bound.ckpt"
    result = bind(
        argparse.Namespace(
            checkpoint=checkpoint,
            checkpoint_sha256=_hash(checkpoint),
            summary=summary,
            summary_sha256=_hash(summary),
            root_source_checkpoint_sha256=root_hash,
            output=output,
        )
    )

    payload = torch.load(output, map_location="cpu", weights_only=False)
    state = payload["complete_total_capacity"]
    assert state["root_source_checkpoint_sha256"] == root_hash
    assert state["strict_active_density_refresh"] is True
    assert state["root_initial_full_hessian_metrics"][0]["relative_frobenius"] == 2.7
    assert state["loss_weights"]["lambda_H"] == 100.0
    assert result["output_checkpoint_sha256"] == _hash(output)


def test_load_checkpoint_retries_pytorch_nested_load_thread_local_error(
    monkeypatch, tmp_path
):
    path = tmp_path / "checkpoint.ckpt"
    calls = []

    def fake_load(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            raise AttributeError(
                "'_thread._local' object has no attribute 'map_location'"
            )
        return {"state_dict": {}}

    monkeypatch.setattr(torch, "load", fake_load)
    assert _load_checkpoint(path) == {"state_dict": {}}
    assert len(calls) == 2
