import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("MLDFT_RUN_RUNTIME_TESTS") != "1",
    reason="requires local P1-410 runtime data and checkpoints",
)
def test_qm9_fixed_density_full_edge_autograd_hessian_is_finite(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    run_dir = repo / "_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000"
    ckpt = run_dir / "checkpoints/epoch_000.ckpt"
    if not ckpt.exists():
        pytest.skip(f"missing checkpoint: {ckpt}")

    output_json = tmp_path / "second_order_runtime.json"
    output_dir = tmp_path / "second_order_runtime"
    env = os.environ.copy()
    env.setdefault("DFT_DATA", str(repo / "_runtime/qm9_p1"))
    env.setdefault("DFT_MODELS", str(repo / "_runtime/qm9_p1_models"))
    device = env.get("MLDFT_RUNTIME_DEVICE", "cpu")
    cmd = [
        sys.executable,
        str(repo / "scripts/qm9_second_order_autograd_hessian_audit.py"),
        "--molecules",
        "0000010",
        "--scf-iteration",
        "1",
        "--run",
        f"EGF_lam1_s3000={run_dir}={ckpt}",
        "--output-dir",
        str(output_dir),
        "--output-json",
        str(output_json),
        "--device",
        device,
        "--no-run-hvp",
        "--no-run-unrolled",
        "--run-self-edge-diagnostic",
    ]
    subprocess.run(cmd, cwd=repo, env=env, check=True)

    data = json.loads(output_json.read_text())
    fixed_rows = data["fixed_density_full_hessian"]
    assert len(fixed_rows) == 1
    fixed = fixed_rows[0]
    assert fixed["success"]
    assert fixed["autograd_stats"]["finite"]
    assert fixed["autograd_vs_fd"]["finite"]
    assert fixed["autograd_vs_fd"]["relative_fro_error"] < 1e-2

    drop_self = data["drop_self_edges_diagnostic"][0]
    assert drop_self["success"]
    assert drop_self["autograd_stats"]["finite"]
