#!/usr/bin/env bash
set -euo pipefail

root=${QM9_NODE02_ROOT:-/home/shenwei01/xzh_node02_20260724}
repo=${QM9_NODE02_REPO:-${root}/work/structures25}
dataset=${root}/data/QM9PBEForceEGFH10ScratchV1
reference_root=${root}/artifacts/graphformer_hybrid_relaxed_hvp_rebuild_v1/pbe_hessian_train20
reference_dir=${reference_root}/cache
reference_manifest=${reference_root}/manifest.json
article_run_dir=${root}/models/train/runs/qm9_graphformer_egfh10_force_secant_article_warmstart_s100_v1
checkpoint=${QM9_IMPLICIT_HVP_FINAL_CHECKPOINT:?Set the final implicit-HVP checkpoint}
expected_sha=${QM9_IMPLICIT_HVP_FINAL_CHECKPOINT_SHA256:?Set its SHA256}
output=${QM9_IMPLICIT_HVP_FINAL_EVAL_OUTPUT:?Set a fresh evaluation output directory}
gpu=${QM9_IMPLICIT_HVP_GPU:-0}
molecule_id=0016298
run_name=GraphformerImplicitRelaxedHVPFinal

[[ "$(hostname)" == node02 ]] || {
  echo "This evaluation is authorized on node02 only." >&2
  exit 2
}
for path in \
  "${root}" "${repo}" "${dataset}" "${reference_root}" \
  "${article_run_dir}" "${checkpoint}" "${output}"; do
  if [[ "${path}" == /scratch* ]]; then
    echo "The frozen implicit-HVP evaluation must not use /scratch." >&2
    exit 2
  fi
done
[[ ! -e "${output}" ]] || {
  echo "Refusing to reuse output directory: ${output}" >&2
  exit 2
}
[[ -f "${checkpoint}" ]] || {
  echo "Missing final checkpoint: ${checkpoint}" >&2
  exit 2
}
actual_sha=$(sha256sum "${checkpoint}" | awk '{print $1}')
[[ "${actual_sha}" == "${expected_sha}" ]] || {
  echo "Final checkpoint hash mismatch: ${actual_sha} != ${expected_sha}" >&2
  exit 3
}
[[ -f "${reference_manifest}" ]] || {
  echo "Missing frozen PBE Hessian manifest." >&2
  exit 2
}

source "${repo}/scripts/activate_qm9_node02_local.sh"
cd "${repo}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export DFT_DATA="${root}/data"
export DFT_MODELS="${root}/models"
export TMPDIR="${root}/tmp"
export XDG_CACHE_HOME="${root}/cache"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}"

python - "${checkpoint}" "${expected_sha}" <<'PY'
import sys
import torch

path, expected_sha = sys.argv[1:]
payload = torch.load(path, map_location="cpu", weights_only=False)
state = payload.get("complete_total_capacity")
if not isinstance(state, dict):
    raise SystemExit("checkpoint lacks complete_total_capacity provenance")
if state.get("protocol_id") != "qm9_graphformer_egfh_implicit_relaxed_hvp_pilot_v2":
    raise SystemExit("checkpoint protocol mismatch")
if state.get("analytic_relaxed_hvp") is not True:
    raise SystemExit("checkpoint is not the analytic relaxed-HVP branch")
if state.get("implicit_density_parameter_response") is not True:
    raise SystemExit("checkpoint lacks implicit density-parameter response")
if state.get("strict_active_density_refresh") is not True:
    raise SystemExit("checkpoint lacks strict active-density refresh")
if state.get("egf_label_density_replay") is not True:
    raise SystemExit("checkpoint lacks the frozen E/G/F replay branch")
if state.get("validation_accessed") is not False or state.get("test100_accessed") is not False:
    raise SystemExit("checkpoint does not certify frozen validation/Test100")
if int(state.get("step", -1)) != 10:
    raise SystemExit(f"expected capacity step 10, found {state.get('step')}")
print(
    {
        "checkpoint": path,
        "sha256": expected_sha,
        "capacity_step": state["step"],
        "protocol_id": state["protocol_id"],
    }
)
PY

mkdir -p "${output}/hessian" "${output}/logs"
/usr/bin/time -v -o "${output}/logs/hessian.resource.time" \
  python scripts/qm9_total_ofdft_hessian_audit.py \
    --dataset-dir "${dataset}" \
    --reference-dir "${reference_dir}" \
    --run "${run_name}=${article_run_dir}=${checkpoint}" \
    --molecules "${molecule_id}" \
    --sample-id 0 \
    --output-dir "${output}/hessian" \
    --displacement 1e-4 \
    --integral-derivative-step 1e-4 \
    --integral-derivative-workers 4 \
    --model-geometry-derivative autograd \
    --base-initialization label_reference \
    --optimizer adam \
    --lr 1e-3 \
    --max-cycle 1000 \
    --convergence-tolerance 1e-2 \
    --fallback-optimizer adam \
    --fallback-lr 3e-4 \
    --fallback-max-cycle 10000 \
    --fallback-convergence-tolerance 1e-5 \
    --fallback-always \
    --lbfgs-refine \
    --lbfgs-tolerance 5e-9 \
    --lbfgs-max-iterations 500 \
    --newton-refine \
    --newton-tolerance 5e-9 \
    --newton-max-iterations 6 \
    --device cuda:0 \
    --transform-device cpu \
    >"${output}/logs/hessian.log" 2>&1

python scripts/qm9_hessian_vibrational_metrics.py \
  --manifest-json "${reference_manifest}" \
  --dataset-dir "${dataset}" \
  --result-json "${run_name}=${output}/hessian/summary.json" \
  --output-dir "${output}/vibrational" \
  >"${output}/logs/vibrational.log" 2>&1

python - "${output}" "${expected_sha}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
checkpoint_sha = sys.argv[2]
hessian = json.loads((output / "hessian" / "summary.json").read_text())
vibration = json.loads((output / "vibrational" / "summary.json").read_text())
hessian_row = hessian["metric_rows"][0]
vibration_row = vibration["rows"][0]
summary = {
    "definition": (
        "Strict self-consistent total-OFDFT Cartesian force-difference Hessian "
        "and vibrational metrics for the final implicit relaxed-HVP checkpoint."
    ),
    "checkpoint_sha256": checkpoint_sha,
    "molecule_id": "0016298",
    "hessian": hessian_row,
    "vibration": vibration_row,
    "validation_accessed": False,
    "test100_accessed": False,
    "test100_evaluations_used": 0,
}
summary_path = output / "final_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
print("final_summary_sha256=" + hashlib.sha256(summary_path.read_bytes()).hexdigest())
PY
