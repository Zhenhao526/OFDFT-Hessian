#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
source "${ENV_FILE:-/scratch/xzh/env.sh}"

export DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
export DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

DATASET_DIR="${DFT_DATA}/QM9PBEForceRandom1000"
REFERENCE_DIR="${DFT_MODELS}/eval/qm9_random1000_pbe_hessian_refs/test100_pbe_hessians"
BASELINE_RUN="${BASELINE_RUN:-${DFT_MODELS}/train/runs/qm9_random1000_egf_forcew1p0_e10_20260714_202203}"
BASELINE_CKPT="${BASELINE_CKPT:-${BASELINE_RUN}/checkpoints/epoch_009.ckpt}"
CANDIDATE_RUN="${CANDIDATE_RUN:?CANDIDATE_RUN is required}"
CANDIDATE_CKPT="${CANDIDATE_CKPT:?CANDIDATE_CKPT is required}"
MOLECULE="${MOLECULE:-0000777}"
DISPLACEMENT="${DISPLACEMENT:-3e-5}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
mkdir -p "${OUT_DIR}/logs"

run_hessian() {
  local gpu="$1" name="$2" run_dir="$3" checkpoint="$4"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    /usr/bin/time -v -o "${OUT_DIR}/logs/${name}.time.txt" \
      python scripts/qm9_total_ofdft_hessian_audit.py \
        --dataset-dir "${DATASET_DIR}" --reference-dir "${REFERENCE_DIR}" \
        --run "${name}=${run_dir}=${checkpoint}" \
        --molecules "${MOLECULE}" --sample-id 0 \
        --output-dir "${OUT_DIR}/${name}" --displacement "${DISPLACEMENT}" \
        --integral-derivative-step 1e-4 --integral-derivative-workers 4 \
        --model-geometry-derivative autograd \
        --optimizer adam --lr 1e-3 --max-cycle 1000 --convergence-tolerance 1e-2 \
        --fallback-optimizer adam --fallback-lr 3e-4 \
        --fallback-max-cycle 10000 --fallback-convergence-tolerance 1e-5 \
        --fallback-always --lbfgs-refine --newton-refine \
        --device cuda:0 --transform-device cpu \
        >"${OUT_DIR}/logs/${name}.log" 2>&1
  ) &
  pids+=("$!")
}

pids=()
run_hessian 0 Baseline "${BASELINE_RUN}" "${BASELINE_CKPT}"
run_hessian 1 Candidate "${CANDIDATE_RUN}" "${CANDIDATE_CKPT}"
for pid in "${pids[@]}"; do
  wait "${pid}"
done

python - "${OUT_DIR}" "${DISPLACEMENT}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
payload = {
    "definition": "Strict complete scalar-derived total-OFDFT full-Hessian step check.",
    "displacement_bohr": float(sys.argv[2]),
    "metric_rows": [],
}
for run in ("Baseline", "Candidate"):
    result = json.loads((out / run / "summary.json").read_text())
    payload["metric_rows"].extend(result["metric_rows"])
(out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

echo "total_full_hessian_step_dir=${OUT_DIR}"
