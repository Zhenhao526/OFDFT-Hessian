#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
source "${ENV_FILE:-/scratch/xzh/env.sh}"

export DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
export DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

DATASET_DIR="${DFT_DATA}/QM9PBEForceRandom1000"
BASELINE_RUN="${BASELINE_RUN:-${DFT_MODELS}/train/runs/qm9_random1000_egf_forcew1p0_e10_20260714_202203}"
BASELINE_CKPT="${BASELINE_CKPT:-${BASELINE_RUN}/checkpoints/epoch_009.ckpt}"
CANDIDATE_RUN="${CANDIDATE_RUN:-${DFT_MODELS}/train/runs/qm9_random1000_energy_secant_w0p1_e10_20260715_171103}"
CANDIDATE_CKPT="${CANDIDATE_CKPT:-${CANDIDATE_RUN}/checkpoints/last.ckpt}"
REFERENCE_DIR="${DFT_MODELS}/eval/qm9_random1000_pbe_hessian_refs/test100_pbe_hessians"
MOLECULES="${MOLECULES:-0000777,0040728,0003027}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${DFT_MODELS}/eval/qm9_random1000_energy_secant_total_hvp/${RUN_STAMP}}"
mkdir -p "${OUT_DIR}/hvp" "${OUT_DIR}/full_hessian_0000777" "${OUT_DIR}/logs"

run_hvp() {
  local gpu="$1" name="$2" run_dir="$3" ckpt="$4"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    /usr/bin/time -v -o "${OUT_DIR}/logs/hvp_${name}.time.txt" \
      python scripts/qm9_total_ofdft_hvp_audit.py \
        --dataset-dir "${DATASET_DIR}" \
        --reference-dir "${REFERENCE_DIR}" \
        --run "${name}=${run_dir}=${ckpt}" \
        --molecules "${MOLECULES}" \
        --sample-id 0 --direction-coordinate 0 \
        --hvp-step 1e-5 \
        --curvature-step 3e-3 --curvature-step 1e-3 \
        --curvature-step 3e-4 --curvature-step 1e-4 --curvature-step 3e-5 \
        --mixed-derivative-step 1e-4 \
        --integral-derivative-step 1e-4 --integral-derivative-workers 4 \
        --model-geometry-derivative autograd \
        --response-solver auto --krylov-tolerance 1e-8 \
        --max-krylov-iterations 1200 --preconditioner-probes 16 \
        --dense-fallback-max-coefficients 2048 \
        --optimizer adam --lr 1e-3 --max-cycle 1000 \
        --convergence-tolerance 1e-2 \
        --fallback-optimizer adam --fallback-lr 3e-4 \
        --fallback-max-cycle 10000 --fallback-convergence-tolerance 1e-5 \
        --fallback-always --lbfgs-refine --newton-refine \
        --device cuda:0 --transform-device cpu \
        --output-dir "${OUT_DIR}/hvp/${name}" \
        >"${OUT_DIR}/logs/hvp_${name}.log" 2>&1
  ) &
  pids+=("$!")
}

pids=()
run_hvp 0 Baseline "${BASELINE_RUN}" "${BASELINE_CKPT}"
sleep "${HVP_STARTUP_STAGGER_SECONDS:-180}"
run_hvp 1 Candidate "${CANDIDATE_RUN}" "${CANDIDATE_CKPT}"
for pid in "${pids[@]}"; do wait "${pid}"; done

if [[ "${HVP_ONLY:-0}" == "1" ]]; then
  echo "total_hvp_only_dir=${OUT_DIR}"
  exit 0
fi

run_full_hessian() {
  local gpu="$1" name="$2" run_dir="$3" ckpt="$4"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    /usr/bin/time -v -o "${OUT_DIR}/logs/full_hessian_${name}.time.txt" \
      python scripts/qm9_total_ofdft_hessian_audit.py \
        --dataset-dir "${DATASET_DIR}" \
        --reference-dir "${REFERENCE_DIR}" \
        --run "${name}=${run_dir}=${ckpt}" \
        --molecules 0000777 --sample-id 0 \
        --output-dir "${OUT_DIR}/full_hessian_0000777/${name}" \
        --displacement 1e-5 \
        --integral-derivative-step 1e-4 --integral-derivative-workers 4 \
        --model-geometry-derivative autograd \
        --optimizer adam --lr 1e-3 --max-cycle 1000 \
        --convergence-tolerance 1e-2 \
        --fallback-optimizer adam --fallback-lr 3e-4 \
        --fallback-max-cycle 10000 --fallback-convergence-tolerance 1e-5 \
        --fallback-always --lbfgs-refine --newton-refine \
        --device cuda:0 --transform-device cpu \
        >"${OUT_DIR}/logs/full_hessian_${name}.log" 2>&1
  ) &
  pids+=("$!")
}

pids=()
run_full_hessian 0 Baseline "${BASELINE_RUN}" "${BASELINE_CKPT}"
run_full_hessian 1 Candidate "${CANDIDATE_RUN}" "${CANDIDATE_CKPT}"
for pid in "${pids[@]}"; do wait "${pid}"; done

python - "${OUT_DIR}" "${BASELINE_CKPT}" "${CANDIDATE_CKPT}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

out = Path(sys.argv[1])
result = {
    "definition": (
        "Complete scalar-derived total-OFDFT forces, strict density relaxation, KKT density "
        "response HVP, and PBE analytic Hessian directional references."
    ),
    "checkpoints": {
        "Baseline": sys.argv[2],
        "CandidateFinal": sys.argv[3],
    },
    "hvp": {},
    "full_hessian_0000777": {},
}
for name in ("Baseline", "Candidate"):
    rows = list(csv.DictReader((out / "hvp" / name / "metrics.csv").open()))
    metrics = {}
    for key in (
        "strict_relaxed_vs_pbe_mae",
        "strict_relaxed_vs_pbe_rmse",
        "strict_relaxed_vs_pbe_relative_frobenius",
        "implicit_vs_pbe_relative_frobenius",
        "implicit_hvp_relative_frobenius",
        "strict_relaxed_max_gradient_norm",
        "wall_time_s",
    ):
        values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
        metrics[f"mean_{key}"] = sum(values) / len(values) if values else None
    result["hvp"][name] = {"molecules": len(rows), **metrics}

    full = json.loads(
        (out / "full_hessian_0000777" / name / "summary.json").read_text()
    )
    result["full_hessian_0000777"][name] = full["metric_rows"]

baseline = result["hvp"]["Baseline"]["mean_strict_relaxed_vs_pbe_relative_frobenius"]
candidate = result["hvp"]["Candidate"]["mean_strict_relaxed_vs_pbe_relative_frobenius"]
result["candidate_hvp_improves"] = bool(
    baseline is not None and candidate is not None and math.isfinite(candidate) and candidate < baseline
)
(out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY

echo "total_hvp_confirm_dir=${OUT_DIR}"
