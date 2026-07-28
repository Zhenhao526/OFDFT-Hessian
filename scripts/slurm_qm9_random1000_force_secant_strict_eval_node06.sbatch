#!/usr/bin/env bash
#SBATCH --job-name=qm9_fsec_strict
#SBATCH --partition=compute
#SBATCH --nodelist=node06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=700G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/xzh/logs/qm9_random1000_force_secant_strict/%j.out
#SBATCH --error=/scratch/xzh/logs/qm9_random1000_force_secant_strict/%j.err

set -euo pipefail

source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
mkdir -p /scratch/xzh/logs/qm9_random1000_force_secant_strict

export DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
export DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

CANDIDATE_RUN="${CANDIDATE_RUN:-${DFT_MODELS}/train/runs/qm9_random1000_forcew1p0_secantw0p01_e10_20260715_210000}"
CANDIDATE_CKPT="${CANDIDATE_CKPT:-${CANDIDATE_RUN}/checkpoints/last.ckpt}"
RUN_LABEL="${RUN_LABEL:-ForceSecant}"
DATASET_DIR="${DFT_DATA}/QM9PBEForceRandom1000"
REFERENCE_DIR="${DFT_MODELS}/eval/qm9_random1000_pbe_hessian_refs/test100_pbe_hessians"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${DFT_MODELS}/eval/qm9_random1000_force_secant_strict/${RUN_STAMP}}"
mkdir -p "${OUT_DIR}/hvp" "${OUT_DIR}/full_hessian_0000777" "${OUT_DIR}/logs"

for path in "${CANDIDATE_CKPT}" "${DATASET_DIR}/split.pkl" "${REFERENCE_DIR}"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: missing required input: ${path}" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES=0
/usr/bin/time -v -o "${OUT_DIR}/logs/hvp.time.txt" \
  python scripts/qm9_total_ofdft_hvp_audit.py \
    --dataset-dir "${DATASET_DIR}" --reference-dir "${REFERENCE_DIR}" \
    --run "${RUN_LABEL}=${CANDIDATE_RUN}=${CANDIDATE_CKPT}" \
    --molecules 0000777,0040728,0003027 --sample-id 0 --direction-coordinate 0 \
    --hvp-step 1e-5 \
    --curvature-step 3e-3 --curvature-step 1e-3 \
    --curvature-step 3e-4 --curvature-step 1e-4 --curvature-step 3e-5 \
    --mixed-derivative-step 1e-4 \
    --integral-derivative-step 1e-4 --integral-derivative-workers 4 \
    --model-geometry-derivative autograd \
    --response-solver auto --krylov-tolerance 1e-8 \
    --max-krylov-iterations 1200 --preconditioner-probes 16 \
    --dense-fallback-max-coefficients 2048 \
    --optimizer adam --lr 1e-3 --max-cycle 1000 --convergence-tolerance 1e-2 \
    --fallback-optimizer adam --fallback-lr 3e-4 \
    --fallback-max-cycle 10000 --fallback-convergence-tolerance 1e-5 \
    --fallback-always --lbfgs-refine --newton-refine \
    --device cuda:0 --transform-device cpu --output-dir "${OUT_DIR}/hvp" \
    >"${OUT_DIR}/logs/hvp.log" 2>&1

if [[ "${STRICT_HVP_ONLY:-0}" == "1" ]]; then
  echo "strict_hvp_only_dir=${OUT_DIR}"
  exit 0
fi

export CUDA_VISIBLE_DEVICES=1
/usr/bin/time -v -o "${OUT_DIR}/logs/full_hessian.time.txt" \
  python scripts/qm9_total_ofdft_hessian_audit.py \
    --dataset-dir "${DATASET_DIR}" --reference-dir "${REFERENCE_DIR}" \
    --run "${RUN_LABEL}=${CANDIDATE_RUN}=${CANDIDATE_CKPT}" \
    --molecules 0000777 --sample-id 0 \
    --output-dir "${OUT_DIR}/full_hessian_0000777" --displacement 1e-5 \
    --integral-derivative-step 1e-4 --integral-derivative-workers 4 \
    --model-geometry-derivative autograd \
    --optimizer adam --lr 1e-3 --max-cycle 1000 --convergence-tolerance 1e-2 \
    --fallback-optimizer adam --fallback-lr 3e-4 \
    --fallback-max-cycle 10000 --fallback-convergence-tolerance 1e-5 \
    --fallback-always --lbfgs-refine --newton-refine \
    --device cuda:0 --transform-device cpu \
    >"${OUT_DIR}/logs/full_hessian.log" 2>&1

python - "${OUT_DIR}" "${CANDIDATE_CKPT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
hvp_rows = list(csv.DictReader((out / "hvp" / "metrics.csv").open()))
full = json.loads((out / "full_hessian_0000777" / "summary.json").read_text())
result = {
    "definition": "Strict complete scalar-derived total-OFDFT candidate HVP and full Hessian.",
    "checkpoint": sys.argv[2],
    "hvp_rows": hvp_rows,
    "full_hessian_metric_rows": full["metric_rows"],
}
(out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY

echo "force_secant_strict_dir=${OUT_DIR}"
