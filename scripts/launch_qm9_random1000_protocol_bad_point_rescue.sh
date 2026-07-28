#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
if [[ -f "${ENV_FILE:-/scratch/xzh/env.sh}" ]]; then
  # shellcheck source=/dev/null
  source "${ENV_FILE:-/scratch/xzh/env.sh}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
export DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
export DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

SCAN_ROOT="${SCAN_ROOT:-${DFT_MODELS}/eval/qm9_random1000_hessian_protocol_scan/20260714_194632}"
INPUT_CSV="${INPUT_CSV:-${SCAN_ROOT}/analysis/per_displacement_optimization.csv}"
DATASET_DIR="${DFT_DATA}/QM9PBEForceRandom1000"
EG_RUN_DIR="${EG_RUN_DIR:-${DFT_MODELS}/train/runs/qm9_random1000_eg_e10_20260713_162829}"
EG_CKPT="${EG_CKPT:-${EG_RUN_DIR}/checkpoints/epoch_009.ckpt}"
EGF_RUN_DIR="${EGF_RUN_DIR:-${DFT_MODELS}/train/runs/qm9_random1000_egf_lam1_e10_20260713_162829}"
EGF_CKPT="${EGF_CKPT:-${EGF_RUN_DIR}/checkpoints/epoch_009.ckpt}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${DFT_MODELS}/eval/qm9_random1000_hessian_protocol_bad_point_rescue/${RUN_STAMP}}"
mkdir -p "${OUT_DIR}"

for path in "${INPUT_CSV}" "${DATASET_DIR}" "${EG_CKPT}" "${EGF_CKPT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: missing required path: ${path}" >&2
    exit 1
  fi
done

printf '%s\n' \
  "host=$(hostname)" \
  "date=$(date --iso-8601=seconds)" \
  "input_csv=${INPUT_CSV}" \
  "source_displacement=1e-3" \
  "source_tolerance=1e-5" \
  "target_threshold=1e-5" \
  "out_dir=${OUT_DIR}" | tee "${OUT_DIR}/run_config.txt"

/usr/bin/time -v -o "${OUT_DIR}/time.txt" \
  "${PYTHON_BIN}" scripts/qm9_density_relaxed_bad_displacement_rescue.py \
    --previous-optimization-csv "${INPUT_CSV}" \
    --dataset-dir "${DATASET_DIR}" \
    --run "EG=${EG_RUN_DIR}=${EG_CKPT}" \
    --run "EGF_lam1=${EGF_RUN_DIR}=${EGF_CKPT}" \
    --variant "label_adam3e4:label:adam:3e-4:20000" \
    --variant "sad_adam3e4_slsqp:sad_default:adam:3e-4:10000:slsqp:1.0:2000" \
    --variant "label_adam3e4_slsqp:label:adam:3e-4:10000:slsqp:1.0:2000" \
    --source-displacement 1e-3 \
    --source-tolerance 1e-5 \
    --threshold 1e-5 \
    --output-json "${OUT_DIR}/rescue_results.json" \
    --output-csv "${OUT_DIR}/rescue_points.csv" \
    --summary-csv "${OUT_DIR}/rescue_summary.csv" \
    --curves-csv "${OUT_DIR}/rescue_curves.csv" \
    --device cuda:0 \
    >"${OUT_DIR}/run.log" 2>&1

echo "Bad-point rescue complete: ${OUT_DIR}"
