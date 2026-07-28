#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
if [[ -f "${ENV_FILE:-/scratch/xzh/env.sh}" ]]; then
  # shellcheck source=/dev/null
  source "${ENV_FILE:-/scratch/xzh/env.sh}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
export DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"
export DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

EG_RUN_DIR="${EG_RUN_DIR:-${DFT_MODELS}/train/runs/qm9_random1000_eg_e10_20260713_162829}"
EG_CKPT="${EG_CKPT:-${EG_RUN_DIR}/checkpoints/epoch_009.ckpt}"
EGF_RUN_DIR="${EGF_RUN_DIR:-${DFT_MODELS}/train/runs/qm9_random1000_egf_lam1_e10_20260713_162829}"
EGF_CKPT="${EGF_CKPT:-${EGF_RUN_DIR}/checkpoints/epoch_009.ckpt}"
REFERENCE_DIR="${REFERENCE_DIR:-${DFT_MODELS}/eval/qm9_random1000_pbe_hessian_refs/test100_pbe_hessians}"
MOLECULES="${MOLECULES:-0000777,0043905,0056566,0072895,0054659,0093887,0040728,0060531}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${DFT_MODELS}/eval/qm9_random1000_fixed_hessian_precision_scan/${RUN_STAMP}}"
mkdir -p "${OUT_DIR}/logs"

conditions=(
  "float64_h3em3:float64:3e-3"
  "float64_h1em3:float64:1e-3"
  "float64_h3em4:float64:3e-4"
  "float32_h3em3:float32:3e-3"
  "float32_h1em3:float32:1e-3"
  "float32_h3em4:float32:3e-4"
)

pids=()
for gpu in "${!conditions[@]}"; do
  IFS=: read -r name dtype displacement <<<"${conditions[$gpu]}"
  condition_dir="${OUT_DIR}/${name}"
  mkdir -p "${condition_dir}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    /usr/bin/time -v -o "${OUT_DIR}/logs/${name}.time.txt" \
      "${PYTHON_BIN}" scripts/qm9_second_order_autograd_hessian_audit.py \
        --run "EG=${EG_RUN_DIR}=${EG_CKPT}" \
        --run "EGF_lam1=${EGF_RUN_DIR}=${EGF_CKPT}" \
        --molecules "${MOLECULES}" \
        --scf-iteration 1 \
        --num-workers 0 \
        --device cuda:0 \
        --model-dtype "${dtype}" \
        --output-json "${condition_dir}/summary.json" \
        --output-dir "${condition_dir}" \
        --reference-dir "${REFERENCE_DIR}" \
        --fd-displacement "${displacement}" \
        --hvp-eps "${displacement}" \
        --run-hvp \
        --no-run-self-edge-diagnostic \
        --no-run-unrolled
  ) >"${OUT_DIR}/logs/${name}.log" 2>&1 &
  pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failures=$((failures + 1))
  fi
done
printf '%s\n' \
  "date=$(date --iso-8601=seconds)" \
  "failed_processes=${failures}" | tee "${OUT_DIR}/scan_status.txt"
if [[ "${failures}" != "0" ]]; then
  exit 1
fi
echo "Fixed-density precision scan complete: ${OUT_DIR}"
