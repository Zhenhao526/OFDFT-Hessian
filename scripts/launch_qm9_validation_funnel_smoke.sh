#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
source "${ENV_FILE:-/scratch/xzh/env.sh}"

export DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
export DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHON_BIN="${PYTHON_BIN:-/scratch/xzh/envs/structures25/bin/python}"
RUN_DIR="${DFT_MODELS}/train/runs/qm9_random1000_egf_lam1_e10_20260713_162829"
OUT_DIR="${OUT_DIR:-${DFT_MODELS}/eval/qm9_random1000_validation_funnel_smoke/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_DIR}/fixed_hvp"

"${PYTHON_BIN}" scripts/qm9_force_eval.py \
  --run-dir "${RUN_DIR}" \
  --ckpt "${RUN_DIR}/checkpoints/epoch_009.ckpt" \
  --split val --ground-state-only --device cuda:0 --batch-size 4 --num-workers 4 \
  --output-json "${OUT_DIR}/validation_force.json" \
  --worst-csv "${OUT_DIR}/validation_force_worst.csv"

"${PYTHON_BIN}" scripts/qm9_second_order_autograd_hessian_audit.py \
  --run "EGF_w0p1=${RUN_DIR}=${RUN_DIR}/checkpoints/epoch_009.ckpt" \
  --molecules 0000751 --split val --scf-iteration -1 \
  --num-workers 0 --device cuda:0 --model-dtype float64 \
  --output-json "${OUT_DIR}/fixed_hvp/summary.json" \
  --output-dir "${OUT_DIR}/fixed_hvp" \
  --reference-dir "${OUT_DIR}/no_pbe_hessian_reference" \
  --fd-displacement 1e-3 --hvp-eps 1e-3 \
  --directional-sample-ids 1 2 3 --compare-pbe-force-secant \
  --run-hvp --no-run-self-edge-diagnostic --no-run-unrolled

echo "validation_funnel_smoke=${OUT_DIR}"
