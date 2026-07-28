#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source "${ENV_FILE:-/scratch/xzh/env.sh}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

PYTHON_BIN="${PYTHON_BIN:-/scratch/xzh/envs/structures25/bin/python}"
DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${DFT_MODELS}/eval/qm9_random1000_prepared_geometry_cache_smoke/${RUN_STAMP}}"
MANIFEST="${MANIFEST:-${DFT_MODELS}/eval/qm9_random1000_hessian_protocol_scan/20260714_194632/representative_manifest_8mol.json}"
EG_RUN_DIR="${EG_RUN_DIR:-${DFT_MODELS}/train/runs/qm9_random1000_eg_e10_20260713_162829}"
EGF_RUN_DIR="${EGF_RUN_DIR:-${DFT_MODELS}/train/runs/qm9_random1000_egf_lam1_e10_20260713_162829}"

mkdir -p "${OUT_DIR}/hessians" "${OUT_DIR}/traces"

/usr/bin/time -v -o "${OUT_DIR}/time.txt" \
  "${PYTHON_BIN}" scripts/qm9_hessian_density_relaxed_eval.py \
    --manifest-json "${MANIFEST}" \
    --dataset-dir "${DFT_DATA}/QM9PBEForceRandom1000" \
    --run "EG=${EG_RUN_DIR}=${EG_RUN_DIR}/checkpoints/epoch_009.ckpt" \
    --run "EGF_lam1=${EGF_RUN_DIR}=${EGF_RUN_DIR}/checkpoints/epoch_009.ckpt" \
    --output-json "${OUT_DIR}/summary.json" \
    --output-csv "${OUT_DIR}/per_molecule.csv" \
    --optimization-csv "${OUT_DIR}/optimization.csv" \
    --optimization-trace-dir "${OUT_DIR}/traces" \
    --hessian-npz-dir "${OUT_DIR}/hessians" \
    --max-molecules 1 \
    --displacement 1e-3 \
    --optimizer adam \
    --lr 1e-3 \
    --max-cycle 1000 \
    --convergence-tolerance 1e-2 \
    --fallback-optimizer adam \
    --fallback-lr 3e-4 \
    --fallback-max-cycle 10000 \
    --fallback-convergence-tolerance 1e-4 \
    --fallback-always \
    --base-density-warm-start \
    --share-prepared-geometry-across-runs \
    --initialization sad_default \
    --device cuda:0

echo "prepared_geometry_cache_smoke=${OUT_DIR}"
