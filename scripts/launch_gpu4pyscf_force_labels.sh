#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f "${ENV_FILE:-/scratch/xzh/env.sh}" ]]; then
  # shellcheck source=/dev/null
  source "${ENV_FILE:-/scratch/xzh/env.sh}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
DATASET_NAME="${DATASET_NAME:-QM9PBEForceRandom1000}"
LABEL_DIR="${LABEL_DIR:-${DFT_DATA}/${DATASET_NAME}/labels}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${DFT_MODELS:-/scratch/xzh/models}/force_labels/gpu4pyscf_${DATASET_NAME}_${RUN_STAMP}}"

NUM_PROCESSES="${NUM_PROCESSES:-8}"
GPU_DEVICE_IDS="${GPU_DEVICE_IDS:-0,1,2,3,4,5,6,7}"
GEOMETRY_UNIT="${GEOMETRY_UNIT:-auto}"
CHK_DERIVATIVES_MODE="${CHK_DERIVATIVES_MODE:-off}"
KOHN_SHAM_DIR="${KOHN_SHAM_DIR:-}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

export OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS
mkdir -p "${RUN_ROOT}"

echo "[gpu4pyscf-force-labels] LABEL_DIR=${LABEL_DIR}"
echo "[gpu4pyscf-force-labels] RUN_ROOT=${RUN_ROOT}"
echo "[gpu4pyscf-force-labels] NUM_PROCESSES=${NUM_PROCESSES}"
echo "[gpu4pyscf-force-labels] GPU_DEVICE_IDS=${GPU_DEVICE_IDS}"
echo "[gpu4pyscf-force-labels] GEOMETRY_UNIT=${GEOMETRY_UNIT}"
echo "[gpu4pyscf-force-labels] CHK_DERIVATIVES_MODE=${CHK_DERIVATIVES_MODE}"
if [[ -n "${KOHN_SHAM_DIR}" ]]; then
  echo "[gpu4pyscf-force-labels] KOHN_SHAM_DIR=${KOHN_SHAM_DIR}"
fi

cmd=(
  "${PYTHON_BIN}" scripts/append_gpu4pyscf_force_labels.py "${LABEL_DIR}"
  --backend gpu4pyscf
  --num-processes "${NUM_PROCESSES}"
  --gpu-device-ids "${GPU_DEVICE_IDS}"
  --geometry-unit "${GEOMETRY_UNIT}"
  --chk-derivatives-mode "${CHK_DERIVATIVES_MODE}"
  --summary-json "${RUN_ROOT}/summary.json"
  --records-jsonl "${RUN_ROOT}/records.jsonl"
)

if [[ -n "${KOHN_SHAM_DIR}" ]]; then
  cmd+=(--kohn-sham-dir "${KOHN_SHAM_DIR}")
fi

"${cmd[@]}" \
  "$@"
