#!/usr/bin/env bash
#SBATCH --job-name=ctH_shared5
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/shared_capacity_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/shared_capacity_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

ROOT=/scratch/xzh/models/complete_total_capacity/20260717
BASELINE_MANIFEST=${SHARED_CAPACITY_BASELINE_MANIFEST:-${ROOT}/five_parent_baseline_manifest.json}
OUTPUT_DIR=${SHARED_CAPACITY_OUTPUT_DIR:-${ROOT}/five_parent_shared_geometry_capacity}
mkdir -p "${OUTPUT_DIR}" /scratch/xzh/logs/complete_total_capacity

EXTRA_ARGS=()
if [[ -n "${SHARED_CAPACITY_DESIGN_ONLY:-}" ]]; then
  EXTRA_ARGS+=(--design-only)
fi

/usr/bin/time -v -o "${OUTPUT_DIR}/resource.time" \
python scripts/qm9_complete_total_geometry_shared_capacity.py \
  --baseline-manifest "${BASELINE_MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --derivative-step 1e-3 \
  --relative-loss-fraction 0.5 \
  --lsqr-tolerance "${SHARED_CAPACITY_LSQR_TOLERANCE:-1e-9}" \
  --lsqr-max-iterations "${SHARED_CAPACITY_LSQR_MAX_ITERATIONS:-1000}" \
  "${EXTRA_ARGS[@]}"
