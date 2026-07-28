#!/usr/bin/env bash
#SBATCH --job-name=ctH_lin
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --nodelist=node01

set -euo pipefail

source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

: "${SOURCE_RUN_DIR:?set SOURCE_RUN_DIR}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

BASELINE_MANIFEST=${BASELINE_MANIFEST:-/scratch/xzh/models/complete_total_capacity/20260717/stage2_direction_v1/selection_uniform_A_v1/selected_baseline_manifest.json}
DIRECTION_MANIFEST=${DIRECTION_MANIFEST:-/scratch/xzh/models/complete_total_capacity/20260717/stage2_direction_v3_near_full/candidate_direction_manifest.json}
DESIGN_ROOT=${DESIGN_ROOT:-/scratch/xzh/models/complete_total_capacity/20260717/stage2_direction_v1/shared_descriptor_design_v2}
RIDGE=${RIDGE:-1e-6}
CG_TOLERANCE=${CG_TOLERANCE:-1e-8}
CG_MAX_ITERATIONS=${CG_MAX_ITERATIONS:-1000}
RIDGE_GRID=${RIDGE_GRID:-}

mkdir -p "${OUTPUT_DIR}"

ridge_args=()
if [[ -n "${RIDGE_GRID}" ]]; then
  read -r -a ridge_values <<< "${RIDGE_GRID}"
  ridge_args+=(--ridge-grid "${ridge_values[@]}")
fi

/usr/bin/time -v -o "${OUTPUT_DIR}/resource.time" \
  python scripts/qm9_complete_total_anchored_linear_direction_capacity.py \
    --baseline-manifest "${BASELINE_MANIFEST}" \
    --source-run-dir "${SOURCE_RUN_DIR}" \
    --direction-manifest "${DIRECTION_MANIFEST}" \
    --design-root "${DESIGN_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda:0 \
    --ridge "${RIDGE}" \
    --cg-tolerance "${CG_TOLERANCE}" \
    --cg-max-iterations "${CG_MAX_ITERATIONS}" \
    "${ridge_args[@]}"
