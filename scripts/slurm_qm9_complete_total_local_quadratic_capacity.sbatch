#!/usr/bin/env bash
#SBATCH --job-name=ctH_quad
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
BOND_SCALE=${BOND_SCALE:-1.25}
INCLUDE_CROSS_TERMS=${INCLUDE_CROSS_TERMS:-1}
CROSS_ALL_PAIRS=${CROSS_ALL_PAIRS:-0}
USE_ENVIRONMENT_TYPES=${USE_ENVIRONMENT_TYPES:-0}
PARENT_SPECIFIC_KEYS=${PARENT_SPECIFIC_KEYS:-0}
SOLVE_PER_PARENT=${SOLVE_PER_PARENT:-0}
RIDGE=${RIDGE:-1}
RIDGE_GRID=${RIDGE_GRID:-"1e-4 1e-2 1 100"}
CG_TOLERANCE=${CG_TOLERANCE:-1e-8}
CG_MAX_ITERATIONS=${CG_MAX_ITERATIONS:-1000}

mkdir -p "${OUTPUT_DIR}"

model_args=()
if [[ "${INCLUDE_CROSS_TERMS}" == "1" ]]; then
  model_args+=(--include-cross-terms)
fi
if [[ "${CROSS_ALL_PAIRS}" == "1" ]]; then
  model_args+=(--cross-all-pairs)
fi
if [[ "${USE_ENVIRONMENT_TYPES}" == "1" ]]; then
  model_args+=(--use-environment-types)
fi
if [[ "${PARENT_SPECIFIC_KEYS}" == "1" ]]; then
  model_args+=(--parent-specific-keys)
fi
if [[ "${SOLVE_PER_PARENT}" == "1" ]]; then
  model_args+=(--solve-per-parent)
fi
if [[ -n "${RIDGE_GRID}" ]]; then
  read -r -a ridge_values <<< "${RIDGE_GRID}"
  model_args+=(--ridge-grid "${ridge_values[@]}")
fi

/usr/bin/time -v -o "${OUTPUT_DIR}/resource.time" \
  python scripts/qm9_complete_total_local_quadratic_capacity.py \
    --baseline-manifest "${BASELINE_MANIFEST}" \
    --source-run-dir "${SOURCE_RUN_DIR}" \
    --direction-manifest "${DIRECTION_MANIFEST}" \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda:0 \
    --bond-scale "${BOND_SCALE}" \
    --ridge "${RIDGE}" \
    --cg-tolerance "${CG_TOLERANCE}" \
    --cg-max-iterations "${CG_MAX_ITERATIONS}" \
    "${model_args[@]}"
