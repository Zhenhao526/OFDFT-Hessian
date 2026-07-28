#!/usr/bin/env bash
#SBATCH --job-name=ctH_specop
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --nodelist=node01

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

: "${OUTPUT_DIR:?set OUTPUT_DIR}"
BASELINE_MANIFEST=${BASELINE_MANIFEST:-/scratch/xzh/models/complete_total_capacity/20260717/stage2_direction_v1/selection_uniform_A_v1/selected_baseline_manifest.json}
DIRECTION_MANIFEST=${DIRECTION_MANIFEST:-/scratch/xzh/models/complete_total_capacity/20260717/stage2_direction_v3_near_full/candidate_direction_manifest.json}
INCLUDE_BLOCK_BASIS=${INCLUDE_BLOCK_BASIS:-0}
BLOCK_PARENT_CONDITIONING=${BLOCK_PARENT_CONDITIONING:-0}
EXACT_TRAIN_INTERPOLANT=${EXACT_TRAIN_INTERPOLANT:-0}
SOURCE_FROM_BASELINE=${SOURCE_FROM_BASELINE:-0}

mkdir -p "${OUTPUT_DIR}"
EXTRA_ARGS=()
if [[ "${SOURCE_FROM_BASELINE}" == "1" ]]; then
  EXTRA_ARGS+=(--source-from-baseline)
else
  : "${SOURCE_RUN_DIR:?set SOURCE_RUN_DIR or SOURCE_FROM_BASELINE=1}"
  EXTRA_ARGS+=(--source-run-dir "${SOURCE_RUN_DIR}")
fi
if [[ "${INCLUDE_BLOCK_BASIS}" == "1" ]]; then
  EXTRA_ARGS+=(--include-block-basis)
fi
if [[ "${BLOCK_PARENT_CONDITIONING}" == "1" ]]; then
  EXTRA_ARGS+=(--block-parent-conditioning)
fi
if [[ "${EXACT_TRAIN_INTERPOLANT}" == "1" ]]; then
  EXTRA_ARGS+=(--exact-train-interpolant)
fi
/usr/bin/time -v -o "${OUTPUT_DIR}/resource.time" \
  python scripts/qm9_complete_total_spectral_operator_capacity.py \
    --baseline-manifest "${BASELINE_MANIFEST}" \
    --direction-manifest "${DIRECTION_MANIFEST}" \
    --output-dir "${OUTPUT_DIR}" \
    "${EXTRA_ARGS[@]}"
