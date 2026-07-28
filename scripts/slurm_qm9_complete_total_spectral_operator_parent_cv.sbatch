#!/usr/bin/env bash
#SBATCH --job-name=ctH_sp_pcv
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/spectral_parent_cv_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/spectral_parent_cv_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

: "${OUTPUT_DIR:?set OUTPUT_DIR}"
: "${VARIANT_ID:?set VARIANT_ID}"
PROTOCOL=${PARENT_CV_PROTOCOL:-/scratch/xzh/code/structures25/configs/audit/qm9_complete_total_hessian_parent_cv_bounded_operator_v1.yaml}
BASELINE_MANIFEST=${BASELINE_MANIFEST:-/scratch/xzh/models/complete_total_capacity/20260717/stage2_direction_v1/selection_uniform_A_v1/selected_baseline_manifest.json}
DIRECTION_MANIFEST=${DIRECTION_MANIFEST:-/scratch/xzh/models/complete_total_capacity/20260717/stage2_direction_v5_symmetric_completion/candidate_direction_manifest.json}
SOURCE_FROM_BASELINE=${SOURCE_FROM_BASELINE:-1}
INCLUDE_BLOCK_BASIS=${INCLUDE_BLOCK_BASIS:-1}
BLOCK_PARENT_CONDITIONING=${BLOCK_PARENT_CONDITIONING:-1}
PARENT_FEATURE_TRANSFORM=${PARENT_FEATURE_TRANSFORM:-standard}
PARENT_FEATURE_TRANSFORM_SCALE=${PARENT_FEATURE_TRANSFORM_SCALE:-2.0}
MAX_CORRECTION_TO_SOURCE=${MAX_CORRECTION_TO_SOURCE:-inf}

mkdir -p "${OUTPUT_DIR}" /scratch/xzh/logs/complete_total_capacity
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

/usr/bin/time -v -o "${OUTPUT_DIR}/resource.time" \
python scripts/qm9_complete_total_spectral_operator_parent_cv.py \
  --protocol "${PROTOCOL}" \
  --variant-id "${VARIANT_ID}" \
  --baseline-manifest "${BASELINE_MANIFEST}" \
  --direction-manifest "${DIRECTION_MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --fold-count 5 \
  --parent-feature-transform "${PARENT_FEATURE_TRANSFORM}" \
  --parent-feature-transform-scale "${PARENT_FEATURE_TRANSFORM_SCALE}" \
  --max-correction-to-source "${MAX_CORRECTION_TO_SOURCE}" \
  "${EXTRA_ARGS[@]}"
