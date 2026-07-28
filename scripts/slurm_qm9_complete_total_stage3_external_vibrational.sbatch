#!/usr/bin/env bash
#SBATCH --job-name=ctH_s3vib
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/stage3_external_vibrational_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/stage3_external_vibrational_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

ROOT=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1
MANIFEST=${STAGE3_VIBRATION_REFERENCE_MANIFEST:-${ROOT}/validation_baselines_v1_uniform_A/validation_baseline_manifest.json}
RESULT=${STAGE3_VIBRATION_RESULT_DIR:-${ROOT}/validation_external_eval_stage2_s30000_uniform_A}
RESULT_NAME=${STAGE3_VIBRATION_RESULT_NAME:-stage2_s30000_uniform_A}
OUTPUT=${STAGE3_VIBRATION_OUTPUT:-${RESULT}/vibrational_metrics}
BASELINE_RESULT=${STAGE3_VIBRATION_BASELINE_RESULT:-}
RESULT_ARGS=(--capacity-result "${RESULT_NAME}=${RESULT}")
if [[ -n "${BASELINE_RESULT}" ]]; then
  RESULT_ARGS=(--baseline-result "${RESULT_NAME}=${BASELINE_RESULT}")
fi

python scripts/qm9_hessian_vibrational_metrics.py \
  --manifest-json "${MANIFEST}" \
  --dataset-dir /scratch/xzh/data/QM9PBEForceRandom1000 \
  "${RESULT_ARGS[@]}" \
  --output-dir "${OUTPUT}"
