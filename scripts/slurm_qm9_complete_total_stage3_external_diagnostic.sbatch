#!/usr/bin/env bash
#SBATCH --job-name=ctH_s3extdiag
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/stage3_external_diagnostic_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/stage3_external_diagnostic_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

ROOT=/scratch/xzh/models/complete_total_capacity/20260717
STAGE3=${ROOT}/stage3_unseen_parent_v1
OUTPUT=${STAGE3_EXTERNAL_DIAGNOSTIC_OUTPUT:-${STAGE3}/validation_external_eval_stage2_s30000_uniform_A_diagnostic_v2}
mkdir -p "${OUTPUT}" /scratch/xzh/logs/complete_total_capacity

/usr/bin/time -v -o "${OUTPUT}/resource.time" \
python scripts/qm9_complete_total_geometry_mlp_external_eval.py \
  --baseline-manifest "${STAGE3}/validation_baselines_v1_uniform_A/validation_baseline_manifest.json" \
  --active-schema-manifest "${STAGE3}/active_feature_schema_uniform_v3/active_feature_schema_manifest.json" \
  --checkpoint "${ROOT}/stage2_direction_uniform_A_v3_near_full/mlp_h128_h10/best.ckpt" \
  --output-dir "${OUTPUT}" \
  --device cpu
