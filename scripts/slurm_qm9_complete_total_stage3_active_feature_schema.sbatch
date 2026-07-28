#!/usr/bin/env bash
#SBATCH --job-name=ctH_s3schema
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/stage3_active_schema_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/stage3_active_schema_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

ROOT=/scratch/xzh/models/complete_total_capacity/20260717
DESIGN=${STAGE3_SCHEMA_DESIGN_ROOT:-${ROOT}/stage2_direction_v1/shared_descriptor_design_v2}
CHECKPOINT=${STAGE3_SCHEMA_CHECKPOINT:-${ROOT}/stage2_direction_uniform_A_v3_near_full/mlp_h128_h10/best.ckpt}
OUTPUT=${STAGE3_SCHEMA_OUTPUT:-${ROOT}/stage3_unseen_parent_v1/active_feature_schema_uniform_v3}

python scripts/prepare_qm9_complete_total_active_feature_schema.py \
  --design-root "${DESIGN}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT}" \
  --column-norm-relative-cutoff "${STAGE3_SCHEMA_COLUMN_NORM_RELATIVE_CUTOFF:-1e-8}" \
  --feature-scale-mode "${STAGE3_SCHEMA_FEATURE_SCALE_MODE:-column_norm}" \
  --feature-scale-floor "${STAGE3_SCHEMA_FEATURE_SCALE_FLOOR:-0}" \
  --norm-chunk-rows 1024
