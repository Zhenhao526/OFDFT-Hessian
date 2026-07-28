#!/usr/bin/env bash
#SBATCH --job-name=ctH_keys800
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/train800_feature_inventory_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/train800_feature_inventory_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

ROOT=/scratch/xzh/models/complete_total_capacity/20260717
STAGE3=${ROOT}/stage3_unseen_parent_v1

/usr/bin/time -v python scripts/prepare_qm9_complete_total_train800_feature_inventory.py \
  --asset-manifest "${STAGE3}/assets_v3_uniform_A/stage3_asset_manifest.json" \
  --source-schema-manifest "${STAGE3}/active_feature_schema_full_floor1_v1/active_feature_schema_manifest.json" \
  --output-dir "${STAGE3}/train800_feature_inventory_floor1_v3" \
  --feature-scale-floor 1.0 \
  --progress-interval 100
