#!/usr/bin/env bash
#SBATCH --job-name=ctH_bindkeys
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/bind_feature_inventory_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/bind_feature_inventory_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

: "${STAGE3_FEATURE_INVENTORY:?STAGE3_FEATURE_INVENTORY is required}"
: "${STAGE3_FEATURE_CHECKPOINT:?STAGE3_FEATURE_CHECKPOINT is required}"
: "${STAGE3_FEATURE_SCHEMA_OUTPUT:?STAGE3_FEATURE_SCHEMA_OUTPUT is required}"

python scripts/bind_qm9_complete_total_feature_inventory.py \
  --inventory-manifest "${STAGE3_FEATURE_INVENTORY}" \
  --checkpoint "${STAGE3_FEATURE_CHECKPOINT}" \
  --output-dir "${STAGE3_FEATURE_SCHEMA_OUTPUT}"
