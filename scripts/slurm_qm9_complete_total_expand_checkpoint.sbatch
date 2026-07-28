#!/usr/bin/env bash
#SBATCH --job-name=ctH_expandckpt
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/expand_checkpoint_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/expand_checkpoint_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

: "${STAGE3_SOURCE_CHECKPOINT:?STAGE3_SOURCE_CHECKPOINT is required}"
: "${STAGE3_SOURCE_SCHEMA:?STAGE3_SOURCE_SCHEMA is required}"
: "${STAGE3_FEATURE_INVENTORY:?STAGE3_FEATURE_INVENTORY is required}"
: "${STAGE3_EXPANDED_CHECKPOINT_OUTPUT:?STAGE3_EXPANDED_CHECKPOINT_OUTPUT is required}"

python scripts/expand_qm9_complete_total_checkpoint_to_feature_inventory.py \
  --source-checkpoint "${STAGE3_SOURCE_CHECKPOINT}" \
  --source-schema-manifest "${STAGE3_SOURCE_SCHEMA}" \
  --feature-inventory-manifest "${STAGE3_FEATURE_INVENTORY}" \
  --output-dir "${STAGE3_EXPANDED_CHECKPOINT_OUTPUT}"
