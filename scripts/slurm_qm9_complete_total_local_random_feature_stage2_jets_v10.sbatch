#!/usr/bin/env bash
#SBATCH --job-name=ct_rf_s2jet
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=18G
#SBATCH --gres=gpu:1
#SBATCH --array=0-19%8
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%A_%a.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%A_%a.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

root=/scratch/xzh/models/complete_total_capacity/20260717/stage4_train20_direction_v10
mkdir -p "${root}/feature_jets_v1" /scratch/xzh/logs/complete_total_capacity
/usr/bin/time -v -o "${root}/feature_jets_v1/resource_${SLURM_ARRAY_TASK_ID}.time" \
  python scripts/qm9_complete_total_local_random_feature_stage2.py feature-jet \
    --protocol configs/audit/qm9_complete_total_hessian_local_random_feature_stage2_v10.yaml \
    --parent-index "${SLURM_ARRAY_TASK_ID}" \
    --output-dir "${root}/feature_jets_v1" \
    --device cuda:0

