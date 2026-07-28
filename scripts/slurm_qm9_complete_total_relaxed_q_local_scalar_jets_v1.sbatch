#!/usr/bin/env bash
#SBATCH --job-name=ct_q59_jet
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=18G
#SBATCH --gres=gpu:1
#SBATCH --array=0-58%8
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%A_%a.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%A_%a.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

root=/scratch/xzh/models/complete_total_capacity/20260717/stage6_relaxed_q_train100_v1/stage2p5_local_scalar_v1
mkdir -p "${root}/feature_jets" /scratch/xzh/logs/complete_total_capacity
/usr/bin/time -v -o "${root}/feature_jets/resource_${SLURM_ARRAY_TASK_ID}.time" \
  python scripts/qm9_complete_total_relaxed_q_local_scalar.py feature-jet \
    --protocol configs/audit/qm9_complete_total_relaxed_q_stage2p5_local_scalar_v1.yaml \
    --parent-index "${SLURM_ARRAY_TASK_ID}" \
    --output-dir "${root}/feature_jets" \
    --device cuda:0
