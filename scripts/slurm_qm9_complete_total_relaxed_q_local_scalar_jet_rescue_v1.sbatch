#!/usr/bin/env bash
#SBATCH --job-name=ct_q59_rescue
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=18G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${PARENT_INDEX:?PARENT_INDEX is required}"
FEATURE_CHUNK_SIZE=${FEATURE_CHUNK_SIZE:-8}
root=/scratch/xzh/models/complete_total_capacity/20260717/stage6_relaxed_q_train100_v1/stage2p5_local_scalar_v1
mkdir -p "${root}/feature_jets" "${root}/rescues" /scratch/xzh/logs/complete_total_capacity
/usr/bin/time -v -o "${root}/rescues/resource_${PARENT_INDEX}_chunk${FEATURE_CHUNK_SIZE}_${SLURM_JOB_ID}.time" \
  python scripts/qm9_complete_total_relaxed_q_local_scalar.py feature-jet \
    --protocol configs/audit/qm9_complete_total_relaxed_q_stage2p5_local_scalar_v1.yaml \
    --parent-index "$PARENT_INDEX" \
    --feature-chunk-size "$FEATURE_CHUNK_SIZE" \
    --output-dir "${root}/feature_jets" \
    --device cuda:0
