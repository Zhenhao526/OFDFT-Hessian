#!/usr/bin/env bash
#SBATCH --job-name=ct_v20_fit
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

root=/scratch/xzh/models/complete_total_capacity/20260717/stage6_relaxed_vector_train20_v1
mkdir -p "${root}/fit" /scratch/xzh/logs/complete_total_capacity
/usr/bin/time -v -o "${root}/fit/resource.time" \
  python scripts/qm9_complete_total_relaxed_vector_local_scalar.py \
    --protocol configs/audit/qm9_complete_total_relaxed_vector_stage2p5_local_scalar_v1.yaml \
    --output-dir "${root}/fit" \
    --device cuda:0
