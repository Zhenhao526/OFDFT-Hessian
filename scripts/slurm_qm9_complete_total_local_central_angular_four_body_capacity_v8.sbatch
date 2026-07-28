#!/usr/bin/env bash
#SBATCH --job-name=ct_cang4_v8
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail

source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

output_dir=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/local_central_angular_four_body_capacity_v8/exact_ridge_v4_chunk48_resume
mkdir -p "${output_dir}" /scratch/xzh/logs/complete_total_capacity

/usr/bin/time -v -o "${output_dir}/resource.time" \
  python scripts/qm9_complete_total_local_angular_four_body_linear_ceiling.py \
    --protocol configs/audit/qm9_complete_total_hessian_local_central_angular_four_body_capacity_v8.yaml \
    --arm-id M1_local_angular_tanh \
    --output-dir "${output_dir}" \
    --device cuda:0
