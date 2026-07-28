#!/usr/bin/env bash
#SBATCH --job-name=ct_eq_l2s
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

root=/scratch/xzh/models/complete_total_capacity/20260717/stage5_tensor_equivariant_v4b_scaled
output_dir="${root}/smoke_fixed_builder/E6_tensor_equivariant_scaled1000"
mkdir -p "${output_dir}" /scratch/xzh/logs/complete_total_capacity

/usr/bin/time -v -o "${output_dir}/resource.time" \
  python scripts/qm9_complete_total_local_scalar_full_hessian_capacity.py \
    --protocol configs/audit/qm9_complete_total_hessian_tensor_equivariant_scaled_v4b.yaml \
    --arm-id E6_tensor_equivariant_scaled1000 \
    --run-mode smoke \
    --output-dir "${output_dir}" \
    --device cuda:0
