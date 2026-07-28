#!/usr/bin/env bash
#SBATCH --job-name=ct_eq_l2h_f
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

root=/scratch/xzh/models/complete_total_capacity/20260717/stage5_tensor_equivariant_v4c_half_width
output_dir="${root}/formal/E7_tensor_equivariant_half_width"
mkdir -p "${output_dir}" /scratch/xzh/logs/complete_total_capacity

/usr/bin/time -v -o "${output_dir}/resource.time" \
  python scripts/qm9_complete_total_local_scalar_full_hessian_capacity.py \
    --protocol configs/audit/qm9_complete_total_hessian_tensor_equivariant_half_width_v4c.yaml \
    --arm-id E7_tensor_equivariant_half_width \
    --run-mode formal \
    --output-dir "${output_dir}" \
    --device cuda:0
