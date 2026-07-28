#!/usr/bin/env bash
#SBATCH --job-name=ct_eq_long_f
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

root=/scratch/xzh/models/complete_total_capacity/20260717/stage5_bounded_equivariant_v3_long_range
output_dir="${root}/formal/E4_bounded_equivariant_h64_l2_cut20"
mkdir -p "${output_dir}" /scratch/xzh/logs/complete_total_capacity

/usr/bin/time -v -o "${output_dir}/resource.time" \
  python scripts/qm9_complete_total_local_scalar_full_hessian_capacity.py \
    --protocol configs/audit/qm9_complete_total_hessian_bounded_equivariant_long_range_v3.yaml \
    --arm-id E4_bounded_equivariant_h64_l2_cut20 \
    --run-mode formal \
    --output-dir "${output_dir}" \
    --device cuda:0
