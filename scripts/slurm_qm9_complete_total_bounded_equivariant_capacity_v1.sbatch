#!/usr/bin/env bash
#SBATCH --job-name=ct_eq_cap
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mode="${RUN_MODE:-smoke}"
root=/scratch/xzh/models/complete_total_capacity/20260717/stage5_bounded_equivariant_v1
output_dir="${root}/${mode}/E1_bounded_equivariant_h32_l2"
mkdir -p "${output_dir}" /scratch/xzh/logs/complete_total_capacity

/usr/bin/time -v -o "${output_dir}/resource.time" \
  python scripts/qm9_complete_total_local_scalar_full_hessian_capacity.py \
    --protocol configs/audit/qm9_complete_total_hessian_bounded_equivariant_capacity_v1.yaml \
    --arm-id E1_bounded_equivariant_h32_l2 \
    --run-mode "${mode}" \
    --output-dir "${output_dir}" \
    --device cuda:0
