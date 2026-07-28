#!/usr/bin/env bash
#SBATCH --job-name=ct_eq_wd
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%A_%a.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%A_%a.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

arms=(E2_bounded_equivariant_h64_l2 E3_bounded_equivariant_h64_l3)
arm="${arms[${SLURM_ARRAY_TASK_ID}]}"
root=/scratch/xzh/models/complete_total_capacity/20260717/stage5_bounded_equivariant_v2
output_dir="${root}/smoke/${arm}"
mkdir -p "${output_dir}" /scratch/xzh/logs/complete_total_capacity

/usr/bin/time -v -o "${output_dir}/resource.time" \
  python scripts/qm9_complete_total_local_scalar_full_hessian_capacity.py \
    --protocol configs/audit/qm9_complete_total_hessian_bounded_equivariant_width_depth_v2.yaml \
    --arm-id "${arm}" \
    --run-mode smoke \
    --output-dir "${output_dir}" \
    --device cuda:0
