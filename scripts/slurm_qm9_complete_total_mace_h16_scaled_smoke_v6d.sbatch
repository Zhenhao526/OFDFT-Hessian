#!/usr/bin/env bash
#SBATCH --job-name=ct_mace_h16smk
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export PYTHONPATH=/scratch/xzh/vendor/mace_torch_0_3_16
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

root=/scratch/xzh/models/complete_total_capacity/20260717/stage5_mace_h16_scaled_smoke_v6d
output_dir="${root}/smoke/E10d_mace_h16_l2_corr3_cut20_scale1000"
mkdir -p "${output_dir}" /scratch/xzh/logs/complete_total_capacity

/usr/bin/time -v -o "${output_dir}/resource.time" \
  python scripts/qm9_complete_total_local_scalar_full_hessian_capacity.py \
    --protocol configs/audit/qm9_complete_total_hessian_mace_h16_scaled_smoke_v6d.yaml \
    --arm-id E10d_mace_h16_l2_corr3_cut20_scale1000 \
    --run-mode smoke \
    --output-dir "${output_dir}" \
    --device cuda:0
