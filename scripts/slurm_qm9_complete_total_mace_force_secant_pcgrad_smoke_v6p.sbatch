#!/usr/bin/env bash
#SBATCH --job-name=ct_mace_h64_pcs
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
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

output=/scratch/xzh/models/complete_total_capacity/20260717/stage5_mace_force_secant_pcgrad_h64_v6p/one_parent_smoke/0028399_s5
mkdir -p "${output}" /scratch/xzh/logs/complete_total_capacity

/usr/bin/time -v -o "${output}/resource.time" \
  python scripts/qm9_complete_total_mace_force_secant_capacity.py \
    --protocol configs/audit/qm9_complete_total_hessian_mace_force_secant_h64_pcgrad_v6p.yaml \
    --arm-id E13_mace_h64_l3_corr3_int3_force_secant_fixed_scale_pcgrad \
    --parent-id 0028399 \
    --run-mode smoke \
    --output-dir "${output}" \
    --device cuda:0
