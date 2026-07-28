#!/usr/bin/env bash
#SBATCH --job-name=ct_mace_h16_jac
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
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

output=/scratch/xzh/models/complete_total_capacity/20260717/stage5_mace_h16_scaled_v6d/jacobian_range_v1/0028399_s75_i8
mkdir -p "${output}" /scratch/xzh/logs/complete_total_capacity

/usr/bin/time -v -o "${output}/resource.time" \
  python scripts/qm9_complete_total_local_scalar_jacobian_range_audit.py \
    --protocol configs/audit/qm9_complete_total_hessian_mace_h16_scaled_smoke_v6d.yaml \
    --arm-id E10d_mace_h16_l2_corr3_cut20_scale1000 \
    --checkpoint /scratch/xzh/models/complete_total_capacity/20260717/stage5_mace_h16_scaled_v6d/one_parent_lbfgs_v1/adam_best_snapshot.ckpt \
    --parent-id 0028399 \
    --output-dir "${output}" \
    --iterations 8 \
    --relative-tolerance 1e-3 \
    --fd-relative-step 3e-6 \
    --nonlinear-alphas 0.01,0.03,0.1,0.3,1.0 \
    --device cuda:0
