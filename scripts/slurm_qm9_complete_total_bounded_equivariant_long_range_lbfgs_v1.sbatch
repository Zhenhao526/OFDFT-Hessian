#!/usr/bin/env bash
#SBATCH --job-name=ct_long_lbfgs
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

root=/scratch/xzh/models/complete_total_capacity/20260717/stage5_bounded_equivariant_v3_long_range/lbfgs_v1
output="${root}/joint_s350_i30"
mkdir -p "${output}" /scratch/xzh/logs/complete_total_capacity

/usr/bin/time -v -o "${output}/resource.time" \
  python scripts/qm9_complete_total_local_scalar_lbfgs_ceiling.py \
    --source-protocol configs/audit/qm9_complete_total_hessian_bounded_equivariant_long_range_v3.yaml \
    --source-arm-id E4_bounded_equivariant_h64_l2_cut20 \
    --source-checkpoint "${root}/adam_best_through_s350.ckpt" \
    --audit-manifest configs/audit/qm9_complete_total_bounded_equivariant_long_range_lbfgs_v1.json \
    --objective joint \
    --iterations 30 \
    --learning-rate 0.5 \
    --history-size 20 \
    --max-evaluations-per-iteration 3 \
    --tolerance-grad 1e-9 \
    --tolerance-change 1e-12 \
    --log-interval 5 \
    --output-dir "${output}" \
    --device cuda:0
