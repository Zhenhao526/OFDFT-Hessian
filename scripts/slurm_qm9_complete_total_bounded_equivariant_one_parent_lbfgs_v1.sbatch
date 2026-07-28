#!/usr/bin/env bash
#SBATCH --job-name=ct_eq_1mol
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%A_%a.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%A_%a.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

parents=(0028399 0031108 0132419 0031012 0121249)
parent=${parents[${SLURM_ARRAY_TASK_ID}]}
root=/scratch/xzh/models/complete_total_capacity/20260717/stage5_bounded_equivariant_v2/lbfgs_v1
output="${root}/one_parent_i20/${parent}"
mkdir -p "${output}" /scratch/xzh/logs/complete_total_capacity

/usr/bin/time -v -o "${output}/resource.time" \
  python scripts/qm9_complete_total_local_scalar_lbfgs_ceiling.py \
    --source-protocol configs/audit/qm9_complete_total_hessian_bounded_equivariant_width_depth_v2.yaml \
    --source-arm-id E2_bounded_equivariant_h64_l2 \
    --source-checkpoint "${root}/adam_best_through_s100.ckpt" \
    --audit-manifest configs/audit/qm9_complete_total_bounded_equivariant_one_parent_lbfgs_v1.json \
    --objective joint \
    --parent-id "${parent}" \
    --iterations 20 \
    --learning-rate 0.5 \
    --history-size 20 \
    --max-evaluations-per-iteration 3 \
    --tolerance-grad 1e-9 \
    --tolerance-change 1e-12 \
    --log-interval 5 \
    --output-dir "${output}" \
    --device cuda:0
