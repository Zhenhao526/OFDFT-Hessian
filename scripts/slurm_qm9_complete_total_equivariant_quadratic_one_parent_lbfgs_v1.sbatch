#!/usr/bin/env bash
#SBATCH --job-name=ct_quad_1mol
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4%4
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%A_%a.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%A_%a.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

parents=(0028399 0031108 0132419 0031012 0121249)
parent=${parents[${SLURM_ARRAY_TASK_ID}]}
root=/scratch/xzh/models/complete_total_capacity/20260717/stage5_equivariant_quadratic_v5/lbfgs_v1
output="${root}/one_parent_i100/${parent}"
mkdir -p "${output}" /scratch/xzh/logs/complete_total_capacity

/usr/bin/time -v -o "${output}/resource.time" \
  python scripts/qm9_complete_total_local_scalar_lbfgs_ceiling.py \
    --source-protocol configs/audit/qm9_complete_total_hessian_equivariant_quadratic_v5.yaml \
    --source-arm-id E8_equivariant_quadratic_h128_cut20 \
    --source-checkpoint "${root}/adam_best_through_s500.ckpt" \
    --audit-manifest configs/audit/qm9_complete_total_equivariant_quadratic_one_parent_lbfgs_v1.json \
    --objective joint \
    --parent-id "${parent}" \
    --iterations 100 \
    --learning-rate 0.5 \
    --history-size 20 \
    --max-evaluations-per-iteration 3 \
    --tolerance-grad 1e-9 \
    --tolerance-change 1e-12 \
    --log-interval 10 \
    --output-dir "${output}" \
    --device cuda:0
