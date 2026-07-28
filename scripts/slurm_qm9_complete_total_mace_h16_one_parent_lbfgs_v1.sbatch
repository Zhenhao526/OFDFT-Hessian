#!/usr/bin/env bash
#SBATCH --job-name=ct_mace_h16_1m
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4%4
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%A_%a.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%A_%a.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export PYTHONPATH=/scratch/xzh/vendor/mace_torch_0_3_16
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

parents=(0028399 0031108 0132419 0031012 0121249)
parent=${parents[${SLURM_ARRAY_TASK_ID}]}
root=/scratch/xzh/models/complete_total_capacity/20260717/stage5_mace_h16_scaled_v6d/one_parent_lbfgs_v1
output="${root}/${parent}"
mkdir -p "${output}" /scratch/xzh/logs/complete_total_capacity

/usr/bin/time -v -o "${output}/resource.time" \
  python scripts/qm9_complete_total_local_scalar_lbfgs_ceiling.py \
    --source-protocol configs/audit/qm9_complete_total_hessian_mace_h16_scaled_smoke_v6d.yaml \
    --source-arm-id E10d_mace_h16_l2_corr3_cut20_scale1000 \
    --source-checkpoint "${root}/adam_best_snapshot.ckpt" \
    --audit-manifest configs/audit/qm9_complete_total_mace_h16_one_parent_lbfgs_v1.json \
    --objective joint \
    --parent-id "${parent}" \
    --iterations 50 \
    --learning-rate 0.5 \
    --history-size 20 \
    --max-evaluations-per-iteration 3 \
    --tolerance-grad 1e-9 \
    --tolerance-change 1e-12 \
    --log-interval 10 \
    --output-dir "${output}" \
    --device cuda:0
