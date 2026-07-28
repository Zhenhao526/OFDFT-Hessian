#!/usr/bin/env bash
#SBATCH --job-name=ct_ang_linear
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail

source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

output=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/local_angular_capacity_v5/linear_ceiling_v2_chunked
mkdir -p "${output}" /scratch/xzh/logs/complete_total_capacity
/usr/bin/time -v -o "${output}/resource.time" \
python scripts/qm9_complete_total_local_angular_linear_ceiling.py \
  --protocol configs/audit/qm9_complete_total_hessian_local_angular_capacity_v5.yaml \
  --arm-id M1_local_angular_tanh \
  --output-dir "${output}" \
  --device cuda:0 \
  --ridge 1e-8 \
  --column-floor 1e-10 \
  --feature-chunk-size 16
