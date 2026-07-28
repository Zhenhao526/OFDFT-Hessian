#!/usr/bin/env bash
#SBATCH --job-name=ct_angular
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail

source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

run_mode=${LOCAL_ANGULAR_RUN_MODE:-smoke}
protocol=configs/audit/qm9_complete_total_hessian_local_angular_capacity_v5.yaml
root=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/local_angular_capacity_v5
arm=M1_local_angular_tanh
output=${LOCAL_ANGULAR_OUTPUT_DIR:-${root}/${run_mode}/${arm}}

mkdir -p "${output}" /scratch/xzh/logs/complete_total_capacity
/usr/bin/time -v -o "${output}/resource.time" \
python scripts/qm9_complete_total_local_scalar_full_hessian_capacity.py \
  --protocol "${protocol}" \
  --arm-id "${arm}" \
  --run-mode "${run_mode}" \
  --output-dir "${output}" \
  --device cuda:0
