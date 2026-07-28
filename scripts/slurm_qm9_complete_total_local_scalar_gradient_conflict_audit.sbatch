#!/usr/bin/env bash
#SBATCH --job-name=ct_grad_audit
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail

source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

root=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/local_scalar_joint_capacity_v2_gradnorm
run_dir=${root}/formal/J2_unanchored_message_passing_gradnorm
output_dir=${root}/gradient_conflict_audit_v1
mkdir -p "${output_dir}" /scratch/xzh/logs/complete_total_capacity

/usr/bin/time -v -o "${output_dir}/resource.time" \
python scripts/qm9_complete_total_local_scalar_gradient_conflict_audit.py \
  --protocol configs/audit/qm9_complete_total_hessian_local_scalar_joint_capacity_v2_gradnorm.yaml \
  --arm-id J2_unanchored_message_passing_gradnorm \
  --checkpoint "${run_dir}/best.ckpt" \
  --output "${output_dir}/gradient_conflict_audit.json" \
  --device cuda:0
