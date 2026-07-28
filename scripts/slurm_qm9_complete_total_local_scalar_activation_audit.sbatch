#!/usr/bin/env bash
#SBATCH --job-name=ct_act_audit
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%A_%a.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%A_%a.err

set -euo pipefail

source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

run_mode=${LOCAL_SCALAR_RUN_MODE:-smoke}
protocol=configs/audit/qm9_complete_total_hessian_local_scalar_activation_audit_v3.yaml
root=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/local_scalar_activation_audit_v3
arms=(
  J2b_message_passing_softplus_beta1_control
  J3_message_passing_softplus_beta5
  J4_message_passing_silu
  J5_message_passing_tanh
)
arm=${arms[${SLURM_ARRAY_TASK_ID}]}
output=${LOCAL_SCALAR_OUTPUT_BASE:-${root}/${run_mode}}/${arm}

mkdir -p "${output}" /scratch/xzh/logs/complete_total_capacity
/usr/bin/time -v -o "${output}/resource.time" \
python scripts/qm9_complete_total_local_scalar_full_hessian_capacity.py \
  --protocol "${protocol}" \
  --arm-id "${arm}" \
  --run-mode "${run_mode}" \
  --output-dir "${output}" \
  --device cuda:0
