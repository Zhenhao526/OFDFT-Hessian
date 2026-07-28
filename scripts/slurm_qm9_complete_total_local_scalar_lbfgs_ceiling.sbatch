#!/usr/bin/env bash
#SBATCH --job-name=ct_lbfgs_ceiling
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail

source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

source_protocol=${LBFGS_SOURCE_PROTOCOL:?LBFGS_SOURCE_PROTOCOL is required}
source_arm=${LBFGS_SOURCE_ARM:?LBFGS_SOURCE_ARM is required}
source_checkpoint=${LBFGS_SOURCE_CHECKPOINT:?LBFGS_SOURCE_CHECKPOINT is required}
objective=${LBFGS_OBJECTIVE:?LBFGS_OBJECTIVE is required}
output=${LBFGS_OUTPUT:?LBFGS_OUTPUT is required}
audit_manifest=${LBFGS_AUDIT_MANIFEST:?LBFGS_AUDIT_MANIFEST is required}

mkdir -p "${output}" /scratch/xzh/logs/complete_total_capacity
/usr/bin/time -v -o "${output}/resource.time" \
python scripts/qm9_complete_total_local_scalar_lbfgs_ceiling.py \
  --source-protocol "${source_protocol}" \
  --source-arm-id "${source_arm}" \
  --source-checkpoint "${source_checkpoint}" \
  --audit-manifest "${audit_manifest}" \
  --objective "${objective}" \
  --iterations "${LBFGS_ITERATIONS:-100}" \
  --learning-rate "${LBFGS_LEARNING_RATE:-0.5}" \
  --history-size "${LBFGS_HISTORY_SIZE:-25}" \
  --log-interval "${LBFGS_LOG_INTERVAL:-5}" \
  --output-dir "${output}" \
  --device cuda:0
