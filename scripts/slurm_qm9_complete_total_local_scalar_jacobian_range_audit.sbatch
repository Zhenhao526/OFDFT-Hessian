#!/usr/bin/env bash
#SBATCH --job-name=ct_jac_range
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail

source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
if [[ -n "${JACOBIAN_VENDOR_PATH:-}" ]]; then
  export PYTHONPATH="${JACOBIAN_VENDOR_PATH}"
fi

protocol=${JACOBIAN_PROTOCOL:?JACOBIAN_PROTOCOL is required}
arm=${JACOBIAN_ARM:?JACOBIAN_ARM is required}
checkpoint=${JACOBIAN_CHECKPOINT:?JACOBIAN_CHECKPOINT is required}
output=${JACOBIAN_OUTPUT:?JACOBIAN_OUTPUT is required}

mkdir -p "${output}" /scratch/xzh/logs/complete_total_capacity
parent_args=()
if [[ -n "${JACOBIAN_PARENT_ID:-}" ]]; then
  parent_args+=(--parent-id "${JACOBIAN_PARENT_ID}")
fi
anchor_args=()
if [[ "${JACOBIAN_ANCHOR_ENERGY_FORCE:-0}" == "1" ]]; then
  anchor_args+=(--anchor-energy-force)
fi
/usr/bin/time -v -o "${output}/resource.time" \
python scripts/qm9_complete_total_local_scalar_jacobian_range_audit.py \
  --protocol "${protocol}" \
  --arm-id "${arm}" \
  --checkpoint "${checkpoint}" \
  "${parent_args[@]}" \
  "${anchor_args[@]}" \
  --output-dir "${output}" \
  --iterations "${JACOBIAN_ITERATIONS:-30}" \
  --relative-tolerance "${JACOBIAN_RELATIVE_TOLERANCE:-1e-3}" \
  --fd-relative-step "${JACOBIAN_FD_RELATIVE_STEP:-3e-6}" \
  --nonlinear-alphas "${JACOBIAN_NONLINEAR_ALPHAS:-0.01,0.03,0.1,0.3,1.0}" \
  --device cuda:0
