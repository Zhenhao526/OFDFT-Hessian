#!/usr/bin/env bash
#SBATCH --job-name=ctH_3body_cap
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/three_body_capacity_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/three_body_capacity_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

ROOT=/scratch/xzh/models/complete_total_capacity/20260717
MOLECULE=${THREE_BODY_MOLECULE:-0028399}
CAPACITY_ARRAY=${THREE_BODY_CAPACITY_ARRAY:-${ROOT}/one_parent_0028399_s50_lr1e-6_h1_q0_job3175/hessian_arrays/step_0000200_0028399.npz}
CENTER_COUNT=${THREE_BODY_CENTER_COUNT:-4}
SIGMA=${THREE_BODY_SIGMA:-0.5}
ANGULAR_ORDER=${THREE_BODY_ANGULAR_ORDER:-3}
DERIVATIVE_STEP=${THREE_BODY_DERIVATIVE_STEP:-1e-3}
OUTPUT_DIR=${THREE_BODY_OUTPUT_DIR:-${ROOT}/geometry_residual_three_body_step200_c${CENTER_COUNT}_s${SIGMA}_l${ANGULAR_ORDER}}
mkdir -p "${OUTPUT_DIR}" /scratch/xzh/logs/complete_total_capacity

ENERGY_ARG=()
if [[ -n "${THREE_BODY_BASELINE_TOTAL_ENERGY:-}" ]]; then
  ENERGY_ARG=(--baseline-total-energy "${THREE_BODY_BASELINE_TOTAL_ENERGY}")
fi

pytest -q \
  tests/test_qm9_complete_total_geometry_three_body_capacity.py \
  tests/test_qm9_complete_total_geometry_residual_capacity.py

/usr/bin/time -v -o "${OUTPUT_DIR}/resource.time" \
python scripts/qm9_complete_total_geometry_three_body_capacity.py \
  --manifest "${ROOT}/stage1_manifest.json" \
  --molecule-id "${MOLECULE}" \
  --capacity-array "${CAPACITY_ARRAY}" \
  --output-dir "${OUTPUT_DIR}" \
  --center-count "${CENTER_COUNT}" \
  --sigma "${SIGMA}" \
  --angular-order "${ANGULAR_ORDER}" \
  --derivative-step "${DERIVATIVE_STEP}" \
  "${ENERGY_ARG[@]}"
