#!/usr/bin/env bash
#SBATCH --job-name=ctH_4body_cap
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/four_body_capacity_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/four_body_capacity_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

ROOT=/scratch/xzh/models/complete_total_capacity/20260717
MOLECULE=${FOUR_BODY_MOLECULE:-0028399}
CAPACITY_ARRAY=${FOUR_BODY_CAPACITY_ARRAY:-${ROOT}/geometry_residual_three_body_full_kernel_scan/c6_l4_s0p50/geometry_three_body_capacity.npz}
CENTER_COUNT=${FOUR_BODY_CENTER_COUNT:-3}
TORSION_ORDER=${FOUR_BODY_TORSION_ORDER:-4}
SIGMA=${FOUR_BODY_SIGMA:-0.5}
BOND_SCALE=${FOUR_BODY_BOND_SCALE:-1.35}
OUTPUT_DIR=${FOUR_BODY_OUTPUT_DIR:-${ROOT}/geometry_residual_four_body_full_c${CENTER_COUNT}_l${TORSION_ORDER}_s${SIGMA}}
mkdir -p "${OUTPUT_DIR}" /scratch/xzh/logs/complete_total_capacity

pytest -q tests/test_qm9_complete_total_geometry_four_body_capacity.py

/usr/bin/time -v -o "${OUTPUT_DIR}/resource.time" \
python scripts/qm9_complete_total_geometry_four_body_capacity.py \
  --manifest "${ROOT}/stage1_manifest.json" \
  --molecule-id "${MOLECULE}" \
  --capacity-array "${CAPACITY_ARRAY}" \
  --output-dir "${OUTPUT_DIR}" \
  --center-count "${CENTER_COUNT}" \
  --torsion-order "${TORSION_ORDER}" \
  --sigma "${SIGMA}" \
  --bond-scale "${BOND_SCALE}" \
  --derivative-step 1e-3
