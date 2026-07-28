#!/usr/bin/env bash
#SBATCH --job-name=ctH_mlp5
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/mlp_capacity_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/mlp_capacity_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

ROOT=/scratch/xzh/models/complete_total_capacity/20260717
OUT=${MLP_CAPACITY_OUTPUT_DIR:-${ROOT}/five_parent_geometry_mlp_h64_seed314159}
mkdir -p "${OUT}" /scratch/xzh/logs/complete_total_capacity

/usr/bin/time -v -o "${OUT}/resource.time" \
python scripts/qm9_complete_total_geometry_mlp_capacity.py \
  --baseline-manifest "${ROOT}/five_parent_baseline_manifest.json" \
  --design-root "${ROOT}/five_parent_shared_geometry_capacity" \
  --initial-root "${MLP_CAPACITY_INITIAL_ROOT:-${ROOT}/five_parent_shared_geometry_resolve/soft_e0p10_f0p05}" \
  --output-dir "${OUT}" \
  --hidden-size "${MLP_CAPACITY_HIDDEN_SIZE:-64}" \
  --seed "${MLP_CAPACITY_SEED:-314159}" \
  --steps "${MLP_CAPACITY_STEPS:-1000}" \
  --learning-rate "${MLP_CAPACITY_LR:-1e-3}" \
  --lambda-hessian "${MLP_CAPACITY_LAMBDA_H:-1}" \
  --lambda-energy "${MLP_CAPACITY_LAMBDA_E:-1}" \
  --lambda-force "${MLP_CAPACITY_LAMBDA_F:-1}"
