#!/usr/bin/env bash
#SBATCH --job-name=ctH_s2audit
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/stage2_uniform_analysis_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/stage2_uniform_analysis_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

ROOT=/scratch/xzh/models/complete_total_capacity/20260717
WARM=${ROOT}/stage2_direction_uniform_A_v1/mlp_h128_warm
V3=${ROOT}/stage2_direction_uniform_A_v3_near_full/mlp_h128_h10
ANALYSIS=${ROOT}/stage2_direction_uniform_A_v3_near_full/analysis_v1
CONFIRM=${ROOT}/stage2_direction_confirmation_uniform_A_v1
VIB=${ROOT}/stage2_direction_uniform_A_v3_near_full/vibrational_metrics_v1
SELECTION=${ROOT}/stage2_direction_v1/selection_uniform_A_v1/selected_baseline_manifest.json
PROTOCOL=configs/audit/qm9_complete_total_hessian_direction_confirmation_uniform_A_v1.yaml

python scripts/qm9_complete_total_stage2_direction_analysis.py \
  --run "uniform_warm=${WARM}" \
  --run "uniform_v3=${V3}" \
  --output-dir "${ANALYSIS}"

python scripts/qm9_complete_total_direction_confirmation.py evaluate \
  --protocol "${PROTOCOL}" \
  --confirmation-manifest "${CONFIRM}/confirmation_manifest.json" \
  --run "uniform_v3=${V3}" \
  --output-dir "${CONFIRM}/evaluation_v1"

python scripts/qm9_hessian_vibrational_metrics.py \
  --manifest-json "${SELECTION}" \
  --dataset-dir /scratch/xzh/data/QM9PBEForceRandom1000 \
  --capacity-result "uniform_v3=${V3}" \
  --output-dir "${VIB}"
