#!/usr/bin/env bash
#SBATCH --job-name=ct_rf_dircov_an
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

root=/scratch/xzh/models/complete_total_capacity/20260717/stage4_train20_direction_v10
python scripts/qm9_complete_total_local_random_feature_stage2_direction_coverage_analysis.py \
  --protocol configs/audit/qm9_complete_total_hessian_local_random_feature_stage2_direction_coverage_v1.yaml \
  --coverage-root "${root}/direction_coverage_v1" \
  --output-dir "${root}/direction_coverage_v1/analysis_v1"

