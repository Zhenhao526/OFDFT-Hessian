#!/usr/bin/env bash
#SBATCH --job-name=ct_rf_s2prep
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

root=/scratch/xzh/models/complete_total_capacity/20260717/stage4_train20_direction_v10
mkdir -p "${root}/preflight_v1" /scratch/xzh/logs/complete_total_capacity
python scripts/qm9_complete_total_local_random_feature_stage2.py prepare \
  --protocol configs/audit/qm9_complete_total_hessian_local_random_feature_stage2_v10.yaml \
  --output-dir "${root}/preflight_v1" \
  --device cpu

