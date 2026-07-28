#!/usr/bin/env bash
#SBATCH --job-name=ct_angular_sel
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail

source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

root=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/local_angular_capacity_v5
python scripts/qm9_complete_total_local_angular_smoke_analysis.py \
  --protocol configs/audit/qm9_complete_total_hessian_local_angular_capacity_v5.yaml \
  --run-dir "${root}/smoke/M1_local_angular_tanh" \
  --output-dir "${root}/smoke_analysis_v1"
