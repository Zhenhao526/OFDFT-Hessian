#!/usr/bin/env bash
#SBATCH --job-name=ct_3task_sel
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail

source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

root=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/local_scalar_three_task_pcgrad_v4
python scripts/qm9_complete_total_local_scalar_three_task_smoke_analysis.py \
  --protocol configs/audit/qm9_complete_total_hessian_local_scalar_three_task_pcgrad_v4.yaml \
  --run-root "${root}/smoke" \
  --output-dir "${root}/smoke_analysis_v1"
