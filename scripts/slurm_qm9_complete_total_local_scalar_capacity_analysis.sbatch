#!/usr/bin/env bash
#SBATCH --job-name=ct_local_merge
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

protocol=${LOCAL_SCALAR_ANALYSIS_PROTOCOL:-configs/audit/qm9_complete_total_hessian_local_scalar_capacity_v1.yaml}
root=${LOCAL_SCALAR_ANALYSIS_ROOT:-/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/local_scalar_capacity_v1}
run_root=${LOCAL_SCALAR_ANALYSIS_RUN_ROOT:-${root}/formal}
output=${LOCAL_SCALAR_ANALYSIS_OUTPUT:-${root}/analysis_v1}
python scripts/qm9_complete_total_local_scalar_capacity_analysis.py \
  --protocol "${protocol}" \
  --run-root "${run_root}" \
  --output-dir "${output}"
