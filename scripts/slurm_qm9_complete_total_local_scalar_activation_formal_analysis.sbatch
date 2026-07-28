#!/usr/bin/env bash
#SBATCH --job-name=ct_act_merge
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

root=${LOCAL_SCALAR_FORMAL_ROOT:-${LOCAL_SCALAR_ACTIVATION_ROOT:-/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/local_scalar_activation_audit_v3}}
protocol=${LOCAL_SCALAR_FORMAL_PROTOCOL:-configs/audit/qm9_complete_total_hessian_local_scalar_activation_audit_v3.yaml}
smoke_summary=${LOCAL_SCALAR_FORMAL_SMOKE_SUMMARY:-${root}/smoke_analysis_v1/summary.json}
run_root=${LOCAL_SCALAR_FORMAL_RUN_ROOT:-${root}/formal}
output=${LOCAL_SCALAR_FORMAL_OUTPUT:-${root}/formal_analysis_v1}
python scripts/qm9_complete_total_local_scalar_activation_formal_analysis.py \
  --protocol "${protocol}" \
  --smoke-summary "${smoke_summary}" \
  --run-root "${run_root}" \
  --output-dir "${output}"
