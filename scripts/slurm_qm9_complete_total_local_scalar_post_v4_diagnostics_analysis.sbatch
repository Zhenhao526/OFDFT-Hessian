#!/usr/bin/env bash
#SBATCH --job-name=ct_post_v4_merge
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail

source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

root=${POST_V4_DIAGNOSTIC_ROOT:-/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/local_scalar_three_task_pcgrad_v4/post_v4_diagnostics_v1}
python scripts/qm9_complete_total_local_scalar_post_v4_diagnostics_analysis.py \
  --manifest "${root}/audit_manifest.json" \
  --jacobian-summary "${root}/jacobian_range_30iter/summary.json" \
  --jacobian-metrics "${root}/jacobian_range_30iter/cgls_metrics.jsonl" \
  --joint-summary "${root}/lbfgs_joint_100iter/summary.json" \
  --joint-metrics "${root}/lbfgs_joint_100iter/training_metrics.jsonl" \
  --hessian-summary "${root}/lbfgs_hessian_only_100iter/summary.json" \
  --hessian-metrics "${root}/lbfgs_hessian_only_100iter/training_metrics.jsonl" \
  --vibrational-summary "${root}/vibrational_v1/summary.json" \
  --output-dir "${root}/analysis_v1"
