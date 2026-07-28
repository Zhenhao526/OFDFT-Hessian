#!/usr/bin/env bash
#SBATCH --job-name=ct_post_v4_vib
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

root=${POST_V4_DIAGNOSTIC_ROOT:?POST_V4_DIAGNOSTIC_ROOT is required}
baseline=/scratch/xzh/models/complete_total_capacity/20260717/five_parent_baseline_manifest_stable5_v2.json
python scripts/qm9_hessian_vibrational_metrics.py \
  --manifest-json "${baseline}" \
  --dataset-dir /scratch/xzh/data/QM9PBEForceRandom1000 \
  --baseline-result "original_A=${baseline}" \
  --capacity-result "lbfgs_joint=${root}/lbfgs_joint_100iter" \
  --capacity-result "lbfgs_hessian_only=${root}/lbfgs_hessian_only_100iter" \
  --output-dir "${root}/vibrational_v1"
