#!/usr/bin/env bash
#SBATCH --job-name=ct_mace_1m_merge
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export MPLBACKEND=Agg

root=/scratch/xzh/models/complete_total_capacity/20260717/stage5_mace_h16_scaled_v6d/one_parent_lbfgs_v1
python scripts/qm9_complete_total_one_parent_lbfgs_analysis.py \
  --root "${root}" \
  --parent-id 0028399 \
  --parent-id 0031108 \
  --parent-id 0132419 \
  --parent-id 0031012 \
  --parent-id 0121249 \
  --source-checkpoint-sha256 33f5842f150d3a822fe31177f59fb7262cec781bbf217fbe2a416d79f0893229 \
  --output-dir "${root}/analysis_v1"
