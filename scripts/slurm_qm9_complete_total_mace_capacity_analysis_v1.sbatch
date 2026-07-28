#!/usr/bin/env bash
#SBATCH --job-name=ct_mace_merge
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

root=/scratch/xzh/models/complete_total_capacity/20260717
python scripts/qm9_complete_total_mace_capacity_analysis.py \
  --run h8_corr2="${root}/stage5_mace_scaled_v6c/formal/E9c_mace_h8_l2_corr2_cut20_scale1000" \
  --run h16_corr3="${root}/stage5_mace_h16_scaled_v6d/formal/E10d_mace_h16_l2_corr3_cut20_scale1000" \
  --output-dir "${root}/stage5_mace_comparison_v1"
