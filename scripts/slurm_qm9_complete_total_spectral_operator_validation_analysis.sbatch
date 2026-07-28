#!/usr/bin/env bash
#SBATCH --job-name=ctH_sp_cmp
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/spectral_validation_analysis_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/spectral_validation_analysis_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

ROOT=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1
OUTPUT=${SPECTRAL_VALIDATION_ANALYSIS_OUTPUT:-${ROOT}/validation_spectral_block_shared_operator_comparison_v1}
mkdir -p "${OUTPUT}" /scratch/xzh/logs/complete_total_capacity

python scripts/qm9_complete_total_spectral_operator_validation_analysis.py \
  --run h512_source=${ROOT}/validation_spectral_block_shared_operator_v1 \
  --run replay_source=${ROOT}/validation_spectral_block_shared_operator_replay_source_v1 \
  --output-dir "${OUTPUT}"
