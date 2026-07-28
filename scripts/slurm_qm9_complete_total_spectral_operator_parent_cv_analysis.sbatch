#!/usr/bin/env bash
#SBATCH --job-name=ct_parent_cv_analysis
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/xzh/logs/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/%x_%j.err

set -euo pipefail

source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

base="${PARENT_CV_BASE:-/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/parent_cv_original_A_v1}"
protocol="${PARENT_CV_PROTOCOL:-configs/audit/qm9_complete_total_hessian_parent_cv_bounded_operator_v1.yaml}"
output="${PARENT_CV_ANALYSIS_OUTPUT:-${base}/analysis_v1}"

python scripts/qm9_complete_total_spectral_operator_parent_cv_analysis.py \
  --protocol "${protocol}" \
  --run "A_standard_conditioned_unbounded=${base}/A_standard_conditioned_unbounded" \
  --run "B_tanh2_conditioned_unbounded=${base}/B_tanh2_conditioned_unbounded" \
  --run "C_tanh2_conditioned_cap1=${base}/C_tanh2_conditioned_cap1" \
  --run "D_clip3_conditioned_cap1=${base}/D_clip3_conditioned_cap1" \
  --run "E_tanh2_unconditioned_cap1=${base}/E_tanh2_unconditioned_cap1" \
  --run "F_constant_unconditioned_cap1=${base}/F_constant_unconditioned_cap1" \
  --run "G_tanh2_conditioned_cap0p5=${base}/G_tanh2_conditioned_cap0p5" \
  --source-vibrational-summary "${base}/original_A_source_vibrational/summary.json" \
  --output-dir "${output}"
