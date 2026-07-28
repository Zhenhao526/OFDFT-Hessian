#!/usr/bin/env bash
#SBATCH --job-name=hvp_strict_analyze
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/xzh/logs/hvp_curvature/strict_analysis_%j.out
#SBATCH --error=/scratch/xzh/logs/hvp_curvature/strict_analysis_%j.err

set -euo pipefail
cd /scratch/xzh/code/structures25
source /scratch/xzh/env.sh
python scripts/qm9_hvp_curvature_strict_analysis.py \
  --protocol configs/audit/qm9_hvp_curvature_validation_v1.yaml \
  --tasks "${HVP_STRICT_TASKS:?Set HVP_STRICT_TASKS}" \
  --strict-root "${HVP_STRICT_ROOT:-/scratch/xzh/models/hvp_curvature_v1/20260716/strict_complete_total}" \
  --stage1-root /scratch/xzh/models/hvp_curvature_v1/20260716/multiseed_stage1 \
  --output-dir /scratch/xzh/models/hvp_curvature_v1/20260716/strict_analysis

TASKS=/scratch/xzh/models/hvp_curvature_v1/20260716/full_hessian_tasks_v1.tsv
MANIFEST=/scratch/xzh/models/hvp_curvature_v1/20260716/full_hessian_tasks_v1.json
python scripts/prepare_qm9_hvp_curvature_full_hessian.py \
  --strict-summary /scratch/xzh/models/hvp_curvature_v1/20260716/strict_analysis/summary.json \
  --strict-per-direction /scratch/xzh/models/hvp_curvature_v1/20260716/strict_analysis/strict_per_direction.csv \
  --reference-dir /scratch/xzh/models/hvp100/20260716/pbe_hessians/cache \
  --output-tsv "${TASKS}" --output-json "${MANIFEST}" --parent-count 5
COUNT=$(($(wc -l < "${TASKS}") - 1))
if [[ "${COUNT}" -eq 0 ]]; then
  echo "No strict-promoted candidate; full Hessian not submitted"
  exit 0
fi
FULL_JOB_ID="$(sbatch --parsable --array="0-$((COUNT - 1))%20" \
  --export=ALL,HVP_FULL_HESSIAN_TASKS="${TASKS}" \
  scripts/slurm_qm9_hvp_curvature_full_hessian.sbatch)"
ANALYSIS_JOB_ID="$(sbatch --parsable --dependency="afterany:${FULL_JOB_ID}" \
  scripts/slurm_qm9_hvp_curvature_full_hessian_analysis.sbatch)"
printf 'full_hessian_job_id=%s\nfull_hessian_analysis_job_id=%s\n' \
  "${FULL_JOB_ID}" "${ANALYSIS_JOB_ID}"
