#!/usr/bin/env bash
#SBATCH --job-name=hvp_strict_prep
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/xzh/logs/hvp_curvature/strict_prep_%j.out
#SBATCH --error=/scratch/xzh/logs/hvp_curvature/strict_prep_%j.err

set -euo pipefail
cd /scratch/xzh/code/structures25
source /scratch/xzh/env.sh
TASKS=/scratch/xzh/models/hvp_curvature_v1/20260716/strict_tasks_v1.tsv
python scripts/prepare_qm9_hvp_curvature_strict_tasks.py \
  --protocol configs/audit/qm9_hvp_curvature_validation_v1.yaml \
  --multiseed-tasks /scratch/xzh/models/hvp_curvature_v1/20260716/multiseed_tasks_v2.tsv \
  --multiseed-analysis /scratch/xzh/models/hvp_curvature_v1/20260716/multiseed_analysis/summary.json \
  --stable-sidecar-dir /scratch/xzh/models/hvp_branch_stability/20260716/formal_val20_v3/analysis_v3/stable_sidecars \
  --reference-dir /scratch/xzh/models/hvp100/20260716/pbe_hessians/cache \
  --output "${TASKS}"
COUNT=$(($(wc -l < "${TASKS}") - 1))
if [[ "${COUNT}" -eq 0 ]]; then
  echo "No A-E setting passed the three-seed fixed-density funnel; strict eval not submitted"
  exit 0
fi
STRICT_JOB_ID="$(sbatch --parsable --array="0-$((COUNT - 1))%40" \
  --export=ALL,HVP_STRICT_TASKS="${TASKS}" \
  scripts/slurm_qm9_hvp_curvature_strict_eval.sbatch)"
ANALYSIS_JOB_ID="$(sbatch --parsable --dependency="afterany:${STRICT_JOB_ID}" \
  --export=ALL,HVP_STRICT_TASKS="${TASKS}" \
  scripts/slurm_qm9_hvp_curvature_strict_analysis.sbatch)"
printf 'strict_job_id=%s\nstrict_analysis_job_id=%s\n' "${STRICT_JOB_ID}" "${ANALYSIS_JOB_ID}"
