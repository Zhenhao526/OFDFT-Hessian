#!/usr/bin/env bash
#SBATCH --job-name=hvp_curve_analyze
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/xzh/logs/hvp_curvature/stage1_analysis_%j.out
#SBATCH --error=/scratch/xzh/logs/hvp_curvature/stage1_analysis_%j.err

set -euo pipefail
cd /scratch/xzh/code/structures25
source /scratch/xzh/env.sh

TASKS="${HVP_CURVATURE_TASKS:?Set HVP_CURVATURE_TASKS}"
STAGE1_ROOT="${HVP_STAGE1_ROOT:-/scratch/xzh/models/hvp_curvature_v1/20260716/stage1}"
OUTPUT="${HVP_STAGE1_ANALYSIS_ROOT:-/scratch/xzh/models/hvp_curvature_v1/20260716/stage1_analysis}"

python scripts/qm9_hvp_curvature_stage1_analysis.py \
  --protocol configs/audit/qm9_hvp_curvature_training_v2.yaml \
  --tasks "${TASKS}" --stage1-root "${STAGE1_ROOT}" --output-dir "${OUTPUT}" \
  --max-per-variant 2
python scripts/qm9_hvp_curvature_training_diagnostics.py \
  --tasks "${TASKS}" --run-root /scratch/xzh/models/train/runs \
  --output-dir "${OUTPUT}/training_diagnostics"

if [[ "${HVP_SUBMIT_MULTISEED:-0}" == 1 ]]; then
  PROTOCOL="${HVP_TRAINING_PROTOCOL:-configs/audit/qm9_hvp_curvature_training_v2.yaml}"
  MULTISEED_TASKS="${HVP_MULTISEED_TASKS:-/scratch/xzh/models/hvp_curvature_v1/20260716/multiseed_tasks_v2.tsv}"
  MULTISEED_STAGE1="${HVP_MULTISEED_STAGE1_ROOT:-/scratch/xzh/models/hvp_curvature_v1/20260716/multiseed_stage1}"
  MULTISEED_ANALYSIS="${HVP_MULTISEED_ANALYSIS_ROOT:-/scratch/xzh/models/hvp_curvature_v1/20260716/multiseed_analysis}"
  python scripts/prepare_qm9_hvp_curvature_multiseed.py \
    --protocol "${PROTOCOL}" --screen-summary "${OUTPUT}/summary.json" \
    --output "${MULTISEED_TASKS}"
  TRAIN_JOB_ID="$(sbatch --parsable --array=0-14%15 \
    --export=ALL,HVP_CURVATURE_TASKS="${MULTISEED_TASKS}" \
    scripts/slurm_qm9_hvp_curvature_multiseed.sbatch)"
  EVAL_JOB_ID="$(sbatch --parsable --dependency="afterok:${TRAIN_JOB_ID}" --array=0-14%15 \
    --export=ALL,HVP_CURVATURE_TASKS="${MULTISEED_TASKS}",HVP_STAGE1_ROOT="${MULTISEED_STAGE1}" \
    scripts/slurm_qm9_hvp_curvature_stage1_eval.sbatch)"
  ANALYSIS_JOB_ID="$(sbatch --parsable --dependency="afterok:${EVAL_JOB_ID}" \
    --export=ALL,HVP_CURVATURE_TASKS="${MULTISEED_TASKS}",HVP_STAGE1_ROOT="${MULTISEED_STAGE1}",HVP_STAGE1_ANALYSIS_ROOT="${MULTISEED_ANALYSIS}",HVP_SUBMIT_MULTISEED=0,HVP_SUBMIT_STRICT=1 \
    scripts/slurm_qm9_hvp_curvature_stage1_analysis.sbatch)"
  printf 'multiseed_train_job_id=%s\nmultiseed_eval_job_id=%s\nmultiseed_analysis_job_id=%s\n' \
    "${TRAIN_JOB_ID}" "${EVAL_JOB_ID}" "${ANALYSIS_JOB_ID}"
fi

if [[ "${HVP_SUBMIT_STRICT:-0}" == 1 ]]; then
  [[ -n "${HVP_VAL_POST_JOB_ID:-}" ]] || {
    echo "HVP_VAL_POST_JOB_ID is required to preserve the stable-only dependency" >&2
    exit 1
  }
  STRICT_PREP_JOB_ID="$(sbatch --parsable --dependency="afterok:${HVP_VAL_POST_JOB_ID}" \
    scripts/slurm_qm9_hvp_curvature_strict_orchestrate.sbatch)"
  printf 'strict_prep_job_id=%s dependency_val_post=%s\n' \
    "${STRICT_PREP_JOB_ID}" "${HVP_VAL_POST_JOB_ID}"
fi
