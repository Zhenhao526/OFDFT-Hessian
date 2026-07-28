#!/usr/bin/env bash
set -euo pipefail

# This script is intentionally not auto-submitted. It resumes only the frozen missing-task table.
SNAPSHOT_DIR="${1:-/scratch/xzh/models/hvp_curvature_v1/20260716/pause_node01_20260716}"
MISSING_TASKS="${SNAPSHOT_DIR}/missing_train100_tasks.tsv"
ROOT=/scratch/xzh/models/hvp_branch_stability/20260716
CODE=/scratch/xzh/code/structures25
COUNT=$(($(wc -l < "${MISSING_TASKS}") - 1))
[[ "${COUNT}" -gt 0 ]] || { echo "No missing train100 task to resume"; exit 0; }

cd "${CODE}"
TRAIN_BRANCH_JOB_ID="$(sbatch --parsable --nodelist=node01 --array="0-$((COUNT - 1))%8" \
  --export=ALL,HVP_BRANCH_ROOT="${ROOT}",HVP_BRANCH_TASKS="${MISSING_TASKS}",HVP_BRANCH_TASK_SUBDIR=formal_train100_v2/train100_tasks,HVP_BRANCH_TASK_OFFSET=0 \
  scripts/slurm_qm9_hvp_branch_stability_array.sbatch)"

VAL_TASKS="${ROOT}/formal_val20_remaining_v2/remaining12_tasks.tsv"
VAL_COUNT=$(($(wc -l < "${VAL_TASKS}") - 1))
VAL_BRANCH_JOB_ID="$(sbatch --parsable --nodelist=node01 \
  --dependency="afterok:${TRAIN_BRANCH_JOB_ID}" --array="0-$((VAL_COUNT - 1))%8" \
  --export=ALL,HVP_BRANCH_ROOT="${ROOT}",HVP_BRANCH_TASKS="${VAL_TASKS}",HVP_BRANCH_TASK_SUBDIR=formal_val20_remaining_v2/validation_remaining_tasks,HVP_BRANCH_TASK_OFFSET=0 \
  scripts/slurm_qm9_hvp_branch_stability_array.sbatch)"

VAL_POST_JOB_ID="$(sbatch --parsable --nodelist=node01 \
  --dependency="afterok:${VAL_BRANCH_JOB_ID}" \
  scripts/slurm_qm9_hvp_val20_postprocess.sbatch)"
TRAIN_POST_JOB_ID="$(sbatch --parsable --nodelist=node01 \
  --dependency="afterok:${TRAIN_BRANCH_JOB_ID}" \
  --export=ALL,HVP_VAL_POST_JOB_ID="${VAL_POST_JOB_ID}" \
  scripts/slurm_qm9_hvp_train100_postprocess.sbatch)"

printf 'train_branch_job_id=%s\nval_branch_job_id=%s\nval_post_job_id=%s\ntrain_post_job_id=%s\n' \
  "${TRAIN_BRANCH_JOB_ID}" "${VAL_BRANCH_JOB_ID}" "${VAL_POST_JOB_ID}" "${TRAIN_POST_JOB_ID}"
