#!/usr/bin/env bash
#SBATCH --job-name=hvp_train_post
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/xzh/logs/hvp_curvature/train100_post_%j.out
#SBATCH --error=/scratch/xzh/logs/hvp_curvature/train100_post_%j.err

set -euo pipefail
cd /scratch/xzh/code/structures25
source /scratch/xzh/env.sh
export DFT_DATA=/scratch/xzh/data DFT_MODELS=/scratch/xzh/models

ROOT=/scratch/xzh/models/hvp_branch_stability/20260716
TRAIN_ROOT="${ROOT}/formal_train100_v2"
TASK_ROOT="${TRAIN_ROOT}/train100_tasks"
EXPECTED=/scratch/xzh/models/hvp100/20260716/selection/train_parent_ids.txt
INPUT_SIDECARS="${ROOT}/multidirection_sidecars/sidecars"
VAL_POST_JOB_ID="${HVP_VAL_POST_JOB_ID:?Set HVP_VAL_POST_JOB_ID to the active val20 postprocessing job}"
SUMMARY_COUNT="$(find "${TASK_ROOT}" -name summary.json | wc -l)"
[[ "${SUMMARY_COUNT}" -eq 1200 ]] || {
  echo "Expected 1200 train100 summaries, found ${SUMMARY_COUNT}" >&2
  exit 1
}

python scripts/qm9_hvp_branch_stability_analysis.py \
  --protocol configs/audit/qm9_hvp_branch_stability_v2.yaml \
  --task-root "${TASK_ROOT}" \
  --output-dir "${TRAIN_ROOT}/analysis_strict_v2" \
  --expected-molecules "${EXPECTED}" --expected-directions 4

python scripts/qm9_hvp_branch_stability_analysis.py \
  --protocol configs/audit/qm9_hvp_branch_stability_v3.yaml \
  --task-root "${TASK_ROOT}" \
  --output-dir "${TRAIN_ROOT}/analysis" \
  --expected-molecules "${EXPECTED}" --expected-directions 4 \
  --input-sidecar-dir "${INPUT_SIDECARS}" \
  --output-sidecar-dir "${TRAIN_ROOT}/analysis/stable_sidecars"

python scripts/qm9_build_implicit_target_sidecars.py \
  --stability-analysis-dir "${TRAIN_ROOT}/analysis" \
  --task-root "${TASK_ROOT}" \
  --input-sidecar-dir "${INPUT_SIDECARS}" \
  --output-sidecar-dir "${TRAIN_ROOT}/implicit_targets/sidecars" \
  --parent-limit 10 --expected-directions 4

STABLE_COUNT="$(find "${TRAIN_ROOT}/analysis/stable_sidecars" -name '*.npz' | wc -l)"
IMPLICIT_COUNT="$(find "${TRAIN_ROOT}/implicit_targets/sidecars" -name '*.npz' | wc -l)"
STABLE_MANIFEST="${TRAIN_ROOT}/analysis/stable_sidecars/manifest.json"
IMPLICIT_MANIFEST="${TRAIN_ROOT}/implicit_targets/manifest.json"
[[ "${STABLE_COUNT}" -gt 0 && "${IMPLICIT_COUNT}" -gt 0 && -f "${STABLE_MANIFEST}" && -f "${IMPLICIT_MANIFEST}" ]] || {
  echo "No stable/implicit parent available; screen is not submitted" >&2
  exit 1
}
MANIFEST_COUNT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["sidecar_count"])' "${STABLE_MANIFEST}")"
IMPLICIT_MANIFEST_COUNT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["sidecar_count"])' "${IMPLICIT_MANIFEST}")"
[[ "${STABLE_COUNT}" -eq "${MANIFEST_COUNT}" ]] || {
  echo "Stable sidecar/manifest count mismatch: ${STABLE_COUNT} != ${MANIFEST_COUNT}" >&2
  exit 1
}
[[ "${IMPLICIT_COUNT}" -eq "${IMPLICIT_MANIFEST_COUNT}" ]] || {
  echo "Implicit sidecar/manifest count mismatch: ${IMPLICIT_COUNT} != ${IMPLICIT_MANIFEST_COUNT}" >&2
  exit 1
}

SCREEN_TASKS=/scratch/xzh/models/hvp_curvature_v1/20260716/screen_tasks_v2.tsv
SCREEN_JOB_ID="$(sbatch --parsable --array=0-19%20 \
  --export=ALL,HVP_CURVATURE_TASKS="${SCREEN_TASKS}" \
  scripts/slurm_qm9_hvp_curvature_screen.sbatch)"
EVAL_JOB_ID="$(sbatch --parsable --dependency="afterok:${SCREEN_JOB_ID}:${VAL_POST_JOB_ID}" --array=0-19%20 \
  --export=ALL,HVP_CURVATURE_TASKS="${SCREEN_TASKS}",HVP_STAGE1_ROOT=/scratch/xzh/models/hvp_curvature_v1/20260716/stage1 \
  scripts/slurm_qm9_hvp_curvature_stage1_eval.sbatch)"
ANALYSIS_JOB_ID="$(sbatch --parsable --dependency="afterok:${EVAL_JOB_ID}" \
  --export=ALL,HVP_CURVATURE_TASKS="${SCREEN_TASKS}",HVP_STAGE1_ROOT=/scratch/xzh/models/hvp_curvature_v1/20260716/stage1,HVP_STAGE1_ANALYSIS_ROOT=/scratch/xzh/models/hvp_curvature_v1/20260716/stage1_analysis,HVP_SUBMIT_MULTISEED=1,HVP_VAL_POST_JOB_ID="${VAL_POST_JOB_ID}" \
  scripts/slurm_qm9_hvp_curvature_stage1_analysis.sbatch)"
printf 'screen_job_id=%s\nstage1_eval_job_id=%s\nstage1_analysis_job_id=%s\n' \
  "${SCREEN_JOB_ID}" "${EVAL_JOB_ID}" "${ANALYSIS_JOB_ID}"
