#!/usr/bin/env bash
set -euo pipefail

source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
ROOT=/scratch/xzh/models/complete_total_capacity/20260717
STAGE2=${ROOT}/stage2_direction_v1
MINIMUM_FORMAL_JOB_ID=${STAGE2_MINIMUM_FORMAL_JOB_ID:-3297}

RUN_ARGS=()
for run in "${STAGE2}"/baseline_runs/*_job*; do
  [[ -d "${run}" && -f "${run}/full_hessian_metrics.csv" && -f "${run}/summary.json" ]] || continue
  job_id=${run##*_job}
  if (( job_id >= MINIMUM_FORMAL_JOB_ID )); then
    RUN_ARGS+=(--run-dir "${run}")
  fi
done

python scripts/qm9_complete_total_stage2_select_baselines.py \
  --candidate-manifest "${STAGE2}/candidate_direction_manifest.json" \
  --known-baseline-manifest "${ROOT}/five_parent_baseline_manifest.json" \
  --known-baseline-manifest "${ROOT}/five_parent_baseline_manifest_stable5.json" \
  --known-baseline-manifest "${ROOT}/five_parent_baseline_manifest_stable5_v2.json" \
  --run-dir "${ROOT}/one_parent_0043438_s0_lr0_h0_q0_job3266" \
  --run-dir "${ROOT}/one_parent_0016298_s0_lr0_h0_q0_job3267" \
  --run-dir "${ROOT}/one_parent_0064547_s0_lr0_h0_q0_job3268" \
  "${RUN_ARGS[@]}" \
  --output-dir "${STAGE2}/selection_v2"
