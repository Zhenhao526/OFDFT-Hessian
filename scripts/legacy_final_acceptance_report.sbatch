#!/usr/bin/env bash
#SBATCH --job-name=legacy-final-report
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail
readonly RESTORE_ROOT=/scratch/xzh/dft/restore
/scratch/xzh/envs/structures25/bin/python \
  "${RESTORE_ROOT}/legacy_immutability_audit.py" \
  --initial-manifest "${RESTORE_ROOT}/recovery_manifest_20260715.json" \
  --output-json "${RESTORE_ROOT}/immutability_audit_20260715.json" \
  > "${RESTORE_ROOT}/immutability_audit_20260715.log" 2>&1
/scratch/xzh/envs/structures25/bin/python \
  "${RESTORE_ROOT}/legacy_pipeline_provenance.py" \
  --output-json "${RESTORE_ROOT}/pipeline_provenance_20260715.json" \
  > "${RESTORE_ROOT}/pipeline_provenance_20260715.log" 2>&1
/scratch/xzh/envs/structures25/bin/python \
  "${RESTORE_ROOT}/legacy_final_acceptance_report.py" \
  --restore-root "${RESTORE_ROOT}" \
  --output-json "${RESTORE_ROOT}/final_acceptance_report_20260715.json" \
  > "${RESTORE_ROOT}/final_acceptance_report_20260715.log" 2>&1
/scratch/xzh/envs/structures25/bin/python \
  "${RESTORE_ROOT}/legacy_render_final_acceptance_report.py" \
  --input-json "${RESTORE_ROOT}/final_acceptance_report_20260715.json" \
  --output-md "${RESTORE_ROOT}/final_acceptance_report_20260715.md" \
  >> "${RESTORE_ROOT}/final_acceptance_report_20260715.log" 2>&1
/scratch/xzh/envs/structures25/bin/python \
  "${RESTORE_ROOT}/legacy_completion_audit.py" \
  --final-report "${RESTORE_ROOT}/final_acceptance_report_20260715.json" \
  --final-markdown "${RESTORE_ROOT}/final_acceptance_report_20260715.md" \
  --handoff "${RESTORE_ROOT}/qm9_force_hessian_project_handoff_20260715.md" \
  --output-json "${RESTORE_ROOT}/completion_audit_20260715.json" \
  >> "${RESTORE_ROOT}/final_acceptance_report_20260715.log" 2>&1
