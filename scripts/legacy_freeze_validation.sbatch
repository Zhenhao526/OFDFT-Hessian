#!/usr/bin/env bash
#SBATCH --job-name=freeze-legacy-val
#SBATCH --partition=compute
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00

set -euo pipefail
readonly RESTORE_ROOT=/scratch/xzh/dft/restore
readonly VALIDATION_ROOT=${RESTORE_ROOT}/scientific_baseline_validation_full_20260715
exec /scratch/xzh/envs/structures25/bin/python \
  "${RESTORE_ROOT}/legacy_freeze_validation_baselines.py" \
  --validation-root "${VALIDATION_ROOT}" \
  --acceptance-report "${RESTORE_ROOT}/model_acceptance_20260715/report.json" \
  --output-json "${VALIDATION_ROOT}/frozen_models.json" \
  > "${VALIDATION_ROOT}/freeze.log" 2>&1
