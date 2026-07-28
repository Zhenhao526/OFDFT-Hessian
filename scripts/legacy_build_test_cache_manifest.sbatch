#!/usr/bin/env bash
#SBATCH --job-name=legacy-test-manifest
#SBATCH --partition=compute
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00

set -euo pipefail
readonly RESTORE_ROOT=/scratch/xzh/dft/restore
readonly FREEZE_REPORT=${RESTORE_ROOT}/scientific_baseline_validation_20260715/frozen_models.json
readonly RUN_ROOT=${RESTORE_ROOT}/baseline_cache_test_20260715
/scratch/xzh/envs/structures25/bin/python - "${FREEZE_REPORT}" <<'PY'
import json,sys
if json.load(open(sys.argv[1])).get("status") != "frozen":
    raise SystemExit("Validation artifact is not frozen")
PY
mkdir -p "${RUN_ROOT}/summaries" "${RUN_ROOT}/logs" "${RUN_ROOT}/samples"
exec /scratch/xzh/envs/structures25/bin/python \
  "${RESTORE_ROOT}/legacy_build_baseline_cache_manifest.py" \
  --safe-root "${RESTORE_ROOT}/group_safe_splits_20260715" \
  --partition test \
  --num-shards 16 \
  --output-json "${RUN_ROOT}/manifest.json" \
  > "${RUN_ROOT}/manifest.log" 2>&1
