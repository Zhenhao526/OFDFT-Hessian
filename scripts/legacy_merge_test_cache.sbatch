#!/usr/bin/env bash
#SBATCH --job-name=merge-legacy-test-cache
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail
readonly RESTORE_ROOT=/scratch/xzh/dft/restore
readonly RUN_ROOT=${RESTORE_ROOT}/baseline_cache_test_20260715
exec /scratch/xzh/envs/structures25/bin/python \
  "${RESTORE_ROOT}/legacy_merge_baseline_cache.py" \
  --run-root "${RUN_ROOT}" \
  --num-shards 16 \
  --output-json "${RUN_ROOT}/report.json" \
  > "${RUN_ROOT}/merge.log" 2>&1
