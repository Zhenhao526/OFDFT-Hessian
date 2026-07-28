#!/usr/bin/env bash
#SBATCH --job-name=merge-legacy-denopt
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail
readonly RESTORE_ROOT=/scratch/xzh/dft/restore
readonly RUN_ROOT=${RESTORE_ROOT}/density_optimization_validation_20260715
exec /scratch/xzh/envs/structures25/bin/python \
  "${RESTORE_ROOT}/legacy_merge_density_optimization.py" \
  --run-root "${RUN_ROOT}" \
  --num-tasks 6 \
  --output-json "${RUN_ROOT}/report.json" \
  > "${RUN_ROOT}/merge.log" 2>&1
