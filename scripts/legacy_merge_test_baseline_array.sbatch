#!/usr/bin/env bash
#SBATCH --job-name=merge-legacy-test
#SBATCH --partition=compute
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

readonly RESTORE_ROOT=/scratch/xzh/dft/restore
readonly OUTPUT_ROOT=${RESTORE_ROOT}/scientific_baseline_test_full_20260715
case ${SLURM_ARRAY_TASK_ID} in
  0) run_name=qm9_model__qm9_domain ;;
  1) run_name=qmugs_model__qmugs_domain ;;
  2) run_name=qm9_model__combined_domain ;;
  3) run_name=qmugs_model__combined_domain ;;
  *) exit 65 ;;
esac
run_root=${OUTPUT_ROOT}/${run_name}
exec /scratch/xzh/envs/structures25/bin/python \
  "${RESTORE_ROOT}/legacy_merge_scientific_baseline.py" \
  --run-root "${run_root}" \
  --num-shards 8 \
  --output-json "${run_root}/report.json" \
  --outliers-jsonl-gz "${run_root}/outliers.jsonl.gz" \
  --top-k 100 \
  > "${run_root}/merge.log" 2>&1
