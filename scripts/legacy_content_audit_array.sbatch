#!/usr/bin/env bash
#SBATCH --job-name=legacy-content-audit
#SBATCH --partition=compute
#SBATCH --array=0-95%8
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=02:00:00

set -euo pipefail

readonly DATA_ROOT=/scratch/xzh/dft/data
readonly RESTORE_ROOT=/scratch/xzh/dft/restore
readonly AUDIT_ROOT=${RESTORE_ROOT}/content_audit_full_20260715
readonly NUM_SHARDS=32

if [[ ${DATA_ROOT} == /export/scratch/ialgroup* ]]; then
  echo "Refusing forbidden legacy data path: ${DATA_ROOT}" >&2
  exit 64
fi

export DFT_DATA=${DATA_ROOT}
export DFT_MODELS=/scratch/xzh/dft/models
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

source_index=$((SLURM_ARRAY_TASK_ID / NUM_SHARDS))
shard_index=$((SLURM_ARRAY_TASK_ID % NUM_SHARDS))
case ${source_index} in
  0) source_name=QM9_perturbed_fock ;;
  1) source_name=QMUGS_perturbed_fock ;;
  2) source_name=QMUGS ;;
  *) echo "Unexpected source index: ${source_index}" >&2; exit 65 ;;
esac

mkdir -p "${AUDIT_ROOT}/rows" "${AUDIT_ROOT}/summaries" "${AUDIT_ROOT}/task_logs"

exec /scratch/xzh/envs/structures25/bin/python \
  "${RESTORE_ROOT}/legacy_dataset_content_audit.py" \
  --data-root "${DATA_ROOT}" \
  --source "${source_name}" \
  --shard-index "${shard_index}" \
  --num-shards "${NUM_SHARDS}" \
  --output-jsonl-gz "${AUDIT_ROOT}/rows/${source_name}_${shard_index}.jsonl.gz" \
  --summary-json "${AUDIT_ROOT}/summaries/${source_name}_${shard_index}.json" \
  --progress-interval 500 \
  > "${AUDIT_ROOT}/task_logs/${source_name}_${shard_index}.log" 2>&1
