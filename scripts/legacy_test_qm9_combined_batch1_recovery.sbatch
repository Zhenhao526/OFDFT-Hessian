#!/usr/bin/env bash
#SBATCH --job-name=legacy-test-qm9-combined-b1
#SBATCH --partition=compute
#SBATCH --array=0-7%8
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=08:00:00

set -euo pipefail

readonly CODE_ROOT=/scratch/xzh/code/structures25
readonly DATA_ROOT=/scratch/xzh/dft/data
readonly MODELS_ROOT=/scratch/xzh/dft/models
readonly RESTORE_ROOT=/scratch/xzh/dft/restore
readonly SAFE_ROOT=${RESTORE_ROOT}/group_safe_splits_20260715
readonly OUTPUT_ROOT=${RESTORE_ROOT}/scientific_baseline_test_full_20260715
readonly CACHE_ROOT=${RESTORE_ROOT}/baseline_cache_test_20260715/samples
readonly FREEZE_REPORT=${RESTORE_ROOT}/scientific_baseline_validation_full_20260715/frozen_models.json
readonly DATASET_NAME=QMUGSBin0QM9_perturbed_fock
readonly NUM_SHARDS=8

if [[ ! -f ${FREEZE_REPORT} ]]; then
  echo "Frozen validation artifact is required before any test evaluation" >&2
  exit 67
fi
if [[ ${DATA_ROOT} == /export/scratch/ialgroup* || ${MODELS_ROOT} == /export/scratch/ialgroup* ]]; then
  echo "Refusing forbidden legacy effective path" >&2
  exit 64
fi
/scratch/xzh/envs/structures25/bin/python - "${FREEZE_REPORT}" <<'PY'
import json,sys
report=json.load(open(sys.argv[1]))
if report.get("status") != "frozen":
    raise SystemExit("Validation artifact is not frozen")
PY

export PROJECT_ROOT=${CODE_ROOT}
export DFT_DATA=${DATA_ROOT}
export DFT_MODELS=${MODELS_ROOT}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

hparams_candidates=("${MODELS_ROOT}"/train/runs/088__*/hparams.yaml)
if [[ ${#hparams_candidates[@]} -ne 1 || ! -f ${hparams_candidates[0]} ]]; then exit 66; fi
hparams=${hparams_candidates[0]}
run_root=${OUTPUT_ROOT}/qm9_model__combined_domain
mkdir -p "${run_root}/rows" "${run_root}/summaries" "${run_root}/task_logs_batch1_recovery"

# The original Test array used batch_size=64.  Only this model/domain pair
# exceeded one A100 on the large QMUGS Test graphs.  Batch size changes the
# execution grouping only; checkpoint, samples, selectors, dtype and metrics
# remain frozen.  Failed tasks produced no summary and no accepted row file.
exec /scratch/xzh/envs/structures25/bin/python \
  "${RESTORE_ROOT}/legacy_scientific_baseline_shard.py" \
  --code-root "${CODE_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --models-root "${MODELS_ROOT}" \
  --hparams "${hparams}" \
  --dataset-name "${DATASET_NAME}" \
  --split-file "${SAFE_ROOT}/${DATASET_NAME}/split.pkl" \
  --partition test \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --num-shards "${NUM_SHARDS}" \
  --device cuda:0 \
  --dtype float32 \
  --batch-size 1 \
  --num-workers 4 \
  --scf-iterations=6,-1 \
  --preprocessed-cache-root "${CACHE_ROOT}" \
  --output-jsonl-gz "${run_root}/rows/shard_${SLURM_ARRAY_TASK_ID}.jsonl.gz" \
  --output-summary-json "${run_root}/summaries/shard_${SLURM_ARRAY_TASK_ID}.json" \
  --progress-interval 500 \
  > "${run_root}/task_logs_batch1_recovery/shard_${SLURM_ARRAY_TASK_ID}.log" 2>&1
