#!/usr/bin/env bash
#SBATCH --job-name=legacy-test-cache-scf6
#SBATCH --partition=compute
#SBATCH --array=0-15%16
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail
readonly CODE_ROOT=/scratch/xzh/code/structures25
readonly DATA_ROOT=/scratch/xzh/dft/data
readonly MODELS_ROOT=/scratch/xzh/dft/models
readonly RESTORE_ROOT=/scratch/xzh/dft/restore
readonly FINAL_CACHE_RUN=${RESTORE_ROOT}/baseline_cache_test_20260715
readonly CACHE_RUN=${RESTORE_ROOT}/baseline_cache_test_scf6_20260715
readonly PREPROCESSING_GATE=${RESTORE_ROOT}/scientific_baseline_validation_20260715/frozen_models.json
/scratch/xzh/envs/structures25/bin/python - "${PREPROCESSING_GATE}" <<'PY'
import json,sys
if json.load(open(sys.argv[1])).get("status") != "frozen":
    raise SystemExit("Initial validation freeze is required before Test preprocessing")
PY
export PROJECT_ROOT=${CODE_ROOT}
export DFT_DATA=${DATA_ROOT}
export DFT_MODELS=${MODELS_ROOT}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
hparams_candidates=("${MODELS_ROOT}"/train/runs/088__*/hparams.yaml)
if [[ ${#hparams_candidates[@]} -ne 1 || ! -f ${hparams_candidates[0]} ]]; then exit 66; fi
mkdir -p "${CACHE_RUN}/summaries" "${CACHE_RUN}/logs" "${FINAL_CACHE_RUN}/samples"
exec /scratch/xzh/envs/structures25/bin/python \
  "${RESTORE_ROOT}/legacy_prepare_baseline_cache_shard.py" \
  --code-root "${CODE_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --models-root "${MODELS_ROOT}" \
  --hparams "${hparams_candidates[0]}" \
  --manifest-json "${CACHE_RUN}/manifest.json" \
  --cache-root "${FINAL_CACHE_RUN}/samples" \
  --scf-selector 6 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --num-shards 16 \
  --num-workers 4 \
  --output-summary-json "${CACHE_RUN}/summaries/shard_${SLURM_ARRAY_TASK_ID}.json" \
  > "${CACHE_RUN}/logs/shard_${SLURM_ARRAY_TASK_ID}.log" 2>&1
