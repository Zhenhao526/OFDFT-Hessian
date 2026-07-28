#!/usr/bin/env bash
#SBATCH --job-name=legacy-cache-gpu-smoke
#SBATCH --partition=compute
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00

set -euo pipefail
readonly CODE_ROOT=/scratch/xzh/code/structures25
readonly DATA_ROOT=/scratch/xzh/dft/data
readonly MODELS_ROOT=/scratch/xzh/dft/models
readonly RESTORE_ROOT=/scratch/xzh/dft/restore
readonly RUN_ROOT=${RESTORE_ROOT}/baseline_cache_test_20260715/gpu_smoke
export PROJECT_ROOT=${CODE_ROOT}
export DFT_DATA=${DATA_ROOT}
export DFT_MODELS=${MODELS_ROOT}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p "${RUN_ROOT}"
exec /scratch/xzh/envs/structures25/bin/python \
  "${RESTORE_ROOT}/legacy_prepare_baseline_cache_shard.py" \
  --code-root "${CODE_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --models-root "${MODELS_ROOT}" \
  --hparams "${MODELS_ROOT}/train/runs/088__from_checkpoint_009__str25\\qm9_tf/hparams.yaml" \
  --manifest-json "${RUN_ROOT}/manifest.json" \
  --cache-root "${RUN_ROOT}/samples" \
  --shard-index 0 \
  --num-shards 1 \
  --num-workers 0 \
  --transform-device cuda:0 \
  --output-summary-json "${RUN_ROOT}/summary.json" \
  --progress-interval 1 \
  > "${RUN_ROOT}/run.log" 2>&1
