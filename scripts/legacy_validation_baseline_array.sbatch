#!/usr/bin/env bash
#SBATCH --job-name=legacy-val-baseline
#SBATCH --partition=compute
#SBATCH --array=0-31%8
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
readonly OUTPUT_ROOT=${RESTORE_ROOT}/scientific_baseline_validation_full_20260715
readonly CACHE_ROOT=${RESTORE_ROOT}/baseline_cache_validation_20260715/samples
readonly NUM_SHARDS=8

if [[ ${DATA_ROOT} == /export/scratch/ialgroup* || ${MODELS_ROOT} == /export/scratch/ialgroup* ]]; then
  echo "Refusing forbidden legacy effective path" >&2
  exit 64
fi
if ! cmp -s "${DATA_ROOT}/QM9_perturbed_fock/dataset_info.yaml" \
  "${DATA_ROOT}/QMUGSBin0_perturbed_fock/dataset_info.yaml"; then
  echo "QM9/QMUGS basis metadata differ; refusing common natural-representation cache" >&2
  exit 68
fi
export PROJECT_ROOT=${CODE_ROOT}
export DFT_DATA=${DATA_ROOT}
export DFT_MODELS=${MODELS_ROOT}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

run_index=$((SLURM_ARRAY_TASK_ID / NUM_SHARDS))
shard_index=$((SLURM_ARRAY_TASK_ID % NUM_SHARDS))
case ${run_index} in
  0)
    run_name=qm9_model__qm9_domain
    model_pattern='088__*'
    dataset_name=QM9_perturbed_fock
    ;;
  1)
    run_name=qmugs_model__qmugs_domain
    model_pattern='214__*'
    dataset_name=QMUGSBin0_perturbed_fock
    ;;
  2)
    run_name=qm9_model__combined_domain
    model_pattern='088__*'
    dataset_name=QMUGSBin0QM9_perturbed_fock
    ;;
  3)
    run_name=qmugs_model__combined_domain
    model_pattern='214__*'
    dataset_name=QMUGSBin0QM9_perturbed_fock
    ;;
  *) echo "Unexpected run index ${run_index}" >&2; exit 65 ;;
esac

run_root=${OUTPUT_ROOT}/${run_name}
mkdir -p "${run_root}/rows" "${run_root}/summaries" "${run_root}/task_logs"
hparams_candidates=("${MODELS_ROOT}"/train/runs/${model_pattern}/hparams.yaml)
if [[ ${#hparams_candidates[@]} -ne 1 || ! -f ${hparams_candidates[0]} ]]; then
  echo "Expected exactly one hparams match for ${model_pattern}" >&2
  exit 66
fi
hparams=${hparams_candidates[0]}

exec /scratch/xzh/envs/structures25/bin/python \
  "${RESTORE_ROOT}/legacy_scientific_baseline_shard.py" \
  --code-root "${CODE_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --models-root "${MODELS_ROOT}" \
  --hparams "${hparams}" \
  --dataset-name "${dataset_name}" \
  --split-file "${SAFE_ROOT}/${dataset_name}/split.pkl" \
  --partition val \
  --shard-index "${shard_index}" \
  --num-shards "${NUM_SHARDS}" \
  --device cuda:0 \
  --dtype float32 \
  --batch-size 64 \
  --num-workers 4 \
  --scf-iterations=6,-1 \
  --preprocessed-cache-root "${CACHE_ROOT}" \
  --output-jsonl-gz "${run_root}/rows/shard_${shard_index}.jsonl.gz" \
  --output-summary-json "${run_root}/summaries/shard_${shard_index}.json" \
  --progress-interval 100 \
  > "${run_root}/task_logs/shard_${shard_index}.log" 2>&1
