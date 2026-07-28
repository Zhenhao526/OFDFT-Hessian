#!/usr/bin/env bash
#SBATCH --job-name=legacy-accept-gpu
#SBATCH --partition=compute
#SBATCH --array=0-3%4
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00

set -euo pipefail

readonly CODE_ROOT=/scratch/xzh/code/structures25
readonly DATA_ROOT=/scratch/xzh/dft/data
readonly MODELS_ROOT=/scratch/xzh/dft/models
readonly RESTORE_ROOT=/scratch/xzh/dft/restore
readonly SAFE_ROOT=${RESTORE_ROOT}/group_safe_splits_20260715
readonly OUTPUT_ROOT=${RESTORE_ROOT}/model_acceptance_20260715

model_index=$((SLURM_ARRAY_TASK_ID / 2))
dtype_index=$((SLURM_ARRAY_TASK_ID % 2))
case ${model_index} in
  0) dataset_name=QM9_perturbed_fock; model_pattern='088__*' ;;
  1) dataset_name=QMUGSBin0_perturbed_fock; model_pattern='214__*' ;;
  *) exit 65 ;;
esac
case ${dtype_index} in
  0) dtype=float32 ;;
  1) dtype=float64 ;;
  *) exit 65 ;;
esac
hparams_candidates=("${MODELS_ROOT}"/train/runs/${model_pattern}/hparams.yaml)
if [[ ${#hparams_candidates[@]} -ne 1 || ! -f ${hparams_candidates[0]} ]]; then exit 66; fi
hparams=${hparams_candidates[0]}
run_name=${dataset_name}__gpu_${dtype}
mkdir -p "${OUTPUT_ROOT}"

export PROJECT_ROOT=${CODE_ROOT}
export DFT_DATA=${DATA_ROOT}
export DFT_MODELS=${MODELS_ROOT}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

exec /scratch/xzh/envs/structures25/bin/python \
  "${RESTORE_ROOT}/legacy_model_acceptance.py" \
  --code-root "${CODE_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --models-root "${MODELS_ROOT}" \
  --hparams "${hparams}" \
  --safe-split "${SAFE_ROOT}/${dataset_name}/split.pkl" \
  --selection-json "${RESTORE_ROOT}/acceptance_selection_20260715.json" \
  --device cuda:0 \
  --dtype "${dtype}" \
  --threads 8 \
  --output-json-gz "${OUTPUT_ROOT}/${run_name}.json.gz" \
  --output-summary-json "${OUTPUT_ROOT}/${run_name}.summary.json" \
  > "${OUTPUT_ROOT}/${run_name}.log" 2>&1
