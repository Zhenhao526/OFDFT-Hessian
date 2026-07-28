#!/usr/bin/env bash
#SBATCH --job-name=legacy-denopt-val
#SBATCH --partition=compute
#SBATCH --array=0-5%6
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00

set -euo pipefail

readonly CODE_ROOT=/scratch/xzh/code/structures25
readonly DATA_ROOT=/scratch/xzh/dft/data
readonly MODELS_ROOT=/scratch/xzh/dft/models
readonly RESTORE_ROOT=/scratch/xzh/dft/restore
readonly OUTPUT_ROOT=${RESTORE_ROOT}/density_optimization_validation_20260715

export PROJECT_ROOT=${CODE_ROOT}
export DFT_DATA=${DATA_ROOT}
export DFT_MODELS=${MODELS_ROOT}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

run_index=$((SLURM_ARRAY_TASK_ID / 3))
selection_index=$((SLURM_ARRAY_TASK_ID % 3))
case ${run_index} in
  0) dataset_name=QM9_perturbed_fock; model_pattern='088__*' ;;
  1) dataset_name=QMUGSBin0_perturbed_fock; model_pattern='214__*' ;;
  *) exit 65 ;;
esac
hparams_candidates=("${MODELS_ROOT}"/train/runs/${model_pattern}/hparams.yaml)
if [[ ${#hparams_candidates[@]} -ne 1 || ! -f ${hparams_candidates[0]} ]]; then exit 66; fi
mkdir -p "${OUTPUT_ROOT}/rows" "${OUTPUT_ROOT}/logs"

exec /scratch/xzh/envs/structures25/bin/python \
  "${RESTORE_ROOT}/legacy_density_optimization_eval.py" \
  --code-root "${CODE_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --models-root "${MODELS_ROOT}" \
  --hparams "${hparams_candidates[0]}" \
  --selection-json "${RESTORE_ROOT}/acceptance_selection_20260715.json" \
  --dataset-name "${dataset_name}" \
  --selection-index "${selection_index}" \
  --device cuda:0 \
  --output-json "${OUTPUT_ROOT}/rows/task_${SLURM_ARRAY_TASK_ID}.json" \
  > "${OUTPUT_ROOT}/logs/task_${SLURM_ARRAY_TASK_ID}.log" 2>&1
