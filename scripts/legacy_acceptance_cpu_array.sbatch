#!/usr/bin/env bash
#SBATCH --job-name=legacy-accept-cpu
#SBATCH --partition=compute
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=04:00:00

set -euo pipefail

readonly CODE_ROOT=/scratch/xzh/code/structures25
readonly DATA_ROOT=/scratch/xzh/dft/data
readonly MODELS_ROOT=/scratch/xzh/dft/models
readonly RESTORE_ROOT=/scratch/xzh/dft/restore
readonly SAFE_ROOT=${RESTORE_ROOT}/group_safe_splits_20260715
readonly OUTPUT_ROOT=${RESTORE_ROOT}/model_acceptance_20260715

case ${SLURM_ARRAY_TASK_ID} in
  0) dataset_name=QM9_perturbed_fock; model_pattern='088__*' ;;
  1) dataset_name=QMUGSBin0_perturbed_fock; model_pattern='214__*' ;;
  *) exit 65 ;;
esac
hparams_candidates=("${MODELS_ROOT}"/train/runs/${model_pattern}/hparams.yaml)
if [[ ${#hparams_candidates[@]} -ne 1 || ! -f ${hparams_candidates[0]} ]]; then exit 66; fi
hparams=${hparams_candidates[0]}
run_name=${dataset_name}__cpu_float64
mkdir -p "${OUTPUT_ROOT}"

export PROJECT_ROOT=${CODE_ROOT}
export DFT_DATA=${DATA_ROOT}
export DFT_MODELS=${MODELS_ROOT}
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

exec /scratch/xzh/envs/structures25/bin/python \
  "${RESTORE_ROOT}/legacy_model_acceptance.py" \
  --code-root "${CODE_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --models-root "${MODELS_ROOT}" \
  --hparams "${hparams}" \
  --safe-split "${SAFE_ROOT}/${dataset_name}/split.pkl" \
  --selection-json "${RESTORE_ROOT}/acceptance_selection_20260715.json" \
  --device cpu \
  --dtype float64 \
  --threads 12 \
  --output-json-gz "${OUTPUT_ROOT}/${run_name}.json.gz" \
  --output-summary-json "${OUTPUT_ROOT}/${run_name}.summary.json" \
  > "${OUTPUT_ROOT}/${run_name}.log" 2>&1
