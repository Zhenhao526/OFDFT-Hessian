#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

NODE_ROOT="${NODE_ROOT:-/home/shenwei01/xzh_node02_20260724}"
source "${ROOT_DIR}/scripts/activate_qm9_node02_local.sh"

export DFT_DATA="${TRAIN_DFT_DATA:-${NODE_ROOT}/data}"
export DFT_MODELS="${TRAIN_DFT_MODELS:-${NODE_ROOT}/models}"
export OMP_NUM_THREADS="${LABEL_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${LABEL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${LABEL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${LABEL_NUM_THREADS:-1}"
export TMPDIR="${TMPDIR:-${NODE_ROOT}/tmp/qm9_train800_fresh_labels_v1}"

DATASET_NAME="QM9PBEForceRandom1000Train800RebuildV1"
DATASET_FILENAME="qm9_pbe_force_train800_rebuild_v1"
DATASET_ROOT="${DFT_DATA}/${DATASET_NAME}"
RAW_DIR="${NODE_ROOT}/data/QM9Train800FrozenRawV1/raw"
SOURCE_CSV="${NODE_ROOT}/artifacts/graphformer_hybrid_relaxed_hvp_rebuild_v1/data/train800_replay_labels.source.csv"
RUN_DIR="${NODE_ROOT}/runs/graphformer_hybrid_relaxed_hvp_rebuild_v1/labelgen"
NUM_PROCESSES="${LABEL_NUM_PROCESSES:-20}"
NUM_THREADS="${LABEL_NUM_THREADS:-1}"

mkdir -p "${DATASET_ROOT}" "${RUN_DIR}/logs" "${TMPDIR}"
if [[ "$(sha256sum "${SOURCE_CSV}" | awk '{print $1}')" != \
  "882a0f1da1ee74575396731156cb4b6c191b456fcddca8681db48910677f90e9" ]]; then
  echo "ERROR: frozen train800 source CSV hash mismatch" >&2
  exit 2
fi

run_timed() {
  local name="$1"
  shift
  if [[ -e "${RUN_DIR}/logs/${name}.log" || -e "${RUN_DIR}/logs/${name}.time.txt" ]]; then
    local archived_at
    archived_at="$(date +%Y%m%dT%H%M%S)"
    [[ ! -e "${RUN_DIR}/logs/${name}.log" ]] || \
      mv "${RUN_DIR}/logs/${name}.log" \
        "${RUN_DIR}/logs/${name}.attempt_${archived_at}.log"
    [[ ! -e "${RUN_DIR}/logs/${name}.time.txt" ]] || \
      mv "${RUN_DIR}/logs/${name}.time.txt" \
        "${RUN_DIR}/logs/${name}.attempt_${archived_at}.time.txt"
  fi
  /usr/bin/time -v -o "${RUN_DIR}/logs/${name}.time.txt" \
    "$@" > "${RUN_DIR}/logs/${name}.log" 2>&1
}

run_timed "00_raw_manifest" \
  "${PYTHON_BIN}" scripts/prepare_qm9_train800_rebuild_assets.py \
  raw-manifest \
  --source-csv "${SOURCE_CSV}" \
  --raw-dir "${RAW_DIR}" \
  --output "${RUN_DIR}/raw_subset_manifest.json"

COMMON_OVERRIDES=(
  "preset=qm9_pbe_force_full"
  "dataset.name=${DATASET_NAME}"
  "dataset.filename=${DATASET_FILENAME}"
  "dataset.raw_data_dir=${RAW_DIR}"
  "start_idx=0"
  "n_molecules=800"
  "num_processes=${NUM_PROCESSES}"
  "num_threads_per_process=${NUM_THREADS}"
  "max_memory_per_process=5000"
  "verify_files=true"
  "remove_broken_files=false"
)

if [[ "$(find "${DATASET_ROOT}/kohn_sham" -maxdepth 1 -type f -name '*.chk' 2>/dev/null | wc -l)" != "3200" ]]; then
  run_timed "01_kohn_sham" \
    "${PYTHON_BIN}" -m mldft.datagen.kohn_sham_dataset \
    "${COMMON_OVERRIDES[@]}" \
    "hydra.run.dir=${RUN_DIR}/hydra/kohn_sham"
fi

if [[ "$(find "${DATASET_ROOT}/labels" -maxdepth 1 -type f -name '*.zarr.zip' 2>/dev/null | wc -l)" != "3200" ]]; then
  run_timed "02_labelgen" \
    "${PYTHON_BIN}" -m mldft.datagen.generate_labels_dataset \
    "${COMMON_OVERRIDES[@]}" \
    "hydra.run.dir=${RUN_DIR}/hydra/labelgen"
fi

run_timed "03_force_check" \
  "${PYTHON_BIN}" scripts/check_qm9_force_smoke.py \
  "${DATASET_ROOT}/labels" \
  --expected-molecules 800 \
  --expected-samples 4 \
  --summary-json "${RUN_DIR}/force_check_summary.json"

if [[ "$(find "${DATASET_ROOT}/labels_local_frames_global_symmetric_natrep" -maxdepth 1 -type f -name '*.zarr.zip' 2>/dev/null | wc -l)" != "3200" ]]; then
  run_timed "04_transform" \
    "${PYTHON_BIN}" -m mldft.datagen.transform_dataset \
    "data=qm9_pbe_force_random1000_train800_rebuild_v1" \
    "extras.enforce_tags=false" \
    "extras.print_config=false" \
    "++hydra.callbacks.git_logging.clean=false" \
    "+num_processes=${NUM_PROCESSES}" \
    "+num_threads_per_process=${NUM_THREADS}" \
    "+start_idx=0" \
    "+num_molecules=999999" \
    "hydra.run.dir=${RUN_DIR}/hydra/transform"
fi

run_timed "05_write_split" \
  "${PYTHON_BIN}" scripts/prepare_qm9_train800_rebuild_assets.py \
  write-split \
  --source-csv "${SOURCE_CSV}" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-name "${DATASET_NAME}" \
  --label-hash-policy regenerated \
  --expected-usable-training-samples 39456 \
  --sample-budget-seed qm9_graphformer_hybrid_relaxed_hvp_rebuild_v1_scf_budget_v1

STAT_PATH="${DATASET_ROOT}/dataset_statistics/dataset_statistics_labels_local_frames_global_symmetric_natrep_e_kin_plus_xc.zarr"
if [[ ! -f "${STAT_PATH}/.zgroup" ]]; then
  run_timed "06_statistics" \
    "${PYTHON_BIN}" -m mldft.ml.compute_dataset_statistics \
    "data=qm9_pbe_force_random1000_train800_rebuild_v1" \
    "name=${DATASET_NAME}_statistics" \
    "extras.enforce_tags=false" \
    "extras.print_config=false" \
    "++hydra.callbacks.git_logging.clean=false" \
    "overwrite=false" \
    "data.datamodule.batch_size=16" \
    "data.datamodule.num_workers=4" \
    "statistic_fitter_kwargs.n_batches=null" \
    "hydra.run.dir=${RUN_DIR}/hydra/statistics"
fi

run_timed "07_finalize" \
  "${PYTHON_BIN}" scripts/prepare_qm9_train800_rebuild_assets.py \
  finalize \
  --source-csv "${SOURCE_CSV}" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-name "${DATASET_NAME}" \
  --label-hash-policy regenerated \
  --expected-usable-training-samples 39456 \
  --sample-budget-seed qm9_graphformer_hybrid_relaxed_hvp_rebuild_v1_scf_budget_v1

sha256sum \
  "${RUN_DIR}/raw_subset_manifest.json" \
  "${RUN_DIR}/force_check_summary.json" \
  "${DATASET_ROOT}/provenance/train_only_dataset_manifest.json" \
  > "${RUN_DIR}/registered_sha256.txt"
