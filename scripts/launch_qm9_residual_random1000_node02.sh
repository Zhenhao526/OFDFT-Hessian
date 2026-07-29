#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

NODE_ROOT="${NODE_ROOT:-/home/shenwei01/xzh_node02_20260724}"
source "${ROOT_DIR}/scripts/activate_qm9_node02_local.sh"

LABEL_NUM_THREADS="${LABEL_NUM_THREADS:-1}"
export DFT_DATA="${TRAIN_DFT_DATA:-${NODE_ROOT}/data}"
export DFT_MODELS="${TRAIN_DFT_MODELS:-${NODE_ROOT}/models}"
export OMP_NUM_THREADS="${LABEL_NUM_THREADS}"
export MKL_NUM_THREADS="${LABEL_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${LABEL_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${LABEL_NUM_THREADS}"
export CUDA_VISIBLE_DEVICES="${RESIDUAL_GPU:-7}"
export TMPDIR="${TMPDIR:-${NODE_ROOT}/tmp/qm9_residual_random1000_v1}"

TRAIN_NAME="QM9PBEForceRandom1000Train800RebuildV1"
VAL_NAME="QM9PBEForceRandom1000Val100RebuildV1"
TEST_NAME="QM9PBEForceRandom1000Test100RebuildV1"
TRAIN_ROOT="${DFT_DATA}/${TRAIN_NAME}"
VAL_ROOT="${DFT_DATA}/${VAL_NAME}"
TEST_ROOT="${DFT_DATA}/${TEST_NAME}"
FULL_QM9_RAW_DIR="${FULL_QM9_RAW_DIR:-${DFT_DATA}/QM9/raw}"
QM9_ARCHIVE="${QM9_ARCHIVE:-${NODE_ROOT}/data/QM9Source/dsgdb9nsd.xyz.tar.bz2}"
VAL_RAW_DIR="${NODE_ROOT}/data/QM9Val100FrozenRawV1/raw"
TEST_RAW_DIR="${NODE_ROOT}/data/QM9Test100FrozenRawV1/raw"
RUN_ROOT="${NODE_ROOT}/runs/qm9_residual_random1000_v1"
SELECTION_DIR="${RUN_ROOT}/selection"
VAL_LABEL_RUN="${RUN_ROOT}/val100_labelgen"
TEST_LABEL_RUN="${RUN_ROOT}/test100_labelgen"
DIRECT_RUN="${DFT_MODELS}/train/runs/qm9_random1000_direct_kin_plus_xc_eg_rebuild_v1"
RESIDUAL_RUN="${DFT_MODELS}/train/runs/qm9_random1000_residual_kin_minus_apbe_eg_rebuild_v1"
STAGE="${STAGE:-all}"
LABEL_NUM_PROCESSES="${LABEL_NUM_PROCESSES:-20}"
export QM9_RESIDUAL1000_SPLIT="${SELECTION_DIR}/split_train_val.pkl"

mkdir -p \
  "${RUN_ROOT}" \
  "${SELECTION_DIR}" \
  "${VAL_LABEL_RUN}/logs" \
  "${TEST_LABEL_RUN}/logs" \
  "${TMPDIR}"

run_timed() {
  local name="$1"
  shift
  /usr/bin/time -v -o "${RUN_ROOT}/${name}.time.txt" \
    "$@" > "${RUN_ROOT}/${name}.log" 2>&1
}

check_gpu_free() {
  if [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
    echo "ERROR: this protocol uses exactly one physical GPU" >&2
    exit 2
  fi
  local active_gpu_pids
  active_gpu_pids="$(
    nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'
  )"
  if [[ -n "${active_gpu_pids}" ]]; then
    echo "ERROR: GPU ${CUDA_VISIBLE_DEVICES} is busy (PIDs: ${active_gpu_pids//$'\n'/,})" >&2
    exit 2
  fi
}

prepare_partition() {
  local partition="$1"
  local dataset_name="$2"
  local dataset_root="$3"
  local raw_dir="$4"
  local label_run="$5"
  local data_config="$6"
  local step_prefix="$7"

  local source_args=()
  if [[ "$(find "${FULL_QM9_RAW_DIR}" -maxdepth 1 -type f -name 'dsgdb9nsd_*.xyz' \
    2>/dev/null | wc -l)" -ge 1000 ]]; then
    source_args=(--full-qm9-raw-dir "${FULL_QM9_RAW_DIR}")
  else
    if [[ ! -f "${QM9_ARCHIVE}" ]]; then
      mkdir -p "$(dirname "${QM9_ARCHIVE}")"
      curl -L --fail \
        https://ndownloader.figshare.com/files/3195389 \
        -o "${QM9_ARCHIVE}.partial"
      mv "${QM9_ARCHIVE}.partial" "${QM9_ARCHIVE}"
    fi
    source_args=(--qm9-archive "${QM9_ARCHIVE}")
  fi

  run_timed "${step_prefix}_prepare_${partition}100_raw" \
    "${PYTHON_BIN}" scripts/prepare_qm9_residual_held200_raw.py \
    "${source_args[@]}" \
    --output-raw-dir "${raw_dir}" \
    --manifest "${SELECTION_DIR}/${partition}100_raw_manifest.json" \
    --partition "${partition}"

  local common_overrides=(
    "preset=qm9_pbe_force_full"
    "dataset.name=${dataset_name}"
    "dataset.filename=qm9_pbe_force_random1000_${partition}100_rebuild_v1"
    "dataset.raw_data_dir=${raw_dir}"
    "start_idx=0"
    "n_molecules=100"
    "num_processes=${LABEL_NUM_PROCESSES}"
    "num_threads_per_process=${LABEL_NUM_THREADS}"
    "max_memory_per_process=5000"
    "verify_files=true"
    "remove_broken_files=false"
  )

  if [[ "$(find "${dataset_root}/kohn_sham" -maxdepth 1 -type f -name '*.chk' \
    2>/dev/null | wc -l)" != "400" ]]; then
    run_timed "${step_prefix}_01_${partition}100_kohn_sham" \
      "${PYTHON_BIN}" -m mldft.datagen.kohn_sham_dataset \
      "${common_overrides[@]}" \
      "hydra.run.dir=${label_run}/hydra/kohn_sham"
  fi

  if [[ "$(find "${dataset_root}/labels" -maxdepth 1 -type f -name '*.zarr.zip' \
    2>/dev/null | wc -l)" != "400" ]]; then
    run_timed "${step_prefix}_02_${partition}100_labelgen" \
      "${PYTHON_BIN}" -m mldft.datagen.generate_labels_dataset \
      "${common_overrides[@]}" \
      "hydra.run.dir=${label_run}/hydra/labelgen"
  fi

  run_timed "${step_prefix}_03_${partition}100_force_check" \
    "${PYTHON_BIN}" scripts/check_qm9_force_smoke.py \
    "${dataset_root}/labels" \
    --expected-molecules 100 \
    --expected-samples 4 \
    --summary-json "${label_run}/force_check_summary.json"

  if [[ "$(find "${dataset_root}/labels_local_frames_global_symmetric_natrep" \
    -maxdepth 1 -type f -name '*.zarr.zip' 2>/dev/null | wc -l)" != "400" ]]; then
    run_timed "${step_prefix}_04_${partition}100_transform" \
      "${PYTHON_BIN}" -m mldft.datagen.transform_dataset \
      "data=${data_config}" \
      "extras.enforce_tags=false" \
      "extras.print_config=false" \
      "++hydra.callbacks.git_logging.clean=false" \
      "+num_processes=${LABEL_NUM_PROCESSES}" \
      "+num_threads_per_process=${LABEL_NUM_THREADS}" \
      "+start_idx=0" \
      "+num_molecules=999999" \
      "hydra.run.dir=${label_run}/hydra/transform"
  fi
}

prepare_validation() {
  prepare_partition \
    val \
    "${VAL_NAME}" \
    "${VAL_ROOT}" \
    "${VAL_RAW_DIR}" \
    "${VAL_LABEL_RUN}" \
    qm9_pbe_force_random1000_val100_rebuild_v1 \
    00
  run_timed "05_build_blind_safe_split" \
    "${PYTHON_BIN}" scripts/prepare_qm9_residual_random1000_split.py \
    --train-root "${TRAIN_ROOT}" \
    --val-root "${VAL_ROOT}" \
    --output-dir "${SELECTION_DIR}"

  local residual_statistics
  residual_statistics="${TRAIN_ROOT}/dataset_statistics/"
  residual_statistics+="dataset_statistics_labels_local_frames_global_symmetric_natrep_"
  residual_statistics+="e_kin_minus_apbe.zarr"
  if [[ ! -f "${residual_statistics}/.zgroup" ]]; then
    run_timed "06_train800_residual_statistics" \
      "${PYTHON_BIN}" -m mldft.ml.compute_dataset_statistics \
      "data=qm9_pbe_force_random1000_train800_rebuild_v1" \
      "data.target_key=kin_minus_apbe" \
      "name=${TRAIN_NAME}_kin_minus_apbe_statistics" \
      "extras.enforce_tags=false" \
      "extras.print_config=false" \
      "++hydra.callbacks.git_logging.clean=false" \
      "overwrite=false" \
      "data.datamodule.batch_size=16" \
      "data.datamodule.num_workers=4" \
      "statistic_fitter_kwargs.n_batches=null" \
      "hydra.run.dir=${VAL_LABEL_RUN}/hydra/residual_statistics"
  fi
}

prepare_test_after_freeze() {
  if [[ ! -f "${RUN_ROOT}/protocol_frozen.json" ]]; then
    echo "ERROR: refusing Test100 access before ${RUN_ROOT}/protocol_frozen.json exists" >&2
    exit 2
  fi
  prepare_partition \
    test \
    "${TEST_NAME}" \
    "${TEST_ROOT}" \
    "${TEST_RAW_DIR}" \
    "${TEST_LABEL_RUN}" \
    qm9_pbe_force_random1000_test100_rebuild_v1 \
    90
  run_timed "95_build_frozen_full_split" \
    "${PYTHON_BIN}" scripts/prepare_qm9_residual_random1000_split.py \
    --train-root "${TRAIN_ROOT}" \
    --val-root "${VAL_ROOT}" \
    --test-root "${TEST_ROOT}" \
    --output-dir "${SELECTION_DIR}" \
    --unlock-test
}

train_one() {
  local experiment="$1"
  local run_name="$2"
  local run_dir="$3"
  if [[ -e "${run_dir}/checkpoints/last.ckpt" ]]; then
    echo "checkpoint exists, skipping ${run_name}"
    return
  fi
  mkdir -p "${run_dir}"
  /usr/bin/time -v -o "${run_dir}/train.time.txt" \
    "${PYTHON_BIN}" -m mldft.ml.train \
    "experiment=str25/${experiment}" \
    "name=${run_name}" \
    "hydra.run.dir=${run_dir}" \
    "extras.enforce_tags=false" \
    "extras.print_config=false" \
    "hydra.callbacks.git_logging.clean=false" \
    > "${run_dir}/train.log" 2>&1
}

train_models() {
  if [[ ! -f "${QM9_RESIDUAL1000_SPLIT}" ]]; then
    echo "ERROR: missing blind-safe split: ${QM9_RESIDUAL1000_SPLIT}" >&2
    exit 2
  fi
  check_gpu_free

  local smoke_root="${RUN_ROOT}/smoke"
  mkdir -p "${smoke_root}"
  for experiment in \
    qm9_random1000_direct_kin_plus_xc_eg_rebuild_v1 \
    qm9_random1000_residual_kin_minus_apbe_eg_rebuild_v1
  do
    local smoke_dir="${smoke_root}/${experiment}"
    if [[ ! -e "${smoke_dir}/checkpoints/last.ckpt" ]]; then
      mkdir -p "${smoke_dir}"
      "${PYTHON_BIN}" -m mldft.ml.train \
        "experiment=str25/${experiment}" \
        "name=${experiment}_smoke" \
        "hydra.run.dir=${smoke_dir}" \
        "trainer.max_epochs=1" \
        "trainer.max_steps=2" \
        "trainer.limit_val_batches=2" \
        "data.datamodule.num_workers=0" \
        "callbacks.model_checkpoint.save_top_k=0" \
        "extras.enforce_tags=false" \
        "extras.print_config=false" \
        "hydra.callbacks.git_logging.clean=false" \
        > "${smoke_dir}/train.log" 2>&1
    fi
  done

  train_one \
    qm9_random1000_direct_kin_plus_xc_eg_rebuild_v1 \
    qm9_random1000_direct_kin_plus_xc_eg_rebuild_v1 \
    "${DIRECT_RUN}"
  train_one \
    qm9_random1000_residual_kin_minus_apbe_eg_rebuild_v1 \
    qm9_random1000_residual_kin_minus_apbe_eg_rebuild_v1 \
    "${RESIDUAL_RUN}"
}

evaluate_split() {
  local split="$1"
  local output_dir="${RUN_ROOT}/${split}_metrics"
  mkdir -p "${output_dir}"
  for suffix in all_scf ground_state_sample0
  do
    local filters=()
    if [[ "${suffix}" == "ground_state_sample0" ]]; then
      filters=(--ground-state-only --base-geometry-only)
    fi
    "${PYTHON_BIN}" scripts/qm9_functional_target_eval.py \
      --run-dir "${RESIDUAL_RUN}" \
      --zero-model \
      --split "${split}" \
      --device "cuda:0" \
      --output-json "${output_dir}/apbek_${suffix}.json" \
      "${filters[@]}"
    "${PYTHON_BIN}" scripts/qm9_functional_target_eval.py \
      --run-dir "${DIRECT_RUN}" \
      --ckpt "${DIRECT_RUN}/checkpoints/last.ckpt" \
      --split "${split}" \
      --device "cuda:0" \
      --output-json "${output_dir}/direct_${suffix}.json" \
      "${filters[@]}"
    "${PYTHON_BIN}" scripts/qm9_functional_target_eval.py \
      --run-dir "${RESIDUAL_RUN}" \
      --ckpt "${RESIDUAL_RUN}/checkpoints/last.ckpt" \
      --split "${split}" \
      --device "cuda:0" \
      --output-json "${output_dir}/residual_${suffix}.json" \
      "${filters[@]}"
  done
}

case "${STAGE}" in
  prepare)
    prepare_validation
    ;;
  train)
    train_models
    ;;
  eval)
    check_gpu_free
    evaluate_split val
    ;;
  freeze)
    "${PYTHON_BIN}" scripts/freeze_qm9_residual_random1000_protocol.py \
      --validation-dir "${RUN_ROOT}/val_metrics" \
      --direct-checkpoint "${DIRECT_RUN}/checkpoints/last.ckpt" \
      --residual-checkpoint "${RESIDUAL_RUN}/checkpoints/last.ckpt" \
      --protocol "${ROOT_DIR}/configs/audit/qm9_residual_random1000_graphformer_v1.yaml" \
      --output "${RUN_ROOT}/protocol_frozen.json"
    ;;
  test)
    prepare_test_after_freeze
    export QM9_RESIDUAL1000_SPLIT="${SELECTION_DIR}/split_full.pkl"
    check_gpu_free
    evaluate_split test
    ;;
  all)
    prepare_validation
    train_models
    evaluate_split val
    ;;
  *)
    echo "ERROR: STAGE must be prepare, train, eval, freeze, test, or all" >&2
    exit 2
    ;;
esac
