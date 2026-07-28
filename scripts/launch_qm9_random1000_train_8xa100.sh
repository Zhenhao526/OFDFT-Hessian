#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f "${ENV_FILE:-/scratch/xzh/env.sh}" ]]; then
  # shellcheck source=/dev/null
  source "${ENV_FILE:-/scratch/xzh/env.sh}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN="${TORCHRUN:-$(command -v torchrun)}"

export DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
export DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"
export DATASET_NAME="${DATASET_NAME:-QM9PBEForceRandom1000}"
export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

EPOCHS="${EPOCHS:-10}"
if [[ "${EPOCHS}" != "10" && "${EPOCHS}" != "20" ]]; then
  echo "ERROR: EPOCHS must be 10 or 20 for scripts/launch_qm9_full_scale_8xa100.sh, got ${EPOCHS}" >&2
  exit 2
fi

EXPECTED_LABELS="${EXPECTED_LABELS:-4000}"
SPLIT_PROCESSES="${SPLIT_PROCESSES:-64}"
TRANSFORM_NUM_PROCESSES="${TRANSFORM_NUM_PROCESSES:-64}"
TRANSFORM_NUM_THREADS="${TRANSFORM_NUM_THREADS:-1}"
STAT_BATCH_SIZE="${STAT_BATCH_SIZE:-16}"
STAT_NUM_WORKERS="${STAT_NUM_WORKERS:-4}"
EG_BATCH_SIZE="${EG_BATCH_SIZE:-8}"
EGF_BATCH_SIZE="${EGF_BATCH_SIZE:-4}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-8}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${DFT_MODELS}/train_random1000/${RUN_STAMP}}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
TMPDIR="${TMPDIR:-/scratch/xzh/tmp/qm9_random1000_train_${RUN_STAMP}}"
mkdir -p "${LOG_DIR}" "${TMPDIR}" "${DFT_MODELS}" "${DFT_DATA}"
export TMPDIR

DATASET_DIR="${DFT_DATA}/${DATASET_NAME}"
LABEL_DIR="${DATASET_DIR}/labels"
TRANSFORM_NAME="${TRANSFORM_NAME:-local_frames_global_symmetric_natrep}"
TRANSFORMED_LABEL_DIR="${DATASET_DIR}/labels_${TRANSFORM_NAME}"
STAT_PATH="${DATASET_DIR}/dataset_statistics/dataset_statistics_labels_${TRANSFORM_NAME}_e_kin_plus_xc.zarr"
SPLIT_FILE="${DATASET_DIR}/split.pkl"

run_timed() {
  local name="$1"
  shift
  echo
  echo "===== ${name} ====="
  echo "command: $*"
  /usr/bin/time -v -o "${LOG_DIR}/${name}.time.txt" "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
}

check_gpus_free() {
  local label="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return
  fi
  local status_file="${LOG_DIR}/gpu_status_${label}.csv"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | tee "${status_file}"
  if [[ "${REQUIRE_FREE_GPUS:-1}" == "1" ]]; then
    local busy_gpus
    busy_gpus="$(awk -F, '{gsub(/ /, "", $4); if ($4 + 0 > 512) busy += 1} END {print busy + 0}' "${status_file}")"
    if [[ "${busy_gpus}" != "0" ]]; then
      echo "ERROR: ${busy_gpus} GPUs have more than 512 MiB allocated at ${label}." >&2
      exit 1
    fi
  fi
}

count_labels() {
  find "${LABEL_DIR}" -maxdepth 1 -type f -name "*.zarr.zip" | wc -l
}

count_transformed_labels() {
  if [[ -d "${TRANSFORMED_LABEL_DIR}" ]]; then
    find "${TRANSFORMED_LABEL_DIR}" -maxdepth 1 -type f -name "*.zarr.zip" | wc -l
  else
    echo 0
  fi
}

train_sample_count_for_throughput() {
  "${PYTHON_BIN}" - <<PY
import pickle
from pathlib import Path

split_file = Path("${SPLIT_FILE}")
if split_file.exists():
    with split_file.open("rb") as f:
        split = pickle.load(f)
    print(int(split["sizes"]["train"]))
else:
    print(int("${LABEL_COUNT:-0}"))
PY
}

write_estimate() {
  local label_count="$1"
  "${PYTHON_BIN}" - <<PY | tee "${LOG_DIR}/estimate.txt"
import math
import pickle
from pathlib import Path

labels = int("${label_count}")
epochs = int("${EPOCHS}")
eg_batch = int("${EG_BATCH_SIZE}") * 8 * int("${ACCUMULATE_GRAD_BATCHES}")
egf_batch = int("${EGF_BATCH_SIZE}") * 8 * int("${ACCUMULATE_GRAD_BATCHES}")
split_path = Path("${SPLIT_FILE}")
if split_path.exists():
    with split_path.open("rb") as f:
        split = pickle.load(f)
    train_samples = int(split["sizes"]["train"])
    val_samples = int(split["sizes"]["val"])
    test_samples = int(split["sizes"]["test"])
    split_source = str(split_path)
else:
    train_samples = int(round(labels * 0.8))
    val_samples = labels // 10
    test_samples = labels // 10
    split_source = "approximate label-file estimate before split creation"
eg_steps_epoch = math.floor(train_samples / eg_batch)
egf_steps_epoch = math.floor(train_samples / egf_batch)

print("QM9PBEForceRandom1000 8xA100 EG/EGF training estimate")
print(f"dataset: ${DATASET_NAME}")
print(f"labels: {labels}")
print(f"split source: {split_source}")
print(f"samples after SCF-iteration expansion/filter basis: train={train_samples}, val={val_samples}, test={test_samples}")
print(f"EG: per-GPU batch=${EG_BATCH_SIZE}, effective global batch={eg_batch}, ~{eg_steps_epoch} steps/epoch, ~{eg_steps_epoch * epochs} train steps")
print(f"EGF lambda=1.0: per-GPU batch=${EGF_BATCH_SIZE}, effective global batch={egf_batch}, ~{egf_steps_epoch} steps/epoch, ~{egf_steps_epoch * epochs} train steps")
print("Rough wall-time budget on one 8xA100 node:")
print("  split/cache/statistics: 0.5-2 h")
print("  EG 10 epoch:           1-3 h")
print("  EGF 10 epoch:          2-5 h")
print("  end-to-end expected:   4-10 h")
print("  Slurm wall time:       12 h")
PY
}

echo "host: $(hostname)"
echo "date: $(date)"
echo "root: ${ROOT_DIR}"
echo "DFT_DATA=${DFT_DATA}"
echo "DFT_MODELS=${DFT_MODELS}"
echo "DATASET_NAME=${DATASET_NAME}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "TMPDIR=${TMPDIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "TORCHRUN=${TORCHRUN}"

if [[ ! -d "${LABEL_DIR}" ]]; then
  echo "ERROR: label directory not found: ${LABEL_DIR}" >&2
  exit 1
fi

check_gpus_free "before_prepare"

LABEL_COUNT="$(count_labels)"
echo "label_count=${LABEL_COUNT}"
if [[ "${LABEL_COUNT}" != "${EXPECTED_LABELS}" ]]; then
  echo "ERROR: expected ${EXPECTED_LABELS} labels, found ${LABEL_COUNT}" >&2
  exit 1
fi

write_estimate "${LABEL_COUNT}"

if [[ -f "${SPLIT_FILE}" ]]; then
  echo "split exists, skipping: ${SPLIT_FILE}"
else
  run_timed "01_grouped_split" \
    "${PYTHON_BIN}" mldft/utils/create_dataset_splits.py \
    "${DATASET_NAME}" \
    --group-by-molecule \
    -p "${SPLIT_PROCESSES}"
fi

TRANSFORMED_COUNT="$(count_transformed_labels)"
if [[ "${TRANSFORMED_COUNT}" == "${LABEL_COUNT}" ]]; then
  echo "cached transformed labels complete, skipping transform: ${TRANSFORMED_LABEL_DIR}"
else
  echo "transformed_count=${TRANSFORMED_COUNT}; expected=${LABEL_COUNT}"
  run_timed "02_transform_cached_labels" \
    "${PYTHON_BIN}" -m mldft.datagen.transform_dataset \
    "data=qm9_pbe_force_full" \
    "data.dataset_name=${DATASET_NAME}" \
    "extras.enforce_tags=false" \
    "extras.print_config=false" \
    "hydra.callbacks.git_logging.clean=false" \
    "+num_processes=${TRANSFORM_NUM_PROCESSES}" \
    "+num_threads_per_process=${TRANSFORM_NUM_THREADS}" \
    "+start_idx=0" \
    "+num_molecules=999999" \
    "hydra.run.dir=${RUN_ROOT}/hydra/transform"
fi

if [[ -d "${STAT_PATH}" ]]; then
  echo "dataset statistics exist, skipping: ${STAT_PATH}"
else
  run_timed "03_statistics" \
    "${PYTHON_BIN}" -m mldft.ml.compute_dataset_statistics \
    "data=qm9_pbe_force_full" \
    "data.dataset_name=${DATASET_NAME}" \
    "name=${DATASET_NAME}_statistics" \
    "extras.enforce_tags=false" \
    "extras.print_config=false" \
    "hydra.callbacks.git_logging.clean=false" \
    "overwrite=false" \
    "data.datamodule.batch_size=${STAT_BATCH_SIZE}" \
    "data.datamodule.num_workers=${STAT_NUM_WORKERS}" \
    "statistic_fitter_kwargs.n_batches=null" \
    "hydra.run.dir=${RUN_ROOT}/hydra/statistics"
fi

export NPROC_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export TORCHRUN
export NUM_WORKERS="${TRAIN_NUM_WORKERS}"
export ACCUMULATE_GRAD_BATCHES

EG_RUN_NAME="${EG_RUN_NAME:-qm9_random1000_eg_e${EPOCHS}_${RUN_STAMP}}"
EGF_RUN_NAME="${EGF_RUN_NAME:-qm9_random1000_egf_lam1_e${EPOCHS}_${RUN_STAMP}}"
THROUGHPUT_ESTIMATE_NUM_SAMPLES="${THROUGHPUT_ESTIMATE_NUM_SAMPLES:-$(train_sample_count_for_throughput)}"

COMMON_TRAIN_OVERRIDES=(
  "data.dataset_name=${DATASET_NAME}"
  "extras.enforce_tags=false"
  "extras.print_config=false"
  "hydra.callbacks.git_logging.clean=false"
  "callbacks.throughput_monitor.estimate_num_samples=${THROUGHPUT_ESTIMATE_NUM_SAMPLES}"
  "callbacks.throughput_monitor.target_num_devices=8"
)

check_gpus_free "before_train_eg"
export MASTER_PORT="${MASTER_PORT_EG:-29500}"
run_timed "04_train_eg_e${EPOCHS}" \
  env PER_GPU_BATCH_SIZE="${EG_BATCH_SIZE}" \
  bash scripts/launch_qm9_full_scale_8xa100.sh \
  eg "${EPOCHS}" "${EG_RUN_NAME}" \
  "${COMMON_TRAIN_OVERRIDES[@]}" \
  "tags=[qm9_random1000,eg,fixed_density,8xa100]"

check_gpus_free "between_train_eg_egf"
export MASTER_PORT="${MASTER_PORT_EGF:-29501}"
run_timed "05_train_egf_lam1_e${EPOCHS}" \
  env PER_GPU_BATCH_SIZE="${EGF_BATCH_SIZE}" \
  bash scripts/launch_qm9_full_scale_8xa100.sh \
  egf "${EPOCHS}" "${EGF_RUN_NAME}" \
  "${COMMON_TRAIN_OVERRIDES[@]}" \
  "tags=[qm9_random1000,egf,lambda1,fixed_density,force,8xa100]"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | tee "${LOG_DIR}/gpu_status_after.csv"
fi

cat > "${RUN_ROOT}/summary.txt" <<EOF
QM9PBEForceRandom1000 8xA100 EG/EGF training complete
dataset=${DATASET_NAME}
data_dir=${DATASET_DIR}
run_root=${RUN_ROOT}
epochs=${EPOCHS}
eg_run=${DFT_MODELS}/train/runs/${EG_RUN_NAME}
egf_run=${DFT_MODELS}/train/runs/${EGF_RUN_NAME}
EOF

cat "${RUN_ROOT}/summary.txt"
