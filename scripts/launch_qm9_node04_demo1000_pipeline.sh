#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f "${ENV_FILE:-/scratch/xzh/env.sh}" ]]; then
  # shellcheck source=/dev/null
  source "${ENV_FILE:-/scratch/xzh/env.sh}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -x ".venv/bin/torchrun" ]]; then
  TORCHRUN="${TORCHRUN:-.venv/bin/torchrun}"
else
  TORCHRUN="${TORCHRUN:-$(command -v torchrun)}"
fi

export DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
export DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

DATASET_NAME="${DATASET_NAME:-QM9PBEForceDemo1000}"
DATASET_FILENAME="${DATASET_FILENAME:-qm9_pbe_force_demo1000}"
START_IDX="${START_IDX:-0}"
N_MOLECULES="${N_MOLECULES:-1000}"
SAMPLES_PER_MOLECULE="${SAMPLES_PER_MOLECULE:-4}"
RAW_DATA_DIR="${RAW_DATA_DIR:-${DFT_DATA}/QM9/raw}"

LABEL_NUM_PROCESSES="${LABEL_NUM_PROCESSES:-20}"
LABEL_NUM_THREADS="${LABEL_NUM_THREADS:-1}"
MAX_MEMORY_PER_PROCESS="${MAX_MEMORY_PER_PROCESS:-4000}"
TRANSFORM_NUM_PROCESSES="${TRANSFORM_NUM_PROCESSES:-24}"
SPLIT_PROCESSES="${SPLIT_PROCESSES:-24}"

DEMO_EPOCHS="${DEMO_EPOCHS:-10}"
EG_BATCH_SIZE="${EG_BATCH_SIZE:-8}"
EGF_BATCH_SIZE="${EGF_BATCH_SIZE:-4}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-4}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${DFT_MODELS}/node04_demo1000/${RUN_STAMP}}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
TMPDIR="${TMPDIR:-/scratch/xzh/tmp/qm9_node04_demo1000_${RUN_STAMP}}"
mkdir -p "${LOG_DIR}" "${TMPDIR}" "${DFT_DATA}" "${DFT_MODELS}"

export TMPDIR
export OMP_NUM_THREADS="${LABEL_NUM_THREADS}"
export MKL_NUM_THREADS="${LABEL_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${LABEL_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${LABEL_NUM_THREADS}"

require_path() {
  local path="$1"
  local message="$2"
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: ${message}: ${path}" >&2
    exit 1
  fi
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
    busy_gpus="$(awk -F, '{gsub(/ /, "", $4); if ($4 + 0 > 256) busy += 1} END {print busy + 0}' "${status_file}")"
    if [[ "${busy_gpus}" != "0" ]]; then
      echo "ERROR: ${busy_gpus} GPUs have more than 256 MiB allocated at ${label}." >&2
      exit 1
    fi
  fi
}

run_timed() {
  local name="$1"
  shift
  echo
  echo "===== ${name} ====="
  echo "command: $*"
  /usr/bin/time -v -o "${LOG_DIR}/${name}.time.txt" "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
}

write_estimate() {
  "${PYTHON_BIN}" - <<PY | tee "${LOG_DIR}/estimate.txt"
n_molecules = int("${N_MOLECULES}")
num_processes = int("${LABEL_NUM_PROCESSES}")
avg_h_per_100_mol_4proc = 5.782314814814815
ideal_label_h = avg_h_per_100_mol_4proc * (n_molecules / 100.0) * (4.0 / num_processes)
print("QM9 node04 demo1000 estimate")
print(f"dataset: ${DATASET_NAME}")
print(f"molecules: {n_molecules}, samples/molecule: ${SAMPLES_PER_MOLECULE}")
print(f"label processes: {num_processes}, per-process memory cap: ${MAX_MEMORY_PER_PROCESS} MB")
print(f"labelgen ideal estimate from P1 batches: {ideal_label_h:.1f} h")
print(f"labelgen conservative 1.5-2.5x budget: {ideal_label_h*1.5:.1f}-{ideal_label_h*2.5:.1f} h")
print("transform/statistics rough budget: 0.5-2 h")
print("8GPU EG+EGF 10 epoch rough budget: 1-4 h")
print(f"recommended Slurm wall time: 48 h")
PY
}

echo "host: $(hostname)"
echo "date: $(date)"
echo "root: ${ROOT_DIR}"
echo "DFT_DATA=${DFT_DATA}"
echo "DFT_MODELS=${DFT_MODELS}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "TMPDIR=${TMPDIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

require_path "${RAW_DATA_DIR}" "QM9 raw xyz directory not found"
check_gpus_free "before_pipeline"

write_estimate

COMMON_DATAGEN_OVERRIDES=(
  "preset=qm9_pbe_force_full"
  "dataset.name=${DATASET_NAME}"
  "dataset.filename=${DATASET_FILENAME}"
  "start_idx=${START_IDX}"
  "n_molecules=${N_MOLECULES}"
  "num_processes=${LABEL_NUM_PROCESSES}"
  "num_threads_per_process=${LABEL_NUM_THREADS}"
  "max_memory_per_process=${MAX_MEMORY_PER_PROCESS}"
  "dataset.raw_data_dir=${RAW_DATA_DIR}"
)

run_timed "01_kohn_sham" \
  "${PYTHON_BIN}" -m mldft.datagen.kohn_sham_dataset \
  "${COMMON_DATAGEN_OVERRIDES[@]}" \
  "hydra.run.dir=${RUN_ROOT}/hydra/kohn_sham"

run_timed "02_labelgen" \
  "${PYTHON_BIN}" -m mldft.datagen.generate_labels_dataset \
  "${COMMON_DATAGEN_OVERRIDES[@]}" \
  "hydra.run.dir=${RUN_ROOT}/hydra/labelgen"

run_timed "03_force_check" \
  "${PYTHON_BIN}" scripts/check_qm9_force_smoke.py \
  "${DFT_DATA}/${DATASET_NAME}/labels" \
  --expected-molecules "${N_MOLECULES}" \
  --expected-samples "${SAMPLES_PER_MOLECULE}" \
  --summary-json "${RUN_ROOT}/force_check_summary.json"

run_timed "04_grouped_split" \
  "${PYTHON_BIN}" mldft/utils/create_dataset_splits.py \
  "${DATASET_NAME}" \
  --override \
  --group-by-molecule \
  -p "${SPLIT_PROCESSES}"

run_timed "05_transform_cached_labels" \
  "${PYTHON_BIN}" -m mldft.datagen.transform_dataset \
  "data=qm9_pbe_force_full" \
  "data.dataset_name=${DATASET_NAME}" \
  "extras.enforce_tags=false" \
  "extras.print_config=false" \
  "+hydra.callbacks.git_logging.clean=false" \
  "+num_processes=${TRANSFORM_NUM_PROCESSES}" \
  "+num_threads_per_process=1" \
  "+start_idx=0" \
  "+num_molecules=999999" \
  "hydra.run.dir=${RUN_ROOT}/hydra/transform"

run_timed "06_statistics" \
  "${PYTHON_BIN}" -m mldft.ml.compute_dataset_statistics \
  "data=qm9_pbe_force_full" \
  "data.dataset_name=${DATASET_NAME}" \
  "name=${DATASET_NAME}_statistics" \
  "extras.enforce_tags=false" \
  "extras.print_config=false" \
  "+hydra.callbacks.git_logging.clean=false" \
  "overwrite=true" \
  "data.datamodule.batch_size=16" \
  "data.datamodule.num_workers=0" \
  "statistic_fitter_kwargs.n_batches=null" \
  "hydra.run.dir=${RUN_ROOT}/hydra/statistics"

export NPROC_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29500}"
export TORCHRUN
export NUM_WORKERS="${TRAIN_NUM_WORKERS}"
export ACCUMULATE_GRAD_BATCHES

check_gpus_free "before_train_eg"
run_timed "07_train_eg_e${DEMO_EPOCHS}" \
  env PER_GPU_BATCH_SIZE="${EG_BATCH_SIZE}" \
  bash scripts/launch_qm9_full_scale_8xa100.sh \
  eg "${DEMO_EPOCHS}" "node04_demo1000_eg_e${DEMO_EPOCHS}_${RUN_STAMP}" \
  "data.dataset_name=${DATASET_NAME}" \
  "tags=[qm9_node04_demo1000,eg,fixed_density]"

export MASTER_PORT="${MASTER_PORT_EGF:-29501}"
check_gpus_free "before_train_egf"
run_timed "08_train_egf_e${DEMO_EPOCHS}" \
  env PER_GPU_BATCH_SIZE="${EGF_BATCH_SIZE}" \
  bash scripts/launch_qm9_full_scale_8xa100.sh \
  egf "${DEMO_EPOCHS}" "node04_demo1000_egf_lam1_e${DEMO_EPOCHS}_${RUN_STAMP}" \
  "data.dataset_name=${DATASET_NAME}" \
  "tags=[qm9_node04_demo1000,egf,lambda1,fixed_density,force]"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | tee "${LOG_DIR}/gpu_status_after.csv"
fi

cat > "${RUN_ROOT}/summary.txt" <<EOF
QM9 node04 demo1000 pipeline complete
dataset=${DATASET_NAME}
data_dir=${DFT_DATA}/${DATASET_NAME}
run_root=${RUN_ROOT}
eg_run=${DFT_MODELS}/train/runs/node04_demo1000_eg_e${DEMO_EPOCHS}_${RUN_STAMP}
egf_run=${DFT_MODELS}/train/runs/node04_demo1000_egf_lam1_e${DEMO_EPOCHS}_${RUN_STAMP}
EOF

cat "${RUN_ROOT}/summary.txt"
