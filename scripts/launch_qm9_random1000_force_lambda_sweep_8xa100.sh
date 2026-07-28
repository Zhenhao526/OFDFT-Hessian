#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
if [[ -f "${ENV_FILE:-/scratch/xzh/env.sh}" ]]; then
  # shellcheck source=/dev/null
  source "${ENV_FILE:-/scratch/xzh/env.sh}"
fi

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
export NPROC_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export NUM_WORKERS="${TRAIN_NUM_WORKERS:-8}"
export ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"

EPOCHS="${EPOCHS:-10}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-4}"
FIXED_SEED="${FIXED_SEED:-676368232}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${DFT_MODELS}/train_lambda_sweep/${RUN_STAMP}}"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}"

for path in \
  "${DFT_DATA}/${DATASET_NAME}/split.pkl" \
  "${DFT_DATA}/${DATASET_NAME}/labels_local_frames_global_symmetric_natrep" \
  "${DFT_DATA}/${DATASET_NAME}/dataset_statistics/dataset_statistics_labels_local_frames_global_symmetric_natrep_e_kin_plus_xc.zarr"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: missing prepared training artifact: ${path}" >&2
    exit 1
  fi
done

# The historical random1000 checkpoint named lam1 used force_loss.weight=0.1.
# Reuse it as the lambda=0.1 baseline and train the four missing actual weights.
lambdas=(0.3 1.0 3.0 10.0)
ports=(29631 29632 29633 29634)

for idx in "${!lambdas[@]}"; do
  lambda="${lambdas[$idx]}"
  tag="${lambda/./p}"
  run_name="qm9_random1000_egf_forcew${tag}_e${EPOCHS}_${RUN_STAMP}"
  export MASTER_PORT="${ports[$idx]}"
  export PER_GPU_BATCH_SIZE
  echo "training force_loss.weight=${lambda}, run=${run_name}, seed=${FIXED_SEED}"
  /usr/bin/time -v -o "${LOG_DIR}/forcew${tag}.time.txt" \
    bash scripts/launch_qm9_full_scale_8xa100.sh \
      egf "${EPOCHS}" "${run_name}" \
      "data.dataset_name=${DATASET_NAME}" \
      "seed=${FIXED_SEED}" \
      "model.loss_function.force_loss.weight=${lambda}" \
      "model.validation_loss_function.force_loss.weight=${lambda}" \
      "callbacks.throughput_monitor.estimate_num_samples=42670" \
      "callbacks.throughput_monitor.target_num_devices=8" \
      "extras.enforce_tags=false" \
      "extras.print_config=false" \
      "hydra.callbacks.git_logging.clean=false" \
      "tags=[qm9_random1000,egf,force_weight_${tag},fixed_density,force,8xa100]" \
      >"${LOG_DIR}/forcew${tag}.log" 2>&1
done

cat >"${RUN_ROOT}/summary.txt" <<EOF
QM9 random1000 actual force-loss-weight sweep complete
historical nominal-lambda1 checkpoint actual force_loss.weight=0.1
trained_weights=0.3,1.0,3.0,10.0
epochs=${EPOCHS}
seed=${FIXED_SEED}
per_gpu_batch_size=${PER_GPU_BATCH_SIZE}
run_stamp=${RUN_STAMP}
EOF
cat "${RUN_ROOT}/summary.txt"
