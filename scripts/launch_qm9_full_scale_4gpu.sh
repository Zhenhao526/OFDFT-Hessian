#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  scripts/launch_qm9_full_scale_4gpu.sh <eg|egf> <10|20> [run_name] [extra hydra overrides...]

Examples:
  scripts/launch_qm9_full_scale_4gpu.sh eg 10
  scripts/launch_qm9_full_scale_4gpu.sh egf 10
  scripts/launch_qm9_full_scale_4gpu.sh egf 20 qm9_full_egf_lam1_e20_4gpu
  scripts/launch_qm9_full_scale_4gpu.sh egf 10 qm9_full_egf_resume ckpt_path=/path/to/last.ckpt

Optional environment overrides:
  DFT_DATA                 Root containing QM9PBEForceFull. If unset, common local paths are tried.
  DFT_MODELS               Model output root. Default: <repo>/_runtime/qm9_full_models
  CUDA_VISIBLE_DEVICES     GPU list. Default: 0,1,2,3
  PER_GPU_BATCH_SIZE       EG default: 8; EGF default: 4
  ACCUMULATE_GRAD_BATCHES  Default: 1
  NUM_WORKERS              Default: 8
  MASTER_PORT              Default: 29500
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_KIND="$1"
EPOCHS="$2"
RUN_NAME="${3:-qm9_full_${MODEL_KIND}_e${EPOCHS}_4gpu}"
EXTRA_OVERRIDES=("${@:4}")

case "${MODEL_KIND}" in
  eg)
    DEFAULT_BATCH_SIZE=8
    ;;
  egf)
    DEFAULT_BATCH_SIZE=4
    ;;
  *)
    echo "MODEL_KIND must be 'eg' or 'egf', got '${MODEL_KIND}'" >&2
    exit 2
    ;;
esac

if [[ "${EPOCHS}" != "10" && "${EPOCHS}" != "20" ]]; then
  echo "EPOCHS must be 10 or 20, got '${EPOCHS}'" >&2
  exit 2
fi

find_dft_data_root() {
  local candidate
  if [[ -n "${DFT_DATA:-}" ]]; then
    echo "${DFT_DATA}"
    return 0
  fi

  for candidate in \
    "${REPO_ROOT}/_runtime/qm9_full" \
    "${REPO_ROOT}/_runtime/qm9_full_data" \
    "/mnt/afs/home/xiazhenhao/dft/qm9_full" \
    "/mnt/afs/home/xiazhenhao/dft/data" \
    "/mnt/afs/home/xiazhenhao/dft" \
    "/mnt/afs/home/xiazhenhao/data" \
    "/mnt/afs/home/xiazhenhao"
  do
    if [[ -d "${candidate}/QM9PBEForceFull" ]]; then
      echo "${candidate}"
      return 0
    fi
  done

  return 1
}

if ! DFT_DATA_RESOLVED="$(find_dft_data_root)"; then
  cat >&2 <<'ERROR'
Could not locate full QM9 data.

Expected layout:
  ${DFT_DATA}/QM9PBEForceFull/split.pkl
  ${DFT_DATA}/QM9PBEForceFull/labels/...

Run with DFT_DATA set once on the command line, for example:
  DFT_DATA=/path/to/data_root scripts/launch_qm9_full_scale_4gpu.sh egf 10
ERROR
  exit 2
fi

export DFT_DATA="${DFT_DATA_RESOLVED}"
export DFT_MODELS="${DFT_MODELS:-${REPO_ROOT}/_runtime/qm9_full_models}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE=4
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
export ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

if [[ ! -d "${DFT_DATA}/QM9PBEForceFull" ]]; then
  echo "Missing dataset directory: ${DFT_DATA}/QM9PBEForceFull" >&2
  exit 2
fi

if [[ ! -f "${DFT_DATA}/QM9PBEForceFull/split.pkl" ]]; then
  echo "Missing split file: ${DFT_DATA}/QM9PBEForceFull/split.pkl" >&2
  exit 2
fi

mkdir -p "${DFT_MODELS}"

echo "Launching full QM9 ${MODEL_KIND^^} training on 4 GPUs"
echo "  repo:                 ${REPO_ROOT}"
echo "  DFT_DATA:             ${DFT_DATA}"
echo "  DFT_MODELS:           ${DFT_MODELS}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  epochs:               ${EPOCHS}"
echo "  per-GPU batch:        ${PER_GPU_BATCH_SIZE}"
echo "  effective batch:      $((NPROC_PER_NODE * PER_GPU_BATCH_SIZE * ACCUMULATE_GRAD_BATCHES))"
echo "  run name:             ${RUN_NAME}"
echo

cd "${REPO_ROOT}"

exec "${SCRIPT_DIR}/launch_qm9_full_scale_8xa100.sh" \
  "${MODEL_KIND}" \
  "${EPOCHS}" \
  "${RUN_NAME}" \
  callbacks.throughput_monitor.target_num_devices=4 \
  "${EXTRA_OVERRIDES[@]}"
