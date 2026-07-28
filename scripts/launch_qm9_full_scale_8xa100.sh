#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <eg|egf|energy_secant|egf_energy_secant> <10|20> [run_name] [extra hydra overrides...]" >&2
  exit 2
fi

MODEL_KIND="$1"
EPOCHS="$2"
RUN_NAME="${3:-qm9_full_${MODEL_KIND}_e${EPOCHS}_8xa100}"
EXTRA_OVERRIDES=("${@:4}")

case "${MODEL_KIND}" in
  eg)
    EXPERIMENT="str25/qm9_pbe_force_full_eg"
    DEFAULT_BATCH_SIZE=8
    ;;
  egf)
    EXPERIMENT="str25/qm9_pbe_force_full_egf"
    DEFAULT_BATCH_SIZE=4
    ;;
  energy_secant)
    EXPERIMENT="str25/qm9_pbe_force_full_energy_secant"
    DEFAULT_BATCH_SIZE=4
    ;;
  egf_energy_secant)
    EXPERIMENT="str25/qm9_pbe_force_full_egf_energy_secant"
    DEFAULT_BATCH_SIZE=4
    ;;
  *)
    echo "Unsupported MODEL_KIND '${MODEL_KIND}'" >&2
    exit 2
    ;;
esac

if [[ "${EPOCHS}" != "10" && "${EPOCHS}" != "20" ]]; then
  echo "EPOCHS must be 10 or 20, got '${EPOCHS}'" >&2
  exit 2
fi

: "${DFT_DATA:?Set DFT_DATA to the directory containing QM9PBEForceFull}"
: "${DFT_MODELS:?Set DFT_MODELS to the model output directory}"

NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
NODE_RANK="${NODE_RANK:-0}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MULTIPROCESSING_SHARING_STRATEGY="${MULTIPROCESSING_SHARING_STRATEGY:-file_system}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-${DFT_MODELS}/train/runs/${RUN_NAME}}"
TORCHRUN="${TORCHRUN:-.venv/bin/torchrun}"

"${TORCHRUN}" \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m mldft.ml.train \
  experiment="${EXPERIMENT}" \
  name="${RUN_NAME}" \
  hydra.run.dir="${HYDRA_RUN_DIR}" \
  trainer.max_epochs="${EPOCHS}" \
  trainer.devices="${NPROC_PER_NODE}" \
  trainer.num_nodes="${NNODES}" \
  trainer.accumulate_grad_batches="${ACCUMULATE_GRAD_BATCHES}" \
  data.datamodule.batch_size="${PER_GPU_BATCH_SIZE}" \
  data.datamodule.num_workers="${NUM_WORKERS}" \
  multiprocessing_sharing_strategy="${MULTIPROCESSING_SHARING_STRATEGY}" \
  callbacks.throughput_monitor.target_num_devices=8 \
  "${EXTRA_OVERRIDES[@]}"
