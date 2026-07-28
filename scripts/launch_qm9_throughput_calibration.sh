#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 <eg|egf> [steps] [nproc_per_node]" >&2
  exit 2
fi

MODEL_KIND="$1"
STEPS="${2:-20}"
NPROC_PER_NODE="${3:-1}"

case "${MODEL_KIND}" in
  eg)
    EXPERIMENT="str25/qm9_pbe_force_full_eg"
    DEFAULT_BATCH_SIZE=8
    ;;
  egf)
    EXPERIMENT="str25/qm9_pbe_force_full_egf"
    DEFAULT_BATCH_SIZE=4
    ;;
  *)
    echo "MODEL_KIND must be 'eg' or 'egf', got '${MODEL_KIND}'" >&2
    exit 2
    ;;
esac

: "${DFT_DATA:?Set DFT_DATA. For development calibration this can point to _runtime/qm9_p1}"
: "${DFT_MODELS:?Set DFT_MODELS to the model output directory}"

RUN_NAME="${RUN_NAME:-qm9_full_${MODEL_KIND}_throughput_${NPROC_PER_NODE}gpu_${STEPS}steps}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-${DFT_MODELS}/train/runs/${RUN_NAME}}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TORCHRUN="${TORCHRUN:-.venv/bin/torchrun}"

COMMON_ARGS=(
  experiment="${EXPERIMENT}"
  data=qm9_pbe_force_pilot
  name="${RUN_NAME}"
  hydra.run.dir="${HYDRA_RUN_DIR}"
  hydra.callbacks.git_logging.clean=false
  extras.enforce_tags=false
  extras.print_config=false
  train=true
  validate=false
  test=false
  trainer.max_epochs=1
  +trainer.max_steps="${STEPS}"
  +trainer.limit_val_batches=0
  trainer.devices="${NPROC_PER_NODE}"
  trainer.num_nodes=1
  trainer.log_every_n_steps=1
  data.datamodule.batch_size="${PER_GPU_BATCH_SIZE}"
  data.datamodule.num_workers="${NUM_WORKERS}"
  callbacks.throughput_monitor.log_every_n_steps=2
  callbacks.throughput_monitor.warmup_steps=1
  callbacks.throughput_monitor.target_num_devices=8
  callbacks.model_checkpoint.monitor=null
  callbacks.model_checkpoint.save_top_k=0
  callbacks.model_checkpoint.save_last=false
)

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  "${TORCHRUN}" \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    -m mldft.ml.train \
    "${COMMON_ARGS[@]}"
else
  .venv/bin/python -m mldft.ml.train trainer.strategy=auto "${COMMON_ARGS[@]}"
fi
