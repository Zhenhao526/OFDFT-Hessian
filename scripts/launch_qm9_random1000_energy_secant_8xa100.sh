#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
source "${ENV_FILE:-/scratch/xzh/env.sh}"

export DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
export DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"
export DATASET_NAME="${DATASET_NAME:-QM9PBEForceRandom1000PairedAug}"
export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE=8 NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29673}"
export TORCHRUN="${TORCHRUN:-/scratch/xzh/envs/structures25/bin/torchrun}"
export NUM_WORKERS="${TRAIN_NUM_WORKERS:-8}"
export MULTIPROCESSING_SHARING_STRATEGY="${TRAIN_MP_SHARING_STRATEGY:-file_system}"
export PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-4}"
export ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

EPOCHS="${EPOCHS:-10}"
BASE_BATCHES_PER_EPOCH="${BASE_BATCHES_PER_EPOCH:-1233}"
FIXED_SEED="${FIXED_SEED:-676368232}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-qm9_random1000_energy_secant_w0p1_e${EPOCHS}_${RUN_STAMP}}"
WEIGHT_CKPT="${WEIGHT_CKPT:-/scratch/xzh/models/train/runs/qm9_random1000_egf_forcew1p0_e10_20260714_202203/checkpoints/epoch_009.ckpt}"
RESUME_CKPT="${RESUME_CKPT:-}"
MANIFEST="${DFT_DATA}/${DATASET_NAME}/paired_augmentation_manifest.json"

for path in \
  "${DFT_DATA}/${DATASET_NAME}/split.pkl" \
  "${MANIFEST}" \
  "${WEIGHT_CKPT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: missing required input: ${path}" >&2
    exit 1
  fi
done
if [[ -n "${RESUME_CKPT}" && ! -f "${RESUME_CKPT}" ]]; then
  echo "ERROR: missing resume checkpoint: ${RESUME_CKPT}" >&2
  exit 1
fi

if [[ -n "${RESUME_CKPT}" ]]; then
  CHECKPOINT_OVERRIDES=("ckpt_path=${RESUME_CKPT}" "weight_ckpt_path=null")
else
  CHECKPOINT_OVERRIDES=("weight_ckpt_path=${WEIGHT_CKPT}")
fi

python - "${MANIFEST}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
assert manifest["paired_parents"] == 800, manifest
assert manifest["paired_entries"] == 1600, manifest
assert not any(manifest["parent_overlap"].values()), manifest["parent_overlap"]
print(json.dumps({
    "paired_parents": manifest["paired_parents"],
    "paired_entries": manifest["paired_entries"],
    "parent_overlap": manifest["parent_overlap"],
    "split_sha256": manifest["split_sha256"],
}, indent=2))
PY

echo "run=${RUN_NAME} dataset=${DATASET_NAME} initial_weights=${WEIGHT_CKPT} resume=${RESUME_CKPT:-none}"
echo "8xA100 batch=4 steps_per_epoch=${BASE_BATCHES_PER_EPOCH} epochs=${EPOCHS} workers=${NUM_WORKERS} sharing=${MULTIPROCESSING_SHARING_STRATEGY}"

bash scripts/launch_qm9_full_scale_8xa100.sh \
  energy_secant "${EPOCHS}" "${RUN_NAME}" \
  "data.dataset_name=${DATASET_NAME}" \
  "${CHECKPOINT_OVERRIDES[@]}" \
  "seed=${FIXED_SEED}" \
  "+trainer.limit_train_batches=${BASE_BATCHES_PER_EPOCH}" \
  "trainer.use_distributed_sampler=false" \
  "callbacks.throughput_monitor.estimate_num_samples=39456" \
  "callbacks.throughput_monitor.target_num_devices=8" \
  "validate=false" \
  "test=false" \
  "extras.enforce_tags=false" \
  "extras.print_config=false" \
  "hydra.callbacks.git_logging.clean=false" \
  "tags=[qm9_random1000,parent_grouped,energy_secant_w0p1,compute_matched,8xa100]"

echo "candidate_run_dir=${DFT_MODELS}/train/runs/${RUN_NAME}"
