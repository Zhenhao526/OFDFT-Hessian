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
export MASTER_PORT="${MASTER_PORT:-29641}"
export NUM_WORKERS="${TRAIN_NUM_WORKERS:-8}"
export PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-4}"
export ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

TIER1_SUMMARY="${TIER1_SUMMARY:-}"
if [[ -z "${FORCE_WEIGHT:-}" ]]; then
  if [[ -z "${TIER1_SUMMARY}" ]]; then
    TIER1_SUMMARY="$(find "${DFT_MODELS}/eval/qm9_random1000_lambda_tier1" \
      -type f -path '*/analysis/tier1_model_summary.json' -printf '%T@ %p\n' \
      | sort -nr | head -1 | cut -d' ' -f2-)"
  fi
  if [[ -z "${TIER1_SUMMARY}" || ! -f "${TIER1_SUMMARY}" ]]; then
    echo "ERROR: FORCE_WEIGHT is unset and no Tier-1 summary was found" >&2
    exit 1
  fi
  selected_run="$(${PYTHON_BIN:-/scratch/xzh/envs/structures25/bin/python} - "${TIER1_SUMMARY}" <<'PY'
import json
import sys

recommended = json.load(open(sys.argv[1]))["recommended_for_tier2"]
if not recommended:
    raise SystemExit("Tier-1 summary contains no recommended model")
print(recommended[0])
PY
)"
  case "${selected_run}" in
    EGF_w0p1) FORCE_WEIGHT=0.1 ;;
    EGF_w0p3) FORCE_WEIGHT=0.3 ;;
    EGF_w1p0) FORCE_WEIGHT=1.0 ;;
    EGF_w3p0) FORCE_WEIGHT=3.0 ;;
    EGF_w10p0) FORCE_WEIGHT=10.0 ;;
    *)
      echo "ERROR: unsupported Tier-1 recommendation: ${selected_run}" >&2
      exit 1
      ;;
  esac
fi
FIXED_SEED="${FIXED_SEED:-676368232}"
EPOCHS="${EPOCHS:-10}"
BASE_BATCHES_PER_EPOCH="${BASE_BATCHES_PER_EPOCH:-1233}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
TAG="${FORCE_WEIGHT/./p}"
RUN_NAME="${RUN_NAME:-qm9_random1000_paired_egf_forcew${TAG}_computematched_${RUN_STAMP}}"
MANIFEST="${DFT_DATA}/${DATASET_NAME}/paired_augmentation_manifest.json"

for path in \
  "${DFT_DATA}/${DATASET_NAME}/split.pkl" \
  "${MANIFEST}"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: missing paired training artifact: ${path}" >&2
    exit 1
  fi
done

"${PYTHON_BIN:-/scratch/xzh/envs/structures25/bin/python}" - "${MANIFEST}" <<'PY'
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

echo "training paired candidate force_loss.weight=${FORCE_WEIGHT} run=${RUN_NAME}"
echo "tier1_summary=${TIER1_SUMMARY:-explicit_FORCE_WEIGHT}"
echo "compute match: epochs=${EPOCHS}, limit_train_batches=${BASE_BATCHES_PER_EPOCH}"
bash scripts/launch_qm9_full_scale_8xa100.sh \
  egf "${EPOCHS}" "${RUN_NAME}" \
  "data.dataset_name=${DATASET_NAME}" \
  "seed=${FIXED_SEED}" \
  "+trainer.limit_train_batches=${BASE_BATCHES_PER_EPOCH}" \
  "model.loss_function.force_loss.weight=${FORCE_WEIGHT}" \
  "model.validation_loss_function.force_loss.weight=${FORCE_WEIGHT}" \
  "callbacks.throughput_monitor.estimate_num_samples=42670" \
  "callbacks.throughput_monitor.target_num_devices=8" \
  "extras.enforce_tags=false" \
  "extras.print_config=false" \
  "hydra.callbacks.git_logging.clean=false" \
  "tags=[qm9_random1000,paired_train_only,egf,force_weight_${TAG},compute_matched,8xa100]"

echo "paired_candidate_run=${DFT_MODELS}/train/runs/${RUN_NAME}"
