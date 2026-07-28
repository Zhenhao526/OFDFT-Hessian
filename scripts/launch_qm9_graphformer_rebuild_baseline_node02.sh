#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

NODE_ROOT="${NODE_ROOT:-/home/shenwei01/xzh_node02_20260724}"
source "${ROOT_DIR}/scripts/activate_qm9_node02_local.sh"

export DFT_DATA="${TRAIN_DFT_DATA:-${NODE_ROOT}/data}"
export DFT_MODELS="${TRAIN_DFT_MODELS:-${NODE_ROOT}/models}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

DATASET_NAME="QM9PBEForceRandom1000Train800RebuildV1"
DATASET_ROOT="${DFT_DATA}/${DATASET_NAME}"
RUN_NAME="${RUN_NAME:-qm9_train800_egf_force1_s12330_rebuild_v1}"
RUN_DIR="${DFT_MODELS}/train/runs/${RUN_NAME}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-4}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-8}"
EFFECTIVE_BATCH="$((NPROC_PER_NODE * PER_GPU_BATCH_SIZE * ACCUMULATE_GRAD_BATCHES))"

if [[ "${EFFECTIVE_BATCH}" != "32" ]]; then
  echo "ERROR: frozen effective global batch must be 32, got ${EFFECTIVE_BATCH}" >&2
  exit 2
fi
if [[ "${NPROC_PER_NODE}" != "1" || "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  echo "ERROR: the frozen node02 baseline protocol uses exactly one physical GPU" >&2
  exit 2
fi
ACTIVE_GPU_PIDS="$(
  nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'
)"
if [[ -n "${ACTIVE_GPU_PIDS}" ]]; then
  echo "ERROR: physical GPU ${CUDA_VISIBLE_DEVICES} is busy (PIDs: ${ACTIVE_GPU_PIDS//$'\n'/,})" >&2
  exit 2
fi
if [[ -e "${RUN_DIR}/checkpoints/last.ckpt" ]]; then
  echo "ERROR: registered run already has a checkpoint: ${RUN_DIR}" >&2
  exit 2
fi

TRAIN_SAMPLES="$("${PYTHON_BIN}" - "${DATASET_ROOT}/split.pkl" <<'PY'
import pickle
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("rb") as handle:
    split = pickle.load(handle)
assert split.get("train_only") is True
assert len(split["train"]) == 3200
assert len(split["val"]) == 0
assert len(split["test"]) == 0
assert int(split["sizes"]["train"]) == 39456
print(int(split["sizes"]["train"]))
PY
)"
echo "train-only split preflight passed: ${TRAIN_SAMPLES} SCF-expanded samples"

mkdir -p "${RUN_DIR}"
{
  echo "host=$(hostname)"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "run_name=${RUN_NAME}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "nproc_per_node=${NPROC_PER_NODE}"
  echo "per_gpu_batch_size=${PER_GPU_BATCH_SIZE}"
  echo "accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES}"
  echo "effective_global_batch=${EFFECTIVE_BATCH}"
  echo "lightning_use_distributed_sampler=false"
  echo "fixed_max_steps=12330"
  echo "train_samples=${TRAIN_SAMPLES}"
  echo "validation_access=false"
  echo "test100_access=false"
} > "${RUN_DIR}/launch_provenance.txt"

/usr/bin/time -v -o "${RUN_DIR}/train.time.txt" \
  "$(dirname "${PYTHON_BIN}")/torchrun" \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    -m mldft.ml.train \
    experiment=str25/qm9_pbe_force_train800_egf_rebuild_v1 \
    name="${RUN_NAME}" \
    hydra.run.dir="${RUN_DIR}" \
    trainer.devices="${NPROC_PER_NODE}" \
    trainer.accumulate_grad_batches="${ACCUMULATE_GRAD_BATCHES}" \
    trainer.use_distributed_sampler=false \
    data.datamodule.batch_size="${PER_GPU_BATCH_SIZE}" \
    callbacks.throughput_monitor.estimate_num_samples="${TRAIN_SAMPLES}" \
    callbacks.throughput_monitor.target_num_devices="${NPROC_PER_NODE}" \
    extras.enforce_tags=false \
    extras.print_config=false \
    hydra.callbacks.git_logging.clean=false \
    > "${RUN_DIR}/train.log" 2>&1

"${PYTHON_BIN}" - \
  "${RUN_DIR}" \
  "${ROOT_DIR}/configs/audit/qm9_graphformer_hybrid_relaxed_hvp_rebuild_v1.yaml" \
  "${ROOT_DIR}/configs/ml/experiment/str25/qm9_pbe_force_train800_egf_rebuild_v1.yaml" \
  "${DATASET_ROOT}/provenance/train_only_dataset_manifest.json" <<'PY'
import hashlib
import csv
import json
import sys
from pathlib import Path

import torch
import yaml

# The checkpoint's Hydra metadata imports local_frames while unpickling. Its
# module-level Jd.pt load is otherwise nested inside this torch.load, which
# corrupts PyTorch 2.4's thread-local map_location state on first import.
import mldft.utils.local_frames  # noqa: F401

run_dir = Path(sys.argv[1])
protocol_path = Path(sys.argv[2])
experiment_path = Path(sys.argv[3])
dataset_manifest_path = Path(sys.argv[4])
sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
protocol = yaml.safe_load(protocol_path.read_text())
checkpoint = run_dir / "checkpoints" / "last.ckpt"
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
step = int(payload["global_step"])
if step != 12330:
    raise RuntimeError(f"Fixed-step checkpoint drift: {step} != 12330")
throughput_path = run_dir / "throughput" / "throughput.csv"
with throughput_path.open(newline="") as handle:
    throughput_rows = list(csv.DictReader(handle))
if not throughput_rows:
    raise RuntimeError(f"No runtime throughput records: {throughput_path}")
batch_tuples = {
    (
        int(row["per_gpu_batch_size"]),
        int(row["accumulate_grad_batches"]),
        int(row["effective_global_batch_size"]),
    )
    for row in throughput_rows
}
expected_batch_tuple = (4, 8, 32)
if batch_tuples != {expected_batch_tuple}:
    raise RuntimeError(
        f"Runtime batch protocol drift: {sorted(batch_tuples)} != "
        f"{expected_batch_tuple}"
    )
record = {
    "baseline_name": protocol["identity"]["baseline_name"],
    "run_dir": str(run_dir),
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": sha256(checkpoint),
    "global_step": step,
    "runtime_batch_protocol": {
        "per_gpu_batch_size": expected_batch_tuple[0],
        "accumulate_grad_batches": expected_batch_tuple[1],
        "effective_global_batch_size": expected_batch_tuple[2],
        "throughput_rows": len(throughput_rows),
    },
    "selection": "fixed_final_step_without_validation",
    "protocol_id": protocol["protocol_id"],
    "protocol_sha256": sha256(protocol_path),
    "experiment_config": str(experiment_path),
    "experiment_config_sha256": sha256(experiment_path),
    "train_only_dataset_manifest": str(dataset_manifest_path),
    "train_only_dataset_manifest_sha256": sha256(dataset_manifest_path),
    "old_original_a_identity_used": False,
    "validation_accessed": False,
    "test100_accessed": False,
}
(run_dir / "checkpoint_registration.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(record, indent=2, sort_keys=True))
PY
