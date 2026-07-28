#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

NODE_ROOT="${NODE_ROOT:-/home/shenwei01/xzh_node02_20260724}"
PROTOCOL="${ROOT_DIR}/configs/audit/qm9_graphformer_hybrid_relaxed_hvp_rebuild_v1.yaml"
BRANCH="graphformer_hybrid_relaxed_hvp_rebuild_v1"
ARTIFACT_ROOT="${NODE_ROOT}/artifacts/${BRANCH}"
RUN_ROOT="${NODE_ROOT}/runs/${BRANCH}"
DATASET_ROOT="${NODE_ROOT}/data/QM9PBEForceRandom1000Train800RebuildV1"
PBE_ROOT="${ARTIFACT_ROOT}/pbe_hessian_train20"
PBE_MANIFEST="${PBE_ROOT}/manifest.json"
MANIFEST_ROOT="${ARTIFACT_ROOT}/manifests"
DIRECTION_ROOT="${ARTIFACT_ROOT}/directions"
BASELINE_RUN="${NODE_ROOT}/models/train/runs/qm9_train800_egf_force1_s12330_rebuild_v1"
GPU_INDEX="${GPU_INDEX:-6}"
GPU_LOCK="${NODE_ROOT}/locks/gpu${GPU_INDEX}.lock"
POLL_SECONDS="${POLL_SECONDS:-300}"
GPU4PYSCF_RUNNER="${GPU4PYSCF_RUNNER:-${ROOT_DIR}/scripts/run_gpu4pyscf_node02.sh}"

source "${ROOT_DIR}/scripts/activate_qm9_node02_local.sh"
export DFT_DATA="${TRAIN_DFT_DATA:-${NODE_ROOT}/data}"
export DFT_MODELS="${TRAIN_DFT_MODELS:-${NODE_ROOT}/models}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"

mkdir -p \
  "${ARTIFACT_ROOT}/provenance" \
  "${PBE_ROOT}" \
  "${MANIFEST_ROOT}" \
  "${DIRECTION_ROOT}" \
  "${RUN_ROOT}/logs" \
  "$(dirname "${GPU_LOCK}")"

PROTOCOL_SHA256="$(sha256sum "${PROTOCOL}" | awk '{print $1}')"
cat > "${ARTIFACT_ROOT}/provenance/protocol_registration.json" <<EOF
{
  "branch_id": "graphformer_hybrid_relaxed_hvp_rebuild_20260724",
  "method_name": "解析密度/KKT响应的hybrid relaxed-HVP",
  "old_original_a_identity_used": false,
  "protocol": "${PROTOCOL}",
  "protocol_sha256": "${PROTOCOL_SHA256}",
  "test100_accessed": false,
  "validation_accessed": false
}
EOF

gpu_is_idle() {
  local pids
  pids="$(
    nvidia-smi -i "${GPU_INDEX}" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'
  )"
  [[ -z "${pids}" ]]
}

pbe_manifest_is_complete() {
  [[ -f "${PBE_MANIFEST}" ]] || return 1
  "${PYTHON_BIN}" - "${PBE_MANIFEST}" "${PROTOCOL_SHA256}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
assert manifest["protocol_sha256"] == sys.argv[2]
assert manifest["parent_count"] == 20
assert manifest["success_count"] == 20
assert manifest["failed_count"] == 0
assert all(row["success"] and row["finite"] for row in manifest["parents"])
PY
}

wait_for_gpu() {
  until gpu_is_idle; do
    echo "$(date --iso-8601=seconds) waiting for idle GPU ${GPU_INDEX}"
    sleep "${POLL_SECONDS}"
  done
}

train20_chk_count() {
  "${PYTHON_BIN}" - "${PROTOCOL}" "${DATASET_ROOT}" <<'PY'
import sys
from pathlib import Path

import yaml

protocol = yaml.safe_load(Path(sys.argv[1]).read_text())
root = Path(sys.argv[2]) / "kohn_sham"
count = 0
for molecule_id in protocol["parent_sets"]["train20"]["molecule_ids"]:
    if len(list(root.glob(f"*_{molecule_id}.0000000.chk"))) == 1:
        count += 1
print(count)
PY
}

if ! pbe_manifest_is_complete; then
  while [[ "$(train20_chk_count)" != "20" ]]; do
    echo "$(date --iso-8601=seconds) waiting for frozen train20 sample-0 chk files: $(train20_chk_count)/20"
    sleep "${POLL_SECONDS}"
  done
  if [[ -f "${PBE_MANIFEST}" ]]; then
    failed_at="$(date +%Y%m%dT%H%M%S)"
    mv "${PBE_MANIFEST}" "${PBE_ROOT}/manifest.failed_${failed_at}.json"
    if [[ -f "${PBE_ROOT}/manifest.registration.json" ]]; then
      mv "${PBE_ROOT}/manifest.registration.json" \
        "${PBE_ROOT}/manifest.registration.failed_${failed_at}.json"
    fi
  fi
  if "${GPU4PYSCF_RUNNER}" -c \
    'import cupy, gpu4pyscf; assert cupy.cuda.runtime.getDeviceCount() == 1' \
    >/dev/null 2>&1; then
    wait_for_gpu
    (
      flock -x 9
      if ! gpu_is_idle; then
        echo "ERROR: GPU ${GPU_INDEX} became busy after acquiring the branch lock" >&2
        exit 2
      fi
      /usr/bin/time -v \
        -o "${RUN_ROOT}/logs/pbe_hessian_train20.time.txt" \
        "${GPU4PYSCF_RUNNER}" scripts/qm9_train800_pbe_hessian_references.py \
          --protocol "${PROTOCOL}" \
          --dataset-dir "${DATASET_ROOT}" \
          --output-dir "${PBE_ROOT}/cache" \
          --manifest "${PBE_MANIFEST}" \
          --backend gpu4pyscf \
          --workers 1 \
          > "${RUN_ROOT}/logs/pbe_hessian_train20.log" 2>&1
    ) 9>"${GPU_LOCK}"
  else
    while pgrep -f "mldft.datagen.transform_dataset.*qm9_pbe_force_random1000_train800_rebuild_v1" >/dev/null; do
      echo "$(date --iso-8601=seconds) waiting for transform before CPU PBE Hessians"
      sleep "${POLL_SECONDS}"
    done
    /usr/bin/time -v \
      -o "${RUN_ROOT}/logs/pbe_hessian_train20.time.txt" \
      "${PYTHON_BIN}" scripts/qm9_train800_pbe_hessian_references.py \
        --protocol "${PROTOCOL}" \
        --dataset-dir "${DATASET_ROOT}" \
        --output-dir "${PBE_ROOT}/cache" \
        --manifest "${PBE_MANIFEST}" \
        --backend cpu \
        --workers 8 \
        > "${RUN_ROOT}/logs/pbe_hessian_train20.log" 2>&1
  fi
  pbe_manifest_is_complete
fi

while [[ ! -f "${DATASET_ROOT}/provenance/train_only_dataset_manifest.json" ]]; do
  if ! pgrep -f "launch_qm9_train800_fresh_labels_node02.sh" >/dev/null; then
    echo "ERROR: label pipeline stopped before final train-only manifest registration" >&2
    exit 2
  fi
  echo "$(date --iso-8601=seconds) waiting for rebuilt train-only labels"
  sleep "${POLL_SECONDS}"
done

"${PYTHON_BIN}" scripts/prepare_qm9_graphformer_rebuild_manifests.py parents \
  --protocol "${PROTOCOL}" \
  --dataset-dir "${DATASET_ROOT}" \
  --pbe-manifest "${PBE_MANIFEST}" \
  --parent-set stable5 \
  --output "${MANIFEST_ROOT}/stable5_parent_manifest.json"
"${PYTHON_BIN}" scripts/prepare_qm9_graphformer_rebuild_manifests.py directions \
  --protocol "${PROTOCOL}" \
  --parent-manifest "${MANIFEST_ROOT}/stable5_parent_manifest.json" \
  --parent-set stable5 \
  --output-dir "${DIRECTION_ROOT}/stable5"
"${PYTHON_BIN}" scripts/prepare_qm9_graphformer_rebuild_manifests.py parents \
  --protocol "${PROTOCOL}" \
  --dataset-dir "${DATASET_ROOT}" \
  --pbe-manifest "${PBE_MANIFEST}" \
  --parent-set train20 \
  --output "${MANIFEST_ROOT}/train20_parent_manifest.json"
"${PYTHON_BIN}" scripts/prepare_qm9_graphformer_rebuild_manifests.py directions \
  --protocol "${PROTOCOL}" \
  --parent-manifest "${MANIFEST_ROOT}/train20_parent_manifest.json" \
  --parent-set train20 \
  --output-dir "${DIRECTION_ROOT}/train20"

wait_for_gpu
(
  flock -x 9
  if [[ ! -f "${BASELINE_RUN}/checkpoint_registration.json" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_INDEX}" \
      bash scripts/launch_qm9_graphformer_rebuild_baseline_node02.sh
  fi
) 9>"${GPU_LOCK}"

AUDIT_DIR="${RUN_ROOT}/audit_0028399"
if [[ -e "${AUDIT_DIR}/summary.json" || -e "${AUDIT_DIR}/PASSED" ]]; then
  echo "ERROR: audit output already exists; refusing to overwrite ${AUDIT_DIR}" >&2
  exit 2
fi
wait_for_gpu
(
  flock -x 9
  if ! gpu_is_idle; then
    echo "ERROR: GPU ${GPU_INDEX} became busy after acquiring the branch lock" >&2
    exit 2
  fi
  mkdir -p "${AUDIT_DIR}"
  /usr/bin/time -v -o "${AUDIT_DIR}/command.time.txt" \
    "${PYTHON_BIN}" scripts/qm9_graphformer_analytic_relaxed_hvp_audit.py \
      --protocol "${PROTOCOL}" \
      --checkpoint-registration "${BASELINE_RUN}/checkpoint_registration.json" \
      --parent-manifest "${MANIFEST_ROOT}/stable5_parent_manifest.json" \
      --direction-manifest "${DIRECTION_ROOT}/stable5/manifest.json" \
      --output-dir "${AUDIT_DIR}" \
      --molecule 0028399 \
      --strict-fd-displacement 1e-4 \
      --symmetric-matrix-power-mode eigh_second_order_audit \
      --device cuda:0 \
      > "${AUDIT_DIR}/command.log" 2>&1
  touch "${AUDIT_DIR}/PASSED"
) 9>"${GPU_LOCK}"

CHECKPOINT="$(
  "${PYTHON_BIN}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' \
    "${BASELINE_RUN}/checkpoint_registration.json"
)"
CHECKPOINT_SHA="$(
  "${PYTHON_BIN}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_sha256"])' \
    "${BASELINE_RUN}/checkpoint_registration.json"
)"
DIRECTION_SHA="$(sha256sum "${DIRECTION_ROOT}/stable5/manifest.json" | awk '{print $1}')"
CAPACITY_DIR="${RUN_ROOT}/capacity_0028399"

if [[ -e "${CAPACITY_DIR}/summary.json" || -e "${CAPACITY_DIR}/PASSED" ]]; then
  echo "ERROR: capacity output already exists; refusing to overwrite ${CAPACITY_DIR}" >&2
  exit 2
fi
wait_for_gpu
(
  flock -x 9
  if ! gpu_is_idle; then
    echo "ERROR: GPU ${GPU_INDEX} became busy after acquiring the branch lock" >&2
    exit 2
  fi
  mkdir -p "${CAPACITY_DIR}"
  /usr/bin/time -v -o "${CAPACITY_DIR}/command.time.txt" \
    "${PYTHON_BIN}" scripts/qm9_complete_total_capacity_train.py \
      --protocol "${PROTOCOL}" \
      --manifest "${MANIFEST_ROOT}/stable5_parent_manifest.json" \
      --direction-manifest "${DIRECTION_ROOT}/stable5/manifest.json" \
      --direction-manifest-sha256 "${DIRECTION_SHA}" \
      --direction-role all \
      --run "qm9_train800_egf_force1_s12330_rebuild_v1=${BASELINE_RUN}=${CHECKPOINT}" \
      --source-checkpoint-sha256 "${CHECKPOINT_SHA}" \
      --root-source-checkpoint-sha256 "${CHECKPOINT_SHA}" \
      --require-complete-total-relaxed-hvp \
      --analytic-relaxed-hvp \
      --symmetric-matrix-power-mode eigh_second_order_audit \
      --output-dir "${CAPACITY_DIR}" \
      --molecules 0028399 \
      --device cuda:0 \
      --seed 20260724 \
      --max-steps 2000 \
      --learning-rate 1e-5 \
      --directions-per-step 1 \
      --eval-interval 50 \
      --checkpoint-interval 50 \
      --early-stop-relative-frobenius 0.05 \
      --gradient-clip-norm 1.0 \
      --strict-active-density-refresh \
      --density-parameter-response-predictor \
      --connect-lagrange-multiplier-response \
      --implicit-density-parameter-response \
      --implicit-response-tolerance 3e-5 \
      --implicit-response-max-iterations 2500 \
      --implicit-response-damping 1e-8 \
      --implicit-response-solver direct \
      --implicit-response-warm-start \
      --base-initialization label_reference \
      --density-strict-threshold 1e-8 \
      --integral-directional-second-step 1e-4 \
      --analytic-response-residual-tolerance 1e-10 \
      > "${CAPACITY_DIR}/command.log" 2>&1
  "${PYTHON_BIN}" - "${CAPACITY_DIR}/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
if summary.get("stage1_gate_passed") is not True:
    raise SystemExit("single-parent capacity gate failed; stable5 remains locked")
PY
  touch "${CAPACITY_DIR}/PASSED"
) 9>"${GPU_LOCK}"

# Deliberately stop here. stable5 and train20 are separate registered stages and
# remain locked until the single-parent result has been reviewed and registered.
sha256sum \
  "${PROTOCOL}" \
  "${DATASET_ROOT}/provenance/train_only_dataset_manifest.json" \
  "${PBE_MANIFEST}" \
  "${MANIFEST_ROOT}/stable5_parent_manifest.json" \
  "${DIRECTION_ROOT}/stable5/manifest.json" \
  "${MANIFEST_ROOT}/train20_parent_manifest.json" \
  "${DIRECTION_ROOT}/train20/manifest.json" \
  "${BASELINE_RUN}/checkpoints/last.ckpt" \
  "${AUDIT_DIR}/summary.json" \
  "${CAPACITY_DIR}/summary.json" \
  > "${ARTIFACT_ROOT}/provenance/registered_sha256.txt"
