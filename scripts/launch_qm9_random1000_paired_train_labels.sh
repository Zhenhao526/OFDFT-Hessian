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
export OMP_NUM_THREADS="${LABEL_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${LABEL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${LABEL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${LABEL_NUM_THREADS:-1}"

DATASET_NAME="QM9PBEForceRandom1000PairedTrain"
RAW_DIR="${DFT_DATA}/${DATASET_NAME}/raw"
NUM_PROCESSES="${LABEL_NUM_PROCESSES:-60}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${DFT_MODELS}/labelgen/runs/${DATASET_NAME}_${RUN_STAMP}}"
LOG_DIR="${RUN_ROOT}/logs"
export TMPDIR="${TMPDIR:-/scratch/xzh/tmp/${DATASET_NAME}_${RUN_STAMP}}"
mkdir -p "${LOG_DIR}" "${TMPDIR}"

raw_count="$(find "${RAW_DIR}" -maxdepth 1 -type l -name '*.xyz' | wc -l)"
if [[ "${raw_count}" -ne 800 ]]; then
  echo "ERROR: expected exactly 800 train-parent raw links, found ${raw_count}" >&2
  exit 1
fi
if [[ ! -f "${DFT_DATA}/${DATASET_NAME}/train_parent_manifest.json" ]]; then
  echo "ERROR: missing train-parent manifest" >&2
  exit 1
fi

common=(
  preset=qm9_pbe_force_random1000_paired_train
  "dataset.raw_data_dir=${RAW_DIR}"
  "n_molecules=-1"
  "start_idx=0"
  "num_processes=${NUM_PROCESSES}"
  "num_threads_per_process=${LABEL_NUM_THREADS:-1}"
  "max_memory_per_process=${MAX_MEMORY_PER_PROCESS:-6000}"
)

echo "run_root=${RUN_ROOT}"
echo "raw_parent_count=${raw_count}"
echo "expected_chk=1600 expected_labels=1600 processes=${NUM_PROCESSES}"

/usr/bin/time -v -o "${LOG_DIR}/kohn_sham.time.txt" \
  python -m mldft.datagen.kohn_sham_dataset \
    "${common[@]}" \
    "hydra.run.dir=${RUN_ROOT}/hydra/kohn_sham" \
    >"${LOG_DIR}/kohn_sham.log" 2>&1

/usr/bin/time -v -o "${LOG_DIR}/labelgen.time.txt" \
  python -m mldft.datagen.generate_labels_dataset \
    "${common[@]}" \
    "hydra.run.dir=${RUN_ROOT}/hydra/labelgen" \
    >"${LOG_DIR}/labelgen.log" 2>&1

chk_count="$(find "${DFT_DATA}/${DATASET_NAME}/kohn_sham" -maxdepth 1 -type f -name '*.chk' | wc -l)"
label_count="$(find "${DFT_DATA}/${DATASET_NAME}/labels" -maxdepth 1 -type f -name '*.zarr.zip' | wc -l)"
python scripts/check_qm9_force_smoke.py \
  "${DFT_DATA}/${DATASET_NAME}/labels" \
  --expected-molecules 800 \
  --expected-samples 2 \
  --summary-json "${RUN_ROOT}/force_check_summary.json" \
  >"${LOG_DIR}/force_check.log" 2>&1

python - <<PY >"${RUN_ROOT}/summary.json"
import json
from pathlib import Path

root = Path("${DFT_DATA}/${DATASET_NAME}")
manifest = json.loads((root / "train_parent_manifest.json").read_text())
summary = {
    "dataset": "${DATASET_NAME}",
    "run_root": "${RUN_ROOT}",
    "raw_parent_count": ${raw_count},
    "chk_count": ${chk_count},
    "label_count": ${label_count},
    "expected_labels": 1600,
    "num_processes": ${NUM_PROCESSES},
    "paired_perturbations": True,
    "samples_per_parent": 2,
    "parent_overlap": manifest.get("overlap", {}),
}
print(json.dumps(summary, indent=2))
PY
cat "${RUN_ROOT}/summary.json"

