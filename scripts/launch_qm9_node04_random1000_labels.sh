#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f "${ENV_FILE:-/scratch/xzh/env.sh}" ]]; then
  # shellcheck source=/dev/null
  source "${ENV_FILE:-/scratch/xzh/env.sh}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

export DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
export DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"

DATASET_NAME="${DATASET_NAME:-QM9PBEForceRandom1000}"
DATASET_FILENAME="${DATASET_FILENAME:-qm9_pbe_force_random1000}"
N_MOLECULES="${N_MOLECULES:-1000}"
SAMPLES_PER_MOLECULE="${SAMPLES_PER_MOLECULE:-4}"
RANDOM_SEED="${RANDOM_SEED:-20260709}"
LABEL_NUM_PROCESSES="${LABEL_NUM_PROCESSES:-20}"
LABEL_NUM_THREADS="${LABEL_NUM_THREADS:-1}"
MAX_MEMORY_PER_PROCESS="${MAX_MEMORY_PER_PROCESS:-4000}"
SOURCE_RAW_DIR="${SOURCE_RAW_DIR:-${DFT_DATA}/QM9/raw}"
SUBSET_ROOT="${SUBSET_ROOT:-${DFT_DATA}/QM9RandomSubsets/random${N_MOLECULES}_seed${RANDOM_SEED}}"
SUBSET_RAW_DIR="${SUBSET_RAW_DIR:-${SUBSET_ROOT}/raw}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${DFT_MODELS}/labelgen/runs/${DATASET_NAME}_${RUN_STAMP}}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
TMPDIR="${TMPDIR:-/scratch/xzh/tmp/${DATASET_NAME}_${RUN_STAMP}}"

mkdir -p "${RUN_ROOT}" "${LOG_DIR}" "${TMPDIR}" "${DFT_DATA}" "${DFT_MODELS}"
export TMPDIR
export OMP_NUM_THREADS="${LABEL_NUM_THREADS}"
export MKL_NUM_THREADS="${LABEL_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${LABEL_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${LABEL_NUM_THREADS}"

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
ideal_h = avg_h_per_100_mol_4proc * (n_molecules / 100.0) * (4.0 / num_processes)
print("QM9 random force-label generation estimate")
print(f"dataset: ${DATASET_NAME}")
print(f"random seed: ${RANDOM_SEED}")
print(f"molecules: {n_molecules}, samples/molecule: ${SAMPLES_PER_MOLECULE}")
print(f"expected labels: {n_molecules * int('${SAMPLES_PER_MOLECULE}')}")
print(f"label processes: {num_processes}")
print(f"ideal estimate from P1 batches: {ideal_h:.1f} h")
print(f"conservative 1.5-2.5x budget: {ideal_h*1.5:.1f}-{ideal_h*2.5:.1f} h")
print("recommended Slurm wall time: 48 h")
PY
}

echo "host: $(hostname)"
echo "date: $(date)"
echo "root: ${ROOT_DIR}"
echo "DFT_DATA=${DFT_DATA}"
echo "SOURCE_RAW_DIR=${SOURCE_RAW_DIR}"
echo "SUBSET_RAW_DIR=${SUBSET_RAW_DIR}"
echo "DATASET_NAME=${DATASET_NAME}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "TMPDIR=${TMPDIR}"

if [[ ! -d "${SOURCE_RAW_DIR}" ]]; then
  echo "ERROR: source QM9 raw dir does not exist: ${SOURCE_RAW_DIR}" >&2
  exit 1
fi

write_estimate

run_timed "00_prepare_random_subset" \
  "${PYTHON_BIN}" scripts/prepare_qm9_random_subset.py \
  --source-raw-dir "${SOURCE_RAW_DIR}" \
  --output-raw-dir "${SUBSET_RAW_DIR}" \
  --n-molecules "${N_MOLECULES}" \
  --seed "${RANDOM_SEED}" \
  --manifest-json "${RUN_ROOT}/random_subset_manifest.json" \
  --overwrite

COMMON_DATAGEN_OVERRIDES=(
  "preset=qm9_pbe_force_full"
  "dataset.name=${DATASET_NAME}"
  "dataset.filename=${DATASET_FILENAME}"
  "start_idx=0"
  "n_molecules=${N_MOLECULES}"
  "num_processes=${LABEL_NUM_PROCESSES}"
  "num_threads_per_process=${LABEL_NUM_THREADS}"
  "max_memory_per_process=${MAX_MEMORY_PER_PROCESS}"
  "dataset.raw_data_dir=${SUBSET_RAW_DIR}"
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

find "${DFT_DATA}/${DATASET_NAME}" -maxdepth 2 -type f | sort > "${RUN_ROOT}/artifact_files.txt"
du -sh "${DFT_DATA}/${DATASET_NAME}" "${SUBSET_ROOT}" 2>/dev/null | tee "${RUN_ROOT}/artifact_sizes.txt"

cat > "${RUN_ROOT}/summary.txt" <<EOF
QM9 random1000 force-label generation complete
dataset=${DATASET_NAME}
data_dir=${DFT_DATA}/${DATASET_NAME}
subset_raw_dir=${SUBSET_RAW_DIR}
run_root=${RUN_ROOT}
expected_labels=$((N_MOLECULES * SAMPLES_PER_MOLECULE))
EOF

cat "${RUN_ROOT}/summary.txt"
