#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 START_IDX N_MOLECULES [NUM_PROCESSES]" >&2
  exit 2
fi

START_IDX="$1"
N_MOLECULES="$2"
NUM_PROCESSES="${3:-${NUM_PROCESSES:-16}}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
RAW_DATA_DIR="${RAW_DATA_DIR:-${DFT_DATA}/QM9/raw}"
TMPDIR="${TMPDIR:-/scratch/xzh/tmp/qm9_full_labels_${START_IDX}_${N_MOLECULES}}"
MAX_MEMORY_PER_PROCESS="${MAX_MEMORY_PER_PROCESS:-5000}"
NUM_THREADS_PER_PROCESS="${NUM_THREADS_PER_PROCESS:-1}"

mkdir -p "${TMPDIR}" "${DFT_DATA}"

export TMPDIR
export OMP_NUM_THREADS="${NUM_THREADS_PER_PROCESS}"
export MKL_NUM_THREADS="${NUM_THREADS_PER_PROCESS}"
export OPENBLAS_NUM_THREADS="${NUM_THREADS_PER_PROCESS}"
export NUMEXPR_NUM_THREADS="${NUM_THREADS_PER_PROCESS}"

PYTHON_BIN="${PYTHON_BIN:-python}"

COMMON_OVERRIDES=(
  "preset=qm9_pbe_force_full"
  "start_idx=${START_IDX}"
  "n_molecules=${N_MOLECULES}"
  "num_processes=${NUM_PROCESSES}"
  "num_threads_per_process=${NUM_THREADS_PER_PROCESS}"
  "max_memory_per_process=${MAX_MEMORY_PER_PROCESS}"
  "dataset.raw_data_dir=${RAW_DATA_DIR}"
)

echo "[qm9-full-labels] start_idx=${START_IDX} n_molecules=${N_MOLECULES} num_processes=${NUM_PROCESSES}"
echo "[qm9-full-labels] DFT_DATA=${DFT_DATA}"
echo "[qm9-full-labels] RAW_DATA_DIR=${RAW_DATA_DIR}"
echo "[qm9-full-labels] TMPDIR=${TMPDIR}"

"${PYTHON_BIN}" -m mldft.datagen.kohn_sham_dataset "${COMMON_OVERRIDES[@]}"
"${PYTHON_BIN}" -m mldft.datagen.generate_labels_dataset "${COMMON_OVERRIDES[@]}"

echo "[qm9-full-labels] completed shard start_idx=${START_IDX} n_molecules=${N_MOLECULES}"
