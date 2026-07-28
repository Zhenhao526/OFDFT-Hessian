#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
source "${ENV_FILE:-/scratch/xzh/env.sh}"
export DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
export DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

PAIR_NAME="QM9PBEForceRandom1000PairedTrain"
AUG_NAME="QM9PBEForceRandom1000PairedAug"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${DFT_MODELS}/preprocess/runs/${AUG_NAME}_${RUN_STAMP}"
mkdir -p "${RUN_ROOT}/logs"

/usr/bin/time -v -o "${RUN_ROOT}/logs/transform.time.txt" \
  python -m mldft.datagen.transform_dataset \
    data=qm9_pbe_force_full \
    "data.dataset_name=${PAIR_NAME}" \
    extras.enforce_tags=false extras.print_config=false \
    +hydra.callbacks.git_logging.clean=false \
    "+num_processes=${TRANSFORM_NUM_PROCESSES:-60}" \
    +num_threads_per_process=1 +start_idx=0 +num_molecules=999999 \
    "hydra.run.dir=${RUN_ROOT}/hydra/transform" \
    >"${RUN_ROOT}/logs/transform.log" 2>&1

transformed_count="$(find "${DFT_DATA}/${PAIR_NAME}/labels_local_frames_global_symmetric_natrep" -maxdepth 1 -type f -name '*.zarr.zip' | wc -l)"
if [[ "${transformed_count}" -ne 1600 ]]; then
  echo "ERROR: expected 1600 transformed paired labels, found ${transformed_count}" >&2
  exit 1
fi

python scripts/prepare_qm9_paired_augmented_training.py \
  --base-root "${DFT_DATA}/QM9PBEForceRandom1000" \
  --paired-root "${DFT_DATA}/${PAIR_NAME}" \
  --output-root "${DFT_DATA}/${AUG_NAME}" \
  --expected-labels 1600 \
  >"${RUN_ROOT}/logs/prepare.log" 2>&1
cp "${DFT_DATA}/${AUG_NAME}/paired_augmentation_manifest.json" "${RUN_ROOT}/"
cat "${RUN_ROOT}/paired_augmentation_manifest.json"
