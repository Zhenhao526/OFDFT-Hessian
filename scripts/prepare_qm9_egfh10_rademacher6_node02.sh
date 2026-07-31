#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/shenwei01/xzh_node02_20260724
REPO="$ROOT/work/structures25"
SOURCE_DATASET="$ROOT/data/QM9PBEForceEGFH10ScratchV1"
DATASET_NAME=QM9PBEForceEGFH10Rademacher6V1
DATASET_ROOT="$ROOT/data/$DATASET_NAME"
RAW_SOURCE="$ROOT/runtime_parent/_runtime/qm9_p0/QM9/raw"
RUN_ROOT="$ROOT/runs/qm9_egfh10_rademacher6_prepare_v1"
IDS=(
  0016298 0028399 0064547 0124009 0132608
  0016142 0031108 0096630 0132419 0049017
)

[[ "$(hostname)" == node02 ]] || {
  echo "This Rademacher EGFH10 preparation is authorized on node02 only." >&2
  exit 2
}
for path in "$ROOT" "$REPO" "$SOURCE_DATASET" "$DATASET_ROOT" "$RUN_ROOT"; do
  if [[ "$path" == /scratch* ]]; then
    echo "Rademacher EGFH10 preparation must not use /scratch." >&2
    exit 2
  fi
done

mkdir -p \
  "$DATASET_ROOT/raw" \
  "$DATASET_ROOT/kohn_sham" \
  "$DATASET_ROOT/labels" \
  "$DATASET_ROOT/labels_local_frames_global_symmetric_natrep" \
  "$RUN_ROOT/logs" \
  "$ROOT/tmp" \
  "$ROOT/cache"
cd "$REPO"
source "$REPO/scripts/activate_qm9_node02_local.sh"
source "$REPO/.venv/bin/activate"
export DFT_DATA="$ROOT/data"
export DFT_MODELS="$ROOT/models"
export TMPDIR="$ROOT/tmp"
export XDG_CACHE_HOME="$ROOT/cache"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

for molecule_id in "${IDS[@]}"; do
  source_name="$(printf 'dsgdb9nsd_%06d.xyz' "$((10#$molecule_id))")"
  source_path="$RAW_SOURCE/$source_name"
  destination="$DATASET_ROOT/raw/$source_name"
  [[ -f "$source_path" ]] || {
    echo "Missing frozen QM9 raw geometry: $source_path" >&2
    exit 1
  }
  [[ -e "$destination" ]] || ln -s "$source_path" "$destination"

  sample_id=0000000
  old_chk="$SOURCE_DATASET/kohn_sham/qm9_pbe_force_egfh10_scratch_v1_${molecule_id}.${sample_id}.chk"
  new_chk="$DATASET_ROOT/kohn_sham/qm9_pbe_force_egfh10_rademacher6_v1_${molecule_id}.${sample_id}.chk"
  [[ -f "$new_chk" ]] || cp -p "$old_chk" "$new_chk"
done

if [[ "$(find "$DATASET_ROOT/kohn_sham" -maxdepth 1 -type f -name '*.chk' | wc -l)" != 130 ]]; then
  python -m mldft.datagen.kohn_sham_dataset \
    preset=qm9_pbe_force_egfh10_rademacher6_v1 \
    "dataset.raw_data_dir=$DATASET_ROOT/raw" \
    n_molecules=-1 start_idx=0 num_processes=10 \
    num_threads_per_process=1 max_memory_per_process=6000 \
    "hydra.run.dir=$RUN_ROOT/hydra/kohn_sham" \
    >"$RUN_ROOT/logs/kohn_sham.log" 2>&1
fi

if [[ "$(find "$DATASET_ROOT/labels" -maxdepth 1 -type f -name '*.zarr.zip' | wc -l)" != 130 ]]; then
  python -m mldft.datagen.generate_labels_dataset \
    preset=qm9_pbe_force_egfh10_rademacher6_v1 \
    "dataset.raw_data_dir=$DATASET_ROOT/raw" \
    n_molecules=-1 start_idx=0 num_processes=10 \
    num_threads_per_process=1 max_memory_per_process=6000 \
    "hydra.run.dir=$RUN_ROOT/hydra/labelgen" \
    >"$RUN_ROOT/logs/labelgen.log" 2>&1
fi

python scripts/check_qm9_force_smoke.py \
  "$DATASET_ROOT/labels" \
  --expected-molecules 10 \
  --expected-samples 13 \
  --summary-json "$RUN_ROOT/force_check_summary.json" \
  >"$RUN_ROOT/logs/force_check.log" 2>&1

if [[ "$(find "$DATASET_ROOT/labels_local_frames_global_symmetric_natrep" \
  -maxdepth 1 -type f -name '*.zarr.zip' | wc -l)" != 130 ]]; then
  python -m mldft.datagen.transform_dataset \
    data=qm9_pbe_force_full \
    "data.dataset_name=$DATASET_NAME" \
    extras.enforce_tags=false extras.print_config=false \
    +hydra.callbacks.git_logging.clean=false \
    +num_processes=10 +num_threads_per_process=1 \
    +start_idx=0 +num_molecules=999999 \
    "hydra.run.dir=$RUN_ROOT/hydra/transform" \
    >"$RUN_ROOT/logs/transform.log" 2>&1
fi

python scripts/prepare_qm9_egfh10_train_only_split.py \
  --dataset-root "$DATASET_ROOT" \
  --dataset-name "$DATASET_NAME" \
  --parent-count 10 \
  --pairs-per-parent 6 \
  >"$RUN_ROOT/logs/split.log" 2>&1

STAT_PATH="$DATASET_ROOT/dataset_statistics/dataset_statistics_labels_local_frames_global_symmetric_natrep_e_kin_plus_xc.zarr"
if [[ ! -f "$STAT_PATH/.zgroup" ]]; then
  python -m mldft.ml.compute_dataset_statistics \
    data=qm9_pbe_force_full \
    "data.dataset_name=$DATASET_NAME" \
    "name=${DATASET_NAME}_statistics" \
    extras.enforce_tags=false extras.print_config=false \
    +hydra.callbacks.git_logging.clean=false \
    overwrite=false \
    data.datamodule.train_only=true \
    data.datamodule.batch_size=12 \
    data.datamodule.num_workers=0 \
    statistic_fitter_kwargs.n_batches=null \
    "hydra.run.dir=$RUN_ROOT/hydra/statistics" \
    >"$RUN_ROOT/logs/statistics.log" 2>&1
fi

python - "$DATASET_ROOT" "$RUN_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

dataset_root, run_root = map(Path, sys.argv[1:])

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

manifest = json.loads((dataset_root / "egfh10_manifest.json").read_text())
summary = {
    "protocol_id": "qm9_egfh10_rademacher6_prepare_v1",
    "dataset_name": "QM9PBEForceEGFH10Rademacher6V1",
    "parent_count": 10,
    "directions_per_parent": 6,
    "samples_per_parent": 13,
    "perturbation_distribution": "independent_coordinatewise_rademacher",
    "coordinate_values": [-1, 1],
    "coordinate_amplitude_angstrom": 0.01,
    "remove_translation": False,
    "paired_endpoints": True,
    "dataset_manifest": str(dataset_root / "egfh10_manifest.json"),
    "dataset_manifest_sha256": sha(dataset_root / "egfh10_manifest.json"),
    "split_sha256": manifest["split_sha256"],
    "validation_accessed": False,
    "test100_accessed": False,
}
(run_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
