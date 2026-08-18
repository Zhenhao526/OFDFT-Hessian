#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/shenwei01/xzh_node02_20260724
REPO="$ROOT/work/structures25"
DATASET_NAME=QM9PBEForceEGFH10ScratchV1
DATASET_ROOT="$ROOT/data/$DATASET_NAME"
RAW_SOURCE="$ROOT/runtime_parent/_runtime/qm9_p0/QM9/raw"
RUN_ROOT="$ROOT/runs/qm9_graphformer_egfh10_scratch_v1"
TRAIN_NAME=qm9_graphformer_egfh10_force_secant_scratch_s20_v1
TRAIN_ROOT="$ROOT/models/train/runs/$TRAIN_NAME"
DEVICE="${EGFH10_DEVICE:-7}"
IDS=(
  0016298 0028399 0064547 0124009 0132608
  0016142 0031108 0096630 0132419 0049017
)

if [[ "$(hostname)" != node02 ]]; then
  echo "This EGFH10 pilot is authorized on node02 only." >&2
  exit 2
fi
for path in "$ROOT" "$REPO" "$DATASET_ROOT" "$RUN_ROOT" "$TRAIN_ROOT"; do
  if [[ "$path" == /scratch* ]]; then
    echo "EGFH10 pilot must not read or write /scratch." >&2
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
export CUDA_VISIBLE_DEVICES="$DEVICE"

for molecule_id in "${IDS[@]}"; do
  source_name="$(printf 'dsgdb9nsd_%06d.xyz' "$((10#$molecule_id))")"
  source_path="$RAW_SOURCE/$source_name"
  destination="$DATASET_ROOT/raw/$source_name"
  [[ -f "$source_path" ]] || {
    echo "Missing frozen QM9 raw geometry: $source_path" >&2
    exit 1
  }
  if [[ -L "$destination" ]]; then
    [[ "$(readlink -f "$destination")" == "$(readlink -f "$source_path")" ]] || {
      echo "Existing raw link points elsewhere: $destination" >&2
      exit 1
    }
  elif [[ -e "$destination" ]]; then
    echo "Refusing to replace existing raw path: $destination" >&2
    exit 1
  else
    ln -s "$source_path" "$destination"
  fi
done

if [[ "$(find "$DATASET_ROOT/kohn_sham" -maxdepth 1 -type f -name '*.chk' 2>/dev/null | wc -l)" != 30 ]]; then
  python -m mldft.datagen.kohn_sham_dataset \
    preset=qm9_pbe_force_egfh10_paired_v1 \
    "dataset.raw_data_dir=$DATASET_ROOT/raw" \
    n_molecules=-1 start_idx=0 num_processes=10 \
    num_threads_per_process=1 max_memory_per_process=6000 \
    "hydra.run.dir=$RUN_ROOT/hydra/kohn_sham" \
    >"$RUN_ROOT/logs/kohn_sham.log" 2>&1
fi

if [[ "$(find "$DATASET_ROOT/labels" -maxdepth 1 -type f -name '*.zarr.zip' 2>/dev/null | wc -l)" != 30 ]]; then
  python -m mldft.datagen.generate_labels_dataset \
    preset=qm9_pbe_force_egfh10_paired_v1 \
    "dataset.raw_data_dir=$DATASET_ROOT/raw" \
    n_molecules=-1 start_idx=0 num_processes=10 \
    num_threads_per_process=1 max_memory_per_process=6000 \
    "hydra.run.dir=$RUN_ROOT/hydra/labelgen" \
    >"$RUN_ROOT/logs/labelgen.log" 2>&1
fi

python scripts/check_qm9_force_smoke.py \
  "$DATASET_ROOT/labels" \
  --expected-molecules 10 \
  --expected-samples 3 \
  --summary-json "$RUN_ROOT/force_check_summary.json" \
  >"$RUN_ROOT/logs/force_check.log" 2>&1

if [[ "$(find "$DATASET_ROOT/labels_local_frames_global_symmetric_natrep" -maxdepth 1 -type f -name '*.zarr.zip' 2>/dev/null | wc -l)" != 30 ]]; then
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
    data.datamodule.batch_size=4 \
    data.datamodule.num_workers=0 \
    statistic_fitter_kwargs.n_batches=null \
    "hydra.run.dir=$RUN_ROOT/hydra/statistics" \
    >"$RUN_ROOT/logs/statistics.log" 2>&1
fi

if [[ ! -f "$TRAIN_ROOT/checkpoints/last.ckpt" ]]; then
  python -m mldft.ml.train \
    experiment=str25/qm9_pbe_force_egfh10_force_secant_scratch_v1 \
    "name=$TRAIN_NAME" \
    "hydra.run.dir=$TRAIN_ROOT" \
    seed=20260730 \
    ckpt_path=null \
    weight_ckpt_path=null \
    trainer.max_steps=20 \
    trainer.accelerator=gpu trainer.devices=1 trainer.strategy=auto \
    extras.enforce_tags=false extras.print_config=false \
    hydra.callbacks.git_logging.clean=false \
    >"$RUN_ROOT/logs/train.log" 2>&1
fi

python - "$DATASET_ROOT" "$TRAIN_ROOT" "$RUN_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

dataset_root, train_root, run_root = map(Path, sys.argv[1:])
checkpoint = train_root / "checkpoints/last.ckpt"
if not checkpoint.is_file():
    raise SystemExit("missing final scratch EGFH10 checkpoint")

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

summary = {
    "protocol_id": "qm9_graphformer_egfh10_force_secant_scratch_v1",
    "initialization": "fresh_hydra_model_instantiation",
    "source_checkpoint": None,
    "ckpt_path": None,
    "weight_ckpt_path": None,
    "molecule_count": 10,
    "geometry_count": 30,
    "losses": {
        "E": "Structures25 e_kin_plus_xc energy label",
        "G": "projected dE_model/dc versus gradient_label",
        "F": "scalar-derived force versus PBE force_label",
        "H": "paired-displacement conservative-force secant",
    },
    "loss_weights": {"E": 0.1, "G": 0.8, "F": 1.0, "H": 0.01},
    "optimizer_steps": 20,
    "validation_accessed": False,
    "test100_accessed": False,
    "dataset_manifest": str(dataset_root / "egfh10_manifest.json"),
    "dataset_manifest_sha256": sha(dataset_root / "egfh10_manifest.json"),
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": sha(checkpoint),
    "train_root": str(train_root),
}
(run_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
