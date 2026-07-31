#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/shenwei01/xzh_node02_20260724
REPO="$ROOT/work/structures25"
OLD_DATASET="$ROOT/data/QM9PBEForceEGFH10ScratchV1"
DATASET_NAME=QM9PBEForceEGFH10MultiDir6V1
DATASET_ROOT="$ROOT/data/$DATASET_NAME"
RAW_SOURCE="$ROOT/runtime_parent/_runtime/qm9_p0/QM9/raw"
RUN_ROOT="$ROOT/runs/qm9_graphformer_egfh10_multidir6_article_warmstart_s100_v1"
TRAIN_NAME=qm9_graphformer_egfh10_multidir6_article_warmstart_s100_v1
TRAIN_ROOT="$ROOT/models/train/runs/$TRAIN_NAME"
ARTICLE_RUN="$ROOT/runs/qm9_graphformer_egfh10_article_warmstart_s100_v1"
ADAPTED_CHECKPOINT="$ARTICLE_RUN/article_weight_adapter/article_qm9_current_compat.ckpt"
EXPECTED_ADAPTED_SHA=aafbdb63edc0a34fa7687ca97b55b73aa2e2a3c3bf4e6ce2b255c5b3bbe347b1
DEVICE="${EGFH10_DEVICE:-0}"
IDS=(
  0016298 0028399 0064547 0124009 0132608
  0016142 0031108 0096630 0132419 0049017
)

[[ "$(hostname)" == node02 ]] || {
  echo "This EGFH10 multi-direction run is authorized on node02 only." >&2
  exit 2
}
for path in \
  "$ROOT" "$REPO" "$OLD_DATASET" "$DATASET_ROOT" "$RUN_ROOT" "$TRAIN_ROOT"; do
  if [[ "$path" == /scratch* ]]; then
    echo "EGFH10 multi-direction run must not use /scratch." >&2
    exit 2
  fi
done
[[ "$(sha256sum "$ADAPTED_CHECKPOINT" | awk '{print $1}')" == \
  "$EXPECTED_ADAPTED_SHA" ]] || {
  echo "Adapted article checkpoint hash mismatch." >&2
  exit 1
}

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
  if [[ ! -e "$destination" ]]; then
    ln -s "$source_path" "$destination"
  fi

  for sample_id in 0000000 0000001 0000002; do
    old_chk="$OLD_DATASET/kohn_sham/qm9_pbe_force_egfh10_scratch_v1_${molecule_id}.${sample_id}.chk"
    new_chk="$DATASET_ROOT/kohn_sham/qm9_pbe_force_egfh10_multidir6_v1_${molecule_id}.${sample_id}.chk"
    old_label="$OLD_DATASET/labels/${molecule_id}.${sample_id}.zarr.zip"
    new_label="$DATASET_ROOT/labels/${molecule_id}.${sample_id}.zarr.zip"
    old_transformed="$OLD_DATASET/labels_local_frames_global_symmetric_natrep/${molecule_id}.${sample_id}.zarr.zip"
    new_transformed="$DATASET_ROOT/labels_local_frames_global_symmetric_natrep/${molecule_id}.${sample_id}.zarr.zip"
    [[ -f "$new_chk" ]] || cp -p "$old_chk" "$new_chk"
    [[ -f "$new_label" ]] || cp -p "$old_label" "$new_label"
    [[ -f "$new_transformed" ]] || cp -p "$old_transformed" "$new_transformed"
  done
done

if [[ "$(find "$DATASET_ROOT/kohn_sham" -maxdepth 1 -type f -name '*.chk' | wc -l)" != 130 ]]; then
  python -m mldft.datagen.kohn_sham_dataset \
    preset=qm9_pbe_force_egfh10_multidir6_v1 \
    "dataset.raw_data_dir=$DATASET_ROOT/raw" \
    n_molecules=-1 start_idx=0 num_processes=10 \
    num_threads_per_process=1 max_memory_per_process=6000 \
    "hydra.run.dir=$RUN_ROOT/hydra/kohn_sham" \
    >"$RUN_ROOT/logs/kohn_sham.log" 2>&1
fi

if [[ "$(find "$DATASET_ROOT/labels" -maxdepth 1 -type f -name '*.zarr.zip' | wc -l)" != 130 ]]; then
  python -m mldft.datagen.generate_labels_dataset \
    preset=qm9_pbe_force_egfh10_multidir6_v1 \
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
    "weight_ckpt_path=$ADAPTED_CHECKPOINT" \
    "data.dataset_name=$DATASET_NAME" \
    "data.datamodule.pair_source_markers=[$DATASET_NAME]" \
    trainer.max_steps=100 \
    trainer.accelerator=gpu trainer.devices=1 trainer.strategy=auto \
    callbacks.model_checkpoint.every_n_train_steps=20 \
    extras.enforce_tags=false extras.print_config=false \
    hydra.callbacks.git_logging.clean=false \
    >"$RUN_ROOT/logs/train.log" 2>&1
fi

python scripts/summarize_qm9_egfh10_tensorboard.py "$TRAIN_ROOT" \
  >"$RUN_ROOT/tensorboard_summary.json"

python - \
  "$DATASET_ROOT" "$ADAPTED_CHECKPOINT" "$TRAIN_ROOT" "$RUN_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

dataset_root, source_checkpoint, train_root, run_root = map(Path, sys.argv[1:])
checkpoint = train_root / "checkpoints" / "last.ckpt"
if not checkpoint.is_file():
    raise SystemExit("missing final multi-direction checkpoint")

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

tensorboard = json.loads((run_root / "tensorboard_summary.json").read_text())
last_logged_step = int(tensorboard["losses"]["total"]["last"]["step"])
if last_logged_step + 1 != 100:
    raise SystemExit(f"expected global step 100, found {last_logged_step + 1}")

summary = {
    "protocol_id": "qm9_graphformer_egfh10_multidir6_article_warmstart_s100_v1",
    "controlled_change": "one to six paired displacement directions per parent",
    "initialization": "released_article_qm9_weight_only_warmstart",
    "source_checkpoint": str(source_checkpoint),
    "source_checkpoint_sha256": sha(source_checkpoint),
    "ckpt_path": None,
    "weight_ckpt_path": str(source_checkpoint),
    "optimizer_state_restored": False,
    "scheduler_state_restored": False,
    "final_optimizer_step": 100,
    "molecule_count": 10,
    "directions_per_molecule": 6,
    "geometry_count": 130,
    "loss_weights": {"E": 0.1, "G": 0.8, "F": 1.0, "H": 0.01},
    "validation_accessed": False,
    "test100_accessed": False,
    "dataset_manifest": str(dataset_root / "egfh10_manifest.json"),
    "dataset_manifest_sha256": sha(dataset_root / "egfh10_manifest.json"),
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": sha(checkpoint),
    "train_root": str(train_root),
    "tensorboard_summary": str(run_root / "tensorboard_summary.json"),
}
(run_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
