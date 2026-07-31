#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/shenwei01/xzh_node02_20260724
REPO="$ROOT/work/structures25"
DATASET_NAME=QM9PBEForceEGFH10MultiDir6V1
DATASET_ROOT="$ROOT/data/$DATASET_NAME"
ARTICLE_RUN="$ROOT/runs/qm9_graphformer_egfh10_article_warmstart_s100_v1"
ADAPTED_CHECKPOINT="$ARTICLE_RUN/article_weight_adapter/article_qm9_current_compat.ckpt"
EXPECTED_ADAPTED_SHA=aafbdb63edc0a34fa7687ca97b55b73aa2e2a3c3bf4e6ce2b255c5b3bbe347b1
TRAIN_NAME=qm9_graphformer_egfh10_multidir6_effbatch12_article_warmstart_s100_v1
TRAIN_ROOT="$ROOT/models/train/runs/$TRAIN_NAME"
RUN_ROOT="$ROOT/runs/qm9_graphformer_egfh10_multidir6_effbatch12_article_warmstart_s100_v1"
DEVICE="${EGFH10_DEVICE:-0}"

[[ "$(hostname)" == node02 ]] || {
  echo "This effective-batch-12 EGFH10 run is authorized on node02 only." >&2
  exit 2
}
for path in "$ROOT" "$REPO" "$DATASET_ROOT" "$TRAIN_ROOT" "$RUN_ROOT"; do
  if [[ "$path" == /scratch* ]]; then
    echo "Effective-batch-12 EGFH10 run must not use /scratch." >&2
    exit 2
  fi
done
[[ -f "$DATASET_ROOT/egfh10_manifest.json" ]] || {
  echo "Missing completed six-direction dataset." >&2
  exit 1
}
[[ "$(sha256sum "$ADAPTED_CHECKPOINT" | awk '{print $1}')" == \
  "$EXPECTED_ADAPTED_SHA" ]] || {
  echo "Adapted article checkpoint hash mismatch." >&2
  exit 1
}

mkdir -p "$TRAIN_ROOT" "$RUN_ROOT/logs" "$ROOT/tmp" "$ROOT/cache"
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
    data.datamodule.batch_size=12 \
    +data.datamodule.pairs_per_batch=3 \
    trainer.accumulate_grad_batches=1 \
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

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

tb_path = run_root / "tensorboard_summary.json"
tensorboard = json.loads(tb_path.read_text())
total = tensorboard["losses"]["total"]
if int(total["last"]["step"]) + 1 != 100 or int(total["count"]) != 100:
    raise SystemExit("unexpected effective-batch-12 TensorBoard range")

summary = {
    "protocol_id": (
        "qm9_graphformer_egfh10_multidir6_effbatch12_"
        "article_warmstart_s100_v1"
    ),
    "controlled_change": (
        "physical batch size 4 to 12 and complete pairs per batch 1 to 3"
    ),
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
    "physical_batch_size": 12,
    "complete_pairs_per_step": 3,
    "graphs_per_optimizer_step": 12,
    "gradient_accumulation": 1,
    "loss_weights": {"E": 0.1, "G": 0.8, "F": 1.0, "H": 0.01},
    "validation_accessed": False,
    "test100_accessed": False,
    "dataset_manifest": str(dataset_root / "egfh10_manifest.json"),
    "dataset_manifest_sha256": sha(dataset_root / "egfh10_manifest.json"),
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": sha(checkpoint),
    "train_root": str(train_root),
    "tensorboard_summary": str(tb_path),
}
(run_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
