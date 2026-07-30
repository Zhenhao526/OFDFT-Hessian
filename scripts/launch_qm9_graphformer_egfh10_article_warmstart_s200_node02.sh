#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/shenwei01/xzh_node02_20260724
REPO="$ROOT/work/structures25"
DATASET_NAME=QM9PBEForceEGFH10ScratchV1
DATASET_ROOT="$ROOT/data/$DATASET_NAME"
SOURCE_NAME=qm9_graphformer_egfh10_force_secant_article_warmstart_s100_v1
SOURCE_ROOT="$ROOT/models/train/runs/$SOURCE_NAME"
SOURCE_CHECKPOINT="$SOURCE_ROOT/checkpoints/last.ckpt"
EXPECTED_SOURCE_SHA=26e244691456c42f2ed1000583da690a9b2b3eefdf1ddd22d9c94ea8b7dc8b85
TRAIN_NAME=qm9_graphformer_egfh10_force_secant_article_warmstart_s200_v1
TRAIN_ROOT="$ROOT/models/train/runs/$TRAIN_NAME"
RUN_ROOT="$ROOT/runs/qm9_graphformer_egfh10_article_warmstart_s200_v1"
DEVICE="${EGFH10_DEVICE:-0}"

[[ "$(hostname)" == node02 ]] || {
  echo "This article-warm-start continuation is authorized on node02 only." >&2
  exit 2
}
for path in \
  "$ROOT" "$REPO" "$DATASET_ROOT" "$SOURCE_ROOT" "$TRAIN_ROOT" "$RUN_ROOT"; do
  if [[ "$path" == /scratch* ]]; then
    echo "Article-warm-start continuation must not read or write /scratch." >&2
    exit 2
  fi
done
[[ -f "$SOURCE_CHECKPOINT" ]] || {
  echo "Missing source step-100 checkpoint: $SOURCE_CHECKPOINT" >&2
  exit 1
}
[[ "$(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}')" == "$EXPECTED_SOURCE_SHA" ]] || {
  echo "Source step-100 checkpoint hash mismatch." >&2
  exit 1
}
[[ -f "$DATASET_ROOT/egfh10_manifest.json" ]] || {
  echo "Missing EGFH10 dataset manifest." >&2
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

resume_checkpoint="$SOURCE_CHECKPOINT"
if [[ -f "$TRAIN_ROOT/checkpoints/last.ckpt" ]]; then
  resume_checkpoint="$TRAIN_ROOT/checkpoints/last.ckpt"
fi

if [[ ! -f "$RUN_ROOT/summary.json" ]]; then
  if [[ ! -f "$RUN_ROOT/tensorboard_summary.json" ]]; then
    python -m mldft.ml.train \
      experiment=str25/qm9_pbe_force_egfh10_force_secant_scratch_v1 \
      "name=$TRAIN_NAME" \
      "hydra.run.dir=$TRAIN_ROOT" \
      seed=20260730 \
      "ckpt_path=$resume_checkpoint" \
      weight_ckpt_path=null \
      trainer.max_steps=200 \
      trainer.accelerator=gpu trainer.devices=1 trainer.strategy=auto \
      callbacks.model_checkpoint.every_n_train_steps=20 \
      extras.enforce_tags=false extras.print_config=false \
      hydra.callbacks.git_logging.clean=false \
      >"$RUN_ROOT/logs/train.log" 2>&1

    python scripts/summarize_qm9_egfh10_tensorboard.py "$TRAIN_ROOT" \
      >"$RUN_ROOT/tensorboard_summary.json"
  fi

  python - \
    "$DATASET_ROOT" "$SOURCE_CHECKPOINT" "$TRAIN_ROOT" "$RUN_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

dataset_root, source_checkpoint, train_root, run_root = map(Path, sys.argv[1:])
checkpoint = train_root / "checkpoints" / "last.ckpt"
if not checkpoint.is_file():
    raise SystemExit("missing final step-200 EGFH10 checkpoint")

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

tensorboard_summary = json.loads(
    (run_root / "tensorboard_summary.json").read_text()
)
last_logged_step = int(tensorboard_summary["losses"]["total"]["last"]["step"])
global_step = last_logged_step + 1
if global_step != 200:
    raise SystemExit(
        f"expected final logged step 199/global step 200, found {last_logged_step}"
    )

summary = {
    "protocol_id": (
        "qm9_graphformer_egfh10_force_secant_article_warmstart_s200_v1"
    ),
    "initialization": "continued_article_warmstart_trajectory",
    "source_optimizer_step": 100,
    "source_checkpoint": str(source_checkpoint),
    "source_checkpoint_sha256": sha(source_checkpoint),
    "ckpt_path": str(source_checkpoint),
    "weight_ckpt_path": None,
    "optimizer_state_restored": True,
    "scheduler_state_restored": True,
    "final_optimizer_step": global_step,
    "final_logged_step": last_logged_step,
    "additional_optimizer_steps": global_step - 100,
    "molecule_count": 10,
    "geometry_count": 30,
    "losses": {
        "E": "Structures25 e_kin_plus_xc energy label",
        "G": "projected dE_model/dc versus gradient_label",
        "F": "scalar-derived force versus PBE force_label",
        "H": "paired-displacement conservative-force secant",
    },
    "loss_weights": {"E": 0.1, "G": 0.8, "F": 1.0, "H": 0.01},
    "optimizer_lr_at_initialization": 7.0e-5,
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
fi

python -m json.tool "$RUN_ROOT/summary.json"
