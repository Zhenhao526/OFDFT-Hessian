#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/shenwei01/xzh_node02_20260724
REPO="$ROOT/work/structures25"
DATASET_NAME=QM9PBEForceEGFH10ScratchV1
DATASET_ROOT="$ROOT/data/$DATASET_NAME"
ARTICLE_ROOT="$ROOT/runtime_parent/_runtime/models/train/runs/trained-on-qm9"
ARTICLE_CHECKPOINT="$ARTICLE_ROOT/checkpoints/last.ckpt"
EXPECTED_ARTICLE_SHA=9759da26660c619de9c3bbf4c2dc164343ee90e08e22b3fcdcc9682dacb9bd09
TRAIN_NAME=qm9_graphformer_egfh10_force_secant_article_warmstart_s100_v1
TRAIN_ROOT="$ROOT/models/train/runs/$TRAIN_NAME"
RUN_ROOT="$ROOT/runs/qm9_graphformer_egfh10_article_warmstart_s100_v1"
ADAPTER_ROOT="$RUN_ROOT/article_weight_adapter"
ADAPTED_CHECKPOINT="$ADAPTER_ROOT/article_qm9_current_compat.ckpt"
ADAPTER_MANIFEST="$ADAPTER_ROOT/manifest.json"
DEVICE="${EGFH10_DEVICE:-0}"

[[ "$(hostname)" == node02 ]] || {
  echo "This article-weight warm-start test is authorized on node02 only." >&2
  exit 2
}
for path in \
  "$ROOT" "$REPO" "$DATASET_ROOT" "$ARTICLE_ROOT" "$TRAIN_ROOT" "$RUN_ROOT"; do
  if [[ "$path" == /scratch* ]]; then
    echo "Article-weight warm-start test must not read or write /scratch." >&2
    exit 2
  fi
done
[[ -f "$ARTICLE_ROOT/hparams.yaml" ]] || {
  echo "Missing released article QM9 hparams." >&2
  exit 1
}
[[ "$(sha256sum "$ARTICLE_CHECKPOINT" | awk '{print $1}')" == "$EXPECTED_ARTICLE_SHA" ]] || {
  echo "Released article QM9 checkpoint hash mismatch." >&2
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

if [[ ! -f "$ADAPTED_CHECKPOINT" || ! -f "$ADAPTER_MANIFEST" ]]; then
  python scripts/adapt_structures25_article_checkpoint.py \
    --source "$ARTICLE_CHECKPOINT" \
    --output "$ADAPTED_CHECKPOINT" \
    --manifest "$ADAPTER_MANIFEST" \
    --expected-source-sha256 "$EXPECTED_ARTICLE_SHA" \
    >"$RUN_ROOT/logs/article_weight_adapter.log" 2>&1
fi

if [[ ! -f "$RUN_ROOT/summary.json" ]]; then
  python -m mldft.ml.train \
    experiment=str25/qm9_pbe_force_egfh10_force_secant_scratch_v1 \
    "name=$TRAIN_NAME" \
    "hydra.run.dir=$TRAIN_ROOT" \
    seed=20260730 \
    ckpt_path=null \
    "weight_ckpt_path=$ADAPTED_CHECKPOINT" \
    trainer.max_steps=100 \
    trainer.accelerator=gpu trainer.devices=1 trainer.strategy=auto \
    callbacks.model_checkpoint.every_n_train_steps=20 \
    extras.enforce_tags=false extras.print_config=false \
    hydra.callbacks.git_logging.clean=false \
    >"$RUN_ROOT/logs/train.log" 2>&1

  python scripts/summarize_qm9_egfh10_tensorboard.py "$TRAIN_ROOT" \
    >"$RUN_ROOT/tensorboard_summary.json"

  python - \
    "$DATASET_ROOT" "$ARTICLE_ROOT" "$ARTICLE_CHECKPOINT" \
    "$ADAPTED_CHECKPOINT" "$ADAPTER_MANIFEST" "$TRAIN_ROOT" "$RUN_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    dataset_root,
    article_root,
    article_checkpoint,
    adapted_checkpoint,
    adapter_manifest,
    train_root,
    run_root,
) = map(
    Path, sys.argv[1:]
)
checkpoint = train_root / "checkpoints" / "last.ckpt"
if not checkpoint.is_file():
    raise SystemExit("missing final article-warm-start EGFH10 checkpoint")

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

tensorboard_summary = json.loads(
    (run_root / "tensorboard_summary.json").read_text()
)
last_logged_step = int(tensorboard_summary["losses"]["total"]["last"]["step"])
global_step = last_logged_step + 1
if global_step != 100:
    raise SystemExit(
        f"expected final logged step 99/global step 100, found {last_logged_step}"
    )

summary = {
    "protocol_id": (
        "qm9_graphformer_egfh10_force_secant_article_warmstart_s100_v1"
    ),
    "initialization": "released_article_qm9_weight_only_warmstart",
    "article_run": str(article_root),
    "article_hparams": str(article_root / "hparams.yaml"),
    "article_hparams_sha256": sha(article_root / "hparams.yaml"),
    "source_checkpoint": str(article_checkpoint),
    "source_checkpoint_sha256": sha(article_checkpoint),
    "adapted_weight_checkpoint": str(adapted_checkpoint),
    "adapted_weight_checkpoint_sha256": sha(adapted_checkpoint),
    "weight_adapter_manifest": str(adapter_manifest),
    "weight_adapter_manifest_sha256": sha(adapter_manifest),
    "ckpt_path": None,
    "weight_ckpt_path": str(adapted_checkpoint),
    "optimizer_state_restored": False,
    "scheduler_state_restored": False,
    "final_optimizer_step": global_step,
    "final_logged_step": last_logged_step,
    "molecule_count": 10,
    "geometry_count": 30,
    "losses": {
        "E": "Structures25 e_kin_plus_xc energy label",
        "G": "projected dE_model/dc versus gradient_label",
        "F": "scalar-derived force versus PBE force_label",
        "H": "paired-displacement conservative-force secant",
    },
    "loss_weights": {"E": 0.1, "G": 0.8, "F": 1.0, "H": 0.01},
    "optimizer_lr": 7.0e-5,
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
