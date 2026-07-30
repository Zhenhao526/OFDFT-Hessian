#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/shenwei01/xzh_node02_20260724
REPO="$ROOT/work/structures25"
DATASET_NAME=QM9PBEForceEGFH10ScratchV1
DATASET_ROOT="$ROOT/data/$DATASET_NAME"
BASELINE_RUN="$ROOT/runs/qm9_graphformer_egfh10_article_warmstart_s200_v1"
BASELINE_TENSORBOARD="$BASELINE_RUN/tensorboard_summary.json"
BASELINE_CHECKPOINT="$ROOT/models/train/runs/qm9_graphformer_egfh10_force_secant_article_warmstart_s200_v1/checkpoints/last.ckpt"
EXPECTED_BASELINE_SHA=0b67be97d9793af912ba53aaef28b525669c3fab0e316da2bd7efdad0e1955fb
RUN_ROOT="$ROOT/runs/qm9_graphformer_egfh10_article_warmstart_convergence_v1"
DEVICE="${EGFH10_DEVICE:-0}"
START_STEP=300
STEP_INCREMENT=100
MAX_STEP="${EGFH10_MAX_STEP:-1000}"
RELATIVE_TOLERANCE="${EGFH10_RELATIVE_TOLERANCE:-0.02}"

[[ "$(hostname)" == node02 ]] || {
  echo "This convergence continuation is authorized on node02 only." >&2
  exit 2
}
for path in "$ROOT" "$REPO" "$DATASET_ROOT" "$BASELINE_RUN" "$RUN_ROOT"; do
  if [[ "$path" == /scratch* ]]; then
    echo "Convergence continuation must not read or write /scratch." >&2
    exit 2
  fi
done
[[ -f "$BASELINE_TENSORBOARD" ]] || {
  echo "Missing step-200 TensorBoard summary." >&2
  exit 1
}
[[ "$(sha256sum "$BASELINE_CHECKPOINT" | awk '{print $1}')" == "$EXPECTED_BASELINE_SHA" ]] || {
  echo "Step-200 baseline checkpoint hash mismatch." >&2
  exit 1
}
[[ -f "$DATASET_ROOT/egfh10_manifest.json" ]] || {
  echo "Missing EGFH10 dataset manifest." >&2
  exit 1
}
if (( MAX_STEP < START_STEP || MAX_STEP % STEP_INCREMENT != 0 )); then
  echo "EGFH10_MAX_STEP must be a multiple of 100 and at least 300." >&2
  exit 2
fi

mkdir -p "$RUN_ROOT" "$ROOT/tmp" "$ROOT/cache"
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

summary_paths=("$BASELINE_TENSORBOARD")
for ((target_step = START_STEP; target_step <= MAX_STEP; target_step += STEP_INCREMENT)); do
  source_step=$((target_step - STEP_INCREMENT))
  if (( source_step == 200 )); then
    source_checkpoint="$BASELINE_CHECKPOINT"
    expected_source_sha="$EXPECTED_BASELINE_SHA"
  else
    source_checkpoint="$ROOT/models/train/runs/qm9_graphformer_egfh10_force_secant_article_warmstart_convergence_s${source_step}_v1/checkpoints/last.ckpt"
    source_summary="$RUN_ROOT/s${source_step}/summary.json"
    [[ -f "$source_summary" ]] || {
      echo "Missing source-stage summary: $source_summary" >&2
      exit 1
    }
    expected_source_sha="$(
      python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_sha256"])' \
        "$source_summary"
    )"
  fi
  [[ -f "$source_checkpoint" ]] || {
    echo "Missing source checkpoint: $source_checkpoint" >&2
    exit 1
  }
  [[ "$(sha256sum "$source_checkpoint" | awk '{print $1}')" == "$expected_source_sha" ]] || {
    echo "Source checkpoint hash mismatch at step $source_step." >&2
    exit 1
  }

  train_name="qm9_graphformer_egfh10_force_secant_article_warmstart_convergence_s${target_step}_v1"
  train_root="$ROOT/models/train/runs/$train_name"
  stage_root="$RUN_ROOT/s${target_step}"
  mkdir -p "$train_root" "$stage_root/logs"

  resume_checkpoint="$source_checkpoint"
  if [[ -f "$train_root/checkpoints/last.ckpt" ]]; then
    resume_checkpoint="$train_root/checkpoints/last.ckpt"
  fi

  if [[ ! -f "$stage_root/summary.json" ]]; then
    if [[ ! -f "$stage_root/tensorboard_summary.json" ]]; then
      python -m mldft.ml.train \
        experiment=str25/qm9_pbe_force_egfh10_force_secant_scratch_v1 \
        "name=$train_name" \
        "hydra.run.dir=$train_root" \
        seed=20260730 \
        "ckpt_path=$resume_checkpoint" \
        weight_ckpt_path=null \
        "trainer.max_steps=$target_step" \
        trainer.accelerator=gpu trainer.devices=1 trainer.strategy=auto \
        callbacks.model_checkpoint.every_n_train_steps=20 \
        extras.enforce_tags=false extras.print_config=false \
        hydra.callbacks.git_logging.clean=false \
        >"$stage_root/logs/train.log" 2>&1

      python scripts/summarize_qm9_egfh10_tensorboard.py "$train_root" \
        >"$stage_root/tensorboard_summary.json"
    fi

    python - \
      "$DATASET_ROOT" "$source_checkpoint" "$source_step" \
      "$train_root" "$stage_root" "$target_step" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

dataset_root = Path(sys.argv[1])
source_checkpoint = Path(sys.argv[2])
source_step = int(sys.argv[3])
train_root = Path(sys.argv[4])
stage_root = Path(sys.argv[5])
target_step = int(sys.argv[6])
checkpoint = train_root / "checkpoints" / "last.ckpt"
if not checkpoint.is_file():
    raise SystemExit(f"missing final step-{target_step} checkpoint")

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

tb_path = stage_root / "tensorboard_summary.json"
tensorboard_summary = json.loads(tb_path.read_text())
total = tensorboard_summary["losses"]["total"]
last_logged_step = int(total["last"]["step"])
count = int(total["count"])
if last_logged_step + 1 != target_step or count != target_step - source_step:
    raise SystemExit(
        f"expected {target_step - source_step} logs ending at "
        f"{target_step - 1}; found {count} ending at {last_logged_step}"
    )

summary = {
    "protocol_id": (
        "qm9_graphformer_egfh10_force_secant_article_warmstart_"
        f"convergence_s{target_step}_v1"
    ),
    "initialization": "continued_article_warmstart_trajectory",
    "source_optimizer_step": source_step,
    "source_checkpoint": str(source_checkpoint),
    "source_checkpoint_sha256": sha(source_checkpoint),
    "ckpt_path": str(source_checkpoint),
    "weight_ckpt_path": None,
    "optimizer_state_restored": True,
    "scheduler_state_restored": True,
    "final_optimizer_step": target_step,
    "final_logged_step": last_logged_step,
    "additional_optimizer_steps": target_step - source_step,
    "molecule_count": 10,
    "geometry_count": 30,
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
(stage_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
  fi

  summary_paths+=("$stage_root/tensorboard_summary.json")
  python scripts/analyze_qm9_egfh10_loss_convergence.py \
    --relative-tolerance "$RELATIVE_TOLERANCE" \
    "${summary_paths[@]}" >"$RUN_ROOT/convergence_status.json"
  python -m json.tool "$RUN_ROOT/convergence_status.json"

  converged="$(
    python -c 'import json,sys; print("yes" if json.load(open(sys.argv[1]))["converged"] else "no")' \
      "$RUN_ROOT/convergence_status.json"
  )"
  if [[ "$converged" == yes ]]; then
    echo "Training-loss convergence criterion met at step $target_step."
    exit 0
  fi
done

echo "Training-loss convergence criterion was not met by step $MAX_STEP." >&2
exit 3
