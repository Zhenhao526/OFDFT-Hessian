#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/shenwei01/xzh_node02_20260724
REPO="$ROOT/work/structures25"
DATASET_NAME=QM9PBEForceEGFH10ScratchV1
DATASET_ROOT="$ROOT/data/$DATASET_NAME"
TAIL_ID="${EGFH10_TAIL_ID:-low_lr_tail}"
SOURCE_STEP="${EGFH10_SOURCE_STEP:-1000}"
SOURCE_RUN="${EGFH10_SOURCE_RUN:-$ROOT/runs/qm9_graphformer_egfh10_article_warmstart_convergence_v1/s1000}"
SOURCE_TENSORBOARD="$SOURCE_RUN/tensorboard_summary.json"
SOURCE_CHECKPOINT="${EGFH10_SOURCE_CHECKPOINT:-$ROOT/models/train/runs/qm9_graphformer_egfh10_force_secant_article_warmstart_convergence_s1000_v1/checkpoints/last.ckpt}"
EXPECTED_SOURCE_SHA="${EGFH10_EXPECTED_SOURCE_SHA:-0bb3be367853d3303e6867b3f1b279ee76084a248d3cfdadea08d63463391aae}"
RUN_ROOT="$ROOT/runs/qm9_graphformer_egfh10_article_warmstart_${TAIL_ID}_v1"
ADAPTER_ROOT="$RUN_ROOT/checkpoint_adapter"
ADAPTED_CHECKPOINT="$ADAPTER_ROOT/step${SOURCE_STEP}_${TAIL_ID}.ckpt"
ADAPTER_MANIFEST="$ADAPTER_ROOT/manifest.json"
DEVICE="${EGFH10_DEVICE:-0}"
TAIL_LR="${EGFH10_TAIL_LR:-7.0e-6}"
TAIL_T_MAX_EPOCHS="${EGFH10_TAIL_T_MAX_EPOCHS:-20}"
STEP_INCREMENT=100
START_STEP=$((SOURCE_STEP + STEP_INCREMENT))
MAX_STEP="${EGFH10_TAIL_MAX_STEP:-$((SOURCE_STEP + 200))}"
MAX_EPOCHS=$((MAX_STEP / 10))
RELATIVE_TOLERANCE="${EGFH10_RELATIVE_TOLERANCE:-0.02}"

[[ "$(hostname)" == node02 ]] || {
  echo "This low-LR tail is authorized on node02 only." >&2
  exit 2
}
for path in "$ROOT" "$REPO" "$DATASET_ROOT" "$SOURCE_RUN" "$RUN_ROOT"; do
  if [[ "$path" == /scratch* ]]; then
    echo "Low-LR convergence tail must not read or write /scratch." >&2
    exit 2
  fi
done
[[ -f "$SOURCE_TENSORBOARD" ]] || {
  echo "Missing step-$SOURCE_STEP TensorBoard summary." >&2
  exit 1
}
[[ "$(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}')" == "$EXPECTED_SOURCE_SHA" ]] || {
  echo "Step-$SOURCE_STEP checkpoint hash mismatch." >&2
  exit 1
}
[[ -f "$DATASET_ROOT/egfh10_manifest.json" ]] || {
  echo "Missing EGFH10 dataset manifest." >&2
  exit 1
}

mkdir -p "$RUN_ROOT" "$ADAPTER_ROOT" "$ROOT/tmp" "$ROOT/cache"
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
  python scripts/reset_qm9_egfh_checkpoint_cosine_tail.py \
    --source "$SOURCE_CHECKPOINT" \
    --output "$ADAPTED_CHECKPOINT" \
    --manifest "$ADAPTER_MANIFEST" \
    --expected-source-sha256 "$EXPECTED_SOURCE_SHA" \
    --expected-global-step "$SOURCE_STEP" \
    --lr "$TAIL_LR" \
    --t-max-epochs "$TAIL_T_MAX_EPOCHS" \
    >"$RUN_ROOT/checkpoint_adapter.log" 2>&1
fi

summary_paths=("$SOURCE_TENSORBOARD")
for ((target_step = START_STEP; target_step <= MAX_STEP; target_step += STEP_INCREMENT)); do
  source_step=$((target_step - STEP_INCREMENT))
  if (( source_step == SOURCE_STEP )); then
    source_checkpoint="$ADAPTED_CHECKPOINT"
    expected_source_sha="$(
      python -c 'import json,sys; print(json.load(open(sys.argv[1]))["adapted_checkpoint_sha256"])' \
        "$ADAPTER_MANIFEST"
    )"
  else
    source_checkpoint="$ROOT/models/train/runs/qm9_graphformer_egfh10_force_secant_article_warmstart_${TAIL_ID}_s${source_step}_v1/checkpoints/last.ckpt"
    source_summary="$RUN_ROOT/s${source_step}/summary.json"
    expected_source_sha="$(
      python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_sha256"])' \
        "$source_summary"
    )"
  fi
  [[ "$(sha256sum "$source_checkpoint" | awk '{print $1}')" == "$expected_source_sha" ]] || {
    echo "Source checkpoint hash mismatch at step $source_step." >&2
    exit 1
  }

  train_name="qm9_graphformer_egfh10_force_secant_article_warmstart_${TAIL_ID}_s${target_step}_v1"
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
        "trainer.max_epochs=$MAX_EPOCHS" \
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
      "$train_root" "$stage_root" "$target_step" "$ADAPTER_MANIFEST" \
      "$TAIL_ID" "$TAIL_LR" "$TAIL_T_MAX_EPOCHS" <<'PY'
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
adapter_manifest = Path(sys.argv[7])
tail_id = sys.argv[8]
tail_lr = float(sys.argv[9])
tail_t_max_epochs = int(sys.argv[10])
checkpoint = train_root / "checkpoints" / "last.ckpt"

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

tb_path = stage_root / "tensorboard_summary.json"
total = json.loads(tb_path.read_text())["losses"]["total"]
if int(total["last"]["step"]) + 1 != target_step or int(total["count"]) != 100:
    raise SystemExit(f"unexpected TensorBoard range for step {target_step}")
summary = {
    "protocol_id": (
        "qm9_graphformer_egfh10_force_secant_article_warmstart_"
        f"{tail_id}_s{target_step}_v1"
    ),
    "source_optimizer_step": source_step,
    "source_checkpoint": str(source_checkpoint),
    "source_checkpoint_sha256": sha(source_checkpoint),
    "optimizer_state_restored": True,
    "optimizer_moments_preserved": True,
    "scheduler_state_restored": True,
    "low_lr_tail": {
        "id": tail_id,
        "initial_lr": tail_lr,
        "t_max_epochs": tail_t_max_epochs,
    },
    "adapter_manifest": str(adapter_manifest),
    "adapter_manifest_sha256": sha(adapter_manifest),
    "final_optimizer_step": target_step,
    "final_logged_step": int(total["last"]["step"]),
    "molecule_count": 10,
    "geometry_count": 30,
    "loss_weights": {"E": 0.1, "G": 0.8, "F": 1.0, "H": 0.01},
    "validation_accessed": False,
    "test100_accessed": False,
    "dataset_manifest_sha256": sha(dataset_root / "egfh10_manifest.json"),
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": sha(checkpoint),
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
    echo "$TAIL_ID training-loss convergence criterion met at step $target_step."
    exit 0
  fi
done

echo "$TAIL_ID convergence criterion was not met by step $MAX_STEP." >&2
exit 3
