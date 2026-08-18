#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/shenwei01/xzh_node02_20260724
REPO="$ROOT/work/structures25"
DATASET="$ROOT/data/QM9PBEForceEGFH10ScratchV1"
REFERENCE_ROOT="$ROOT/artifacts/graphformer_hybrid_relaxed_hvp_rebuild_v1/pbe_hessian_train20"
REFERENCE_DIR="$REFERENCE_ROOT/cache"
REFERENCE_MANIFEST="$REFERENCE_ROOT/manifest.json"
TRAIN_ROOT="$ROOT/models/train/runs/qm9_graphformer_egfh10_force_secant_article_warmstart_s100_v1"
CHECKPOINT="$TRAIN_ROOT/checkpoints/last.ckpt"
EXPECTED_CHECKPOINT_SHA=26e244691456c42f2ed1000583da690a9b2b3eefdf1ddd22d9c94ea8b7dc8b85
RUN_NAME=EGFH10ArticleWarmS100
MOLECULE_ID=0016298
OUT_ROOT="$ROOT/runs/qm9_graphformer_egfh10_total_hessian_vibration_article_warmstart_s100_v1"
MOLECULE_OUT="$OUT_ROOT/per_molecule/$MOLECULE_ID"
SCRATCH_ROOT="$ROOT/runs/qm9_graphformer_egfh10_total_hessian_vibration_s100_v1"
DEVICE="${EGFH10_HESSIAN_DEVICE:-0}"

[[ "$(hostname)" == node02 ]] || {
  echo "This article-warm-start Hessian comparison is authorized on node02 only." >&2
  exit 2
}
for path in \
  "$ROOT" "$REPO" "$DATASET" "$REFERENCE_ROOT" "$TRAIN_ROOT" \
  "$OUT_ROOT" "$SCRATCH_ROOT"; do
  if [[ "$path" == /scratch* ]]; then
    echo "Article-warm-start Hessian comparison must not use /scratch." >&2
    exit 2
  fi
done
[[ "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" == "$EXPECTED_CHECKPOINT_SHA" ]] || {
  echo "Article-warm-start checkpoint hash mismatch." >&2
  exit 1
}
[[ -f "$SCRATCH_ROOT/per_molecule/$MOLECULE_ID/summary.json" ]] || {
  echo "Missing completed scratch step-100 Hessian result." >&2
  exit 1
}
[[ -f "$SCRATCH_ROOT/vibrational_0016298/summary.json" ]] || {
  echo "Missing completed scratch step-100 vibrational result." >&2
  exit 1
}

mkdir -p "$MOLECULE_OUT" "$OUT_ROOT/logs" "$ROOT/tmp" "$ROOT/cache"
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

hessian_npz="$MOLECULE_OUT/${RUN_NAME}_${MOLECULE_ID}_0000000_total_hessian.npz"
if [[ ! -f "$MOLECULE_OUT/summary.json" || ! -f "$hessian_npz" ]]; then
  /usr/bin/time -v -o "$OUT_ROOT/logs/${MOLECULE_ID}.time.txt" \
    python scripts/qm9_total_ofdft_hessian_audit.py \
      --dataset-dir "$DATASET" \
      --reference-dir "$REFERENCE_DIR" \
      --run "${RUN_NAME}=${TRAIN_ROOT}=${CHECKPOINT}" \
      --molecules "$MOLECULE_ID" \
      --sample-id 0 \
      --output-dir "$MOLECULE_OUT" \
      --displacement 1e-4 \
      --integral-derivative-step 1e-4 \
      --integral-derivative-workers 4 \
      --model-geometry-derivative autograd \
      --base-initialization label_reference \
      --optimizer adam \
      --lr 1e-3 \
      --max-cycle 1000 \
      --convergence-tolerance 1e-2 \
      --fallback-optimizer adam \
      --fallback-lr 3e-4 \
      --fallback-max-cycle 10000 \
      --fallback-convergence-tolerance 1e-5 \
      --fallback-always \
      --lbfgs-refine \
      --newton-refine \
      --device cuda:0 \
      --transform-device cpu \
      >"$OUT_ROOT/logs/${MOLECULE_ID}.log" 2>&1
fi

python scripts/qm9_hessian_vibrational_metrics.py \
  --manifest-json "$REFERENCE_MANIFEST" \
  --dataset-dir "$DATASET" \
  --result-json "${RUN_NAME}=$MOLECULE_OUT/summary.json" \
  --output-dir "$OUT_ROOT/vibrational_0016298" \
  >"$OUT_ROOT/logs/vibrational_0016298.log" 2>&1

python - "$SCRATCH_ROOT" "$OUT_ROOT" "$EXPECTED_CHECKPOINT_SHA" <<'PY'
import json
import sys
from pathlib import Path

scratch_root = Path(sys.argv[1])
article_root = Path(sys.argv[2])
checkpoint_sha = sys.argv[3]
molecule_id = "0016298"

scratch_hessian = json.loads(
    (scratch_root / "per_molecule" / molecule_id / "summary.json").read_text()
)["metric_rows"][0]
article_hessian = json.loads(
    (article_root / "per_molecule" / molecule_id / "summary.json").read_text()
)["metric_rows"][0]
scratch_vibration = json.loads(
    (scratch_root / "vibrational_0016298" / "summary.json").read_text()
)["rows"][0]
article_vibration = json.loads(
    (article_root / "vibrational_0016298" / "summary.json").read_text()
)["rows"][0]

def comparison(key: str, before: dict, after: dict) -> dict:
    old = float(before[key])
    new = float(after[key])
    return {
        "scratch_s100": old,
        "article_warmstart_s100": new,
        "absolute_change": new - old,
        "relative_change": None if old == 0.0 else (new - old) / old,
    }

summary = {
    "definition": (
        "Same-molecule strict density-relaxed total-OFDFT Hessian and "
        "vibrational comparison between scratch and released-article-weight "
        "warm-start, both trained for 100 optimizer steps."
    ),
    "molecule_id": molecule_id,
    "article_warmstart_checkpoint_sha256": checkpoint_sha,
    "relative_frobenius": comparison(
        "relative_fro_error", scratch_hessian, article_hessian
    ),
    "frequency_mae_cm-1": comparison(
        "frequency_mae_cm-1", scratch_vibration, article_vibration
    ),
    "frequency_rmse_cm-1": comparison(
        "frequency_rmse_cm-1", scratch_vibration, article_vibration
    ),
    "frequency_max_abs_cm-1": comparison(
        "frequency_max_abs_cm-1", scratch_vibration, article_vibration
    ),
    "model_imaginary_modes": {
        "scratch_s100": int(scratch_vibration["model_imaginary_modes"]),
        "article_warmstart_s100": int(
            article_vibration["model_imaginary_modes"]
        ),
        "pbe": int(article_vibration["pbe_imaginary_modes"]),
    },
    "mean_mode_overlap": comparison(
        "mean_mode_overlap", scratch_vibration, article_vibration
    ),
    "article_density_convergence": {
        "strict_points": int(article_hessian["strict_points"]),
        "total_displaced_points": int(
            article_hessian["total_displaced_points"]
        ),
        "max_displaced_gradient_norm": float(
            article_hessian["max_displaced_gradient_norm"]
        ),
    },
    "test100_accessed": False,
    "test100_evaluations_used": 0,
}
(article_root / "comparison_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
