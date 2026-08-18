#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/shenwei01/xzh_node02_20260724
REPO="$ROOT/work/structures25"
DATASET="$ROOT/data/QM9PBEForceEGFH10ScratchV1"
REFERENCE_ROOT="$ROOT/artifacts/graphformer_hybrid_relaxed_hvp_rebuild_v1/pbe_hessian_train20"
REFERENCE_DIR="$REFERENCE_ROOT/cache"
REFERENCE_MANIFEST="$REFERENCE_ROOT/manifest.json"
TRAIN_ROOT="$ROOT/models/train/runs/qm9_graphformer_egfh10_force_secant_article_warmstart_ultralow_lr_tail_s1300_v1"
CHECKPOINT="$TRAIN_ROOT/checkpoints/last.ckpt"
EXPECTED_CHECKPOINT_SHA=7c40f4dfb442f606a278b796e973d2669108b3a306a6b1e065143ee9e3e71545
RUN_NAME=EGFH10ArticleWarmS1300
MOLECULE_ID=0016298
OUT_ROOT="$ROOT/runs/qm9_graphformer_egfh10_total_hessian_vibration_final_s1300_v1"
MOLECULE_OUT="$OUT_ROOT/per_molecule/$MOLECULE_ID"
WARM100_ROOT="$ROOT/runs/qm9_graphformer_egfh10_total_hessian_vibration_article_warmstart_s100_v1"
DEVICE="${EGFH10_HESSIAN_DEVICE:-0}"

[[ "$(hostname)" == node02 ]] || {
  echo "This final step-1300 Hessian comparison is authorized on node02 only." >&2
  exit 2
}
for path in \
  "$ROOT" "$REPO" "$DATASET" "$REFERENCE_ROOT" "$TRAIN_ROOT" \
  "$OUT_ROOT" "$WARM100_ROOT"; do
  if [[ "$path" == /scratch* ]]; then
    echo "Final step-1300 Hessian comparison must not use /scratch." >&2
    exit 2
  fi
done
[[ "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" == "$EXPECTED_CHECKPOINT_SHA" ]] || {
  echo "Final step-1300 checkpoint hash mismatch." >&2
  exit 1
}
[[ -f "$WARM100_ROOT/per_molecule/$MOLECULE_ID/summary.json" ]] || {
  echo "Missing completed article warm-start step-100 Hessian result." >&2
  exit 1
}
[[ -f "$WARM100_ROOT/vibrational_0016298/summary.json" ]] || {
  echo "Missing completed article warm-start step-100 vibrational result." >&2
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

python - "$WARM100_ROOT" "$OUT_ROOT" "$EXPECTED_CHECKPOINT_SHA" <<'PY'
import json
import sys
from pathlib import Path

warm100_root = Path(sys.argv[1])
final_root = Path(sys.argv[2])
checkpoint_sha = sys.argv[3]
molecule_id = "0016298"

warm100_hessian = json.loads(
    (warm100_root / "per_molecule" / molecule_id / "summary.json").read_text()
)["metric_rows"][0]
final_hessian = json.loads(
    (final_root / "per_molecule" / molecule_id / "summary.json").read_text()
)["metric_rows"][0]
warm100_vibration = json.loads(
    (warm100_root / "vibrational_0016298" / "summary.json").read_text()
)["rows"][0]
final_vibration = json.loads(
    (final_root / "vibrational_0016298" / "summary.json").read_text()
)["rows"][0]

def comparison(key: str, before: dict, after: dict) -> dict:
    old = float(before[key])
    new = float(after[key])
    return {
        "article_warmstart_s100": old,
        "final_s1300": new,
        "absolute_change": new - old,
        "relative_change": None if old == 0.0 else (new - old) / old,
    }

summary = {
    "definition": (
        "Same-molecule strict density-relaxed total-OFDFT Hessian and "
        "vibrational comparison between released-article warm-start "
        "optimizer steps 100 and converged step 1300."
    ),
    "molecule_id": molecule_id,
    "final_s1300_checkpoint_sha256": checkpoint_sha,
    "hessian": {
        "relative_frobenius": comparison(
            "relative_fro_error", warm100_hessian, final_hessian
        ),
        "element_mae_ha_per_bohr2": comparison(
            "mae", warm100_hessian, final_hessian
        ),
        "element_rmse_ha_per_bohr2": comparison(
            "rmse", warm100_hessian, final_hessian
        ),
        "element_max_abs_ha_per_bohr2": comparison(
            "max_abs_error", warm100_hessian, final_hessian
        ),
        "antisymmetric_over_symmetric_fro": float(
            final_hessian["antisymmetric_over_symmetric_fro"]
        ),
        "model_symmetry_max_abs_error": float(
            final_hessian["model_symmetry_max_abs_error"]
        ),
    },
    "vibration": {
        "frequency_mae_cm-1": comparison(
            "frequency_mae_cm-1", warm100_vibration, final_vibration
        ),
        "frequency_rmse_cm-1": comparison(
            "frequency_rmse_cm-1", warm100_vibration, final_vibration
        ),
        "frequency_max_abs_cm-1": comparison(
            "frequency_max_abs_cm-1", warm100_vibration, final_vibration
        ),
        "mean_mode_overlap": comparison(
            "mean_mode_overlap", warm100_vibration, final_vibration
        ),
        "model_imaginary_modes": int(
            final_vibration["model_imaginary_modes"]
        ),
        "pbe_imaginary_modes": int(final_vibration["pbe_imaginary_modes"]),
    },
    "density_convergence": {
        "strict_points": int(final_hessian["strict_points"]),
        "total_displaced_points": int(
            final_hessian["total_displaced_points"]
        ),
        "max_displaced_gradient_norm": float(
            final_hessian["max_displaced_gradient_norm"]
        ),
        "mean_displaced_cycles": float(
            final_hessian["mean_displaced_cycles"]
        ),
        "wall_time_s": float(final_hessian["wall_time_s"]),
    },
    "test100_accessed": False,
    "test100_evaluations_used": 0,
}
(final_root / "comparison_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
