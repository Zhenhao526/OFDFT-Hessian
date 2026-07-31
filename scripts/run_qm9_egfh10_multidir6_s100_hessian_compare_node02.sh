#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/shenwei01/xzh_node02_20260724
REPO="$ROOT/work/structures25"
DATASET="$ROOT/data/QM9PBEForceEGFH10MultiDir6V1"
REFERENCE_ROOT="$ROOT/artifacts/graphformer_hybrid_relaxed_hvp_rebuild_v1/pbe_hessian_train20"
REFERENCE_DIR="$REFERENCE_ROOT/cache"
REFERENCE_MANIFEST="$REFERENCE_ROOT/manifest.json"
TRAIN_ROOT="${EGFH10_MULTIDIR_TRAIN_ROOT:-$ROOT/models/train/runs/qm9_graphformer_egfh10_multidir6_article_warmstart_s100_v1}"
TRAIN_SUMMARY="${EGFH10_MULTIDIR_TRAIN_SUMMARY:-$ROOT/runs/qm9_graphformer_egfh10_multidir6_article_warmstart_s100_v1/summary.json}"
CHECKPOINT="$TRAIN_ROOT/checkpoints/last.ckpt"
MOLECULE_ID=0016298
BASELINE_ROOT="$ROOT/runs/qm9_graphformer_egfh10_total_hessian_vibration_article_warmstart_s100_v1"
DEVICE="${EGFH10_HESSIAN_DEVICE:-0}"

[[ "$(hostname)" == node02 ]] || {
  echo "This EGFH10 multi-direction Hessian comparison is authorized on node02 only." >&2
  exit 2
}
[[ -f "$TRAIN_SUMMARY" && -f "$CHECKPOINT" ]] || {
  echo "Missing completed multi-direction training result." >&2
  exit 1
}
EXPECTED_CHECKPOINT_SHA="$(
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_sha256"])' \
    "$TRAIN_SUMMARY"
)"
FINAL_STEP="$(
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["final_optimizer_step"])' \
    "$TRAIN_SUMMARY"
)"
RUN_NAME="${EGFH10_MULTIDIR_RUN_NAME:-EGFH10MultiDir6S${FINAL_STEP}}"
OUT_ROOT="${EGFH10_MULTIDIR_HESSIAN_OUT:-$ROOT/runs/qm9_graphformer_egfh10_multidir6_total_hessian_vibration_s${FINAL_STEP}_v1}"
MOLECULE_OUT="$OUT_ROOT/per_molecule/$MOLECULE_ID"
for path in \
  "$ROOT" "$REPO" "$DATASET" "$REFERENCE_ROOT" "$TRAIN_ROOT" \
  "$OUT_ROOT" "$BASELINE_ROOT"; do
  if [[ "$path" == /scratch* ]]; then
    echo "EGFH10 multi-direction Hessian comparison must not use /scratch." >&2
    exit 2
  fi
done
[[ "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" == \
  "$EXPECTED_CHECKPOINT_SHA" ]] || {
  echo "Multi-direction checkpoint hash mismatch." >&2
  exit 1
}
[[ -f "$BASELINE_ROOT/per_molecule/$MOLECULE_ID/summary.json" ]] || {
  echo "Missing one-direction step-100 Hessian baseline." >&2
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

python - \
  "$BASELINE_ROOT" "$OUT_ROOT" "$EXPECTED_CHECKPOINT_SHA" "$FINAL_STEP" <<'PY'
import json
import sys
from pathlib import Path

baseline_root = Path(sys.argv[1])
candidate_root = Path(sys.argv[2])
checkpoint_sha = sys.argv[3]
final_step = int(sys.argv[4])
molecule_id = "0016298"

baseline_hessian = json.loads(
    (baseline_root / "per_molecule" / molecule_id / "summary.json").read_text()
)["metric_rows"][0]
candidate_hessian = json.loads(
    (candidate_root / "per_molecule" / molecule_id / "summary.json").read_text()
)["metric_rows"][0]
baseline_vibration = json.loads(
    (baseline_root / "vibrational_0016298" / "summary.json").read_text()
)["rows"][0]
candidate_vibration = json.loads(
    (candidate_root / "vibrational_0016298" / "summary.json").read_text()
)["rows"][0]

def comparison(key: str, before: dict, after: dict) -> dict:
    old = float(before[key])
    new = float(after[key])
    return {
        "one_direction_s100": old,
        f"six_directions_s{final_step}": new,
        "absolute_change": new - old,
        "relative_change": None if old == 0.0 else (new - old) / old,
    }

summary = {
    "definition": (
        "One-versus-six paired-direction comparison with identical ten "
        "parents, article weight-only initialization, E/G/F/H weights, and "
        "strict density-relaxed Hessian protocol. The six-direction candidate "
        f"was trained to its declared convergence point at step {final_step}."
    ),
    "molecule_id": molecule_id,
    "candidate_final_optimizer_step": final_step,
    "candidate_checkpoint_sha256": checkpoint_sha,
    "hessian": {
        "relative_frobenius": comparison(
            "relative_fro_error", baseline_hessian, candidate_hessian
        ),
        "element_mae_ha_per_bohr2": comparison(
            "mae", baseline_hessian, candidate_hessian
        ),
        "element_rmse_ha_per_bohr2": comparison(
            "rmse", baseline_hessian, candidate_hessian
        ),
        "element_max_abs_ha_per_bohr2": comparison(
            "max_abs_error", baseline_hessian, candidate_hessian
        ),
    },
    "vibration": {
        "frequency_mae_cm-1": comparison(
            "frequency_mae_cm-1", baseline_vibration, candidate_vibration
        ),
        "frequency_rmse_cm-1": comparison(
            "frequency_rmse_cm-1", baseline_vibration, candidate_vibration
        ),
        "mean_mode_overlap": comparison(
            "mean_mode_overlap", baseline_vibration, candidate_vibration
        ),
        "candidate_model_imaginary_modes": int(
            candidate_vibration["model_imaginary_modes"]
        ),
        "pbe_imaginary_modes": int(candidate_vibration["pbe_imaginary_modes"]),
    },
    "density_convergence": {
        "strict_points": int(candidate_hessian["strict_points"]),
        "total_displaced_points": int(
            candidate_hessian["total_displaced_points"]
        ),
        "max_displaced_gradient_norm": float(
            candidate_hessian["max_displaced_gradient_norm"]
        ),
        "mean_displaced_cycles": float(
            candidate_hessian["mean_displaced_cycles"]
        ),
        "wall_time_s": float(candidate_hessian["wall_time_s"]),
    },
    "test100_accessed": False,
    "test100_evaluations_used": 0,
}
(candidate_root / "comparison_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
