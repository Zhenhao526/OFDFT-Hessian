#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/shenwei01/xzh_node02_20260724
REPO="$ROOT/work/structures25"
DATASET="$ROOT/data/QM9PBEForceEGFH10ScratchV1"
REFERENCE_ROOT="$ROOT/artifacts/graphformer_hybrid_relaxed_hvp_rebuild_v1/pbe_hessian_train20"
REFERENCE_DIR="$REFERENCE_ROOT/cache"
REFERENCE_MANIFEST="$REFERENCE_ROOT/manifest.json"
TRAIN_ROOT="$ROOT/models/train/runs/qm9_graphformer_egfh10_force_secant_scratch_s20_v1"
CHECKPOINT="$TRAIN_ROOT/checkpoints/last.ckpt"
RUN_NAME=EGFH10ScratchS20
OUT_ROOT="$ROOT/runs/qm9_graphformer_egfh10_total_hessian_vibration_s20_v1"
DEVICE="${EGFH10_HESSIAN_DEVICE:-0}"
IDS=(
  0016298 0028399 0064547 0124009 0132608
  0016142 0031108 0096630 0132419 0049017
)
EXPECTED_CHECKPOINT_SHA=ab7070d1739f57684a4955ebab9dcce903073b039721fdb391774b8989f71c1e

if [[ "$(hostname)" != node02 ]]; then
  echo "This EGFH10 full-Hessian evaluation is authorized on node02 only." >&2
  exit 2
fi
for path in "$ROOT" "$REPO" "$DATASET" "$REFERENCE_ROOT" "$TRAIN_ROOT" "$OUT_ROOT"; do
  if [[ "$path" == /scratch* ]]; then
    echo "EGFH10 evaluation must not read or write /scratch." >&2
    exit 2
  fi
done
[[ -f "$CHECKPOINT" ]] || {
  echo "Missing checkpoint: $CHECKPOINT" >&2
  exit 1
}
[[ "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" == "$EXPECTED_CHECKPOINT_SHA" ]] || {
  echo "Checkpoint hash mismatch." >&2
  exit 1
}
[[ -f "$REFERENCE_MANIFEST" ]] || {
  echo "Missing PBE Hessian manifest: $REFERENCE_MANIFEST" >&2
  exit 1
}

mkdir -p "$OUT_ROOT/per_molecule" "$OUT_ROOT/logs" "$ROOT/tmp" "$ROOT/cache"
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

python - "$DATASET" "$REFERENCE_MANIFEST" "${IDS[@]}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import zarr

dataset = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
ids = sys.argv[3:]
manifest = json.loads(manifest_path.read_text())
rows = manifest if isinstance(manifest, list) else manifest["parents"]
references = {
    str(row["molecule_id"]): row
    for row in rows
    if row.get("success", True) and int(row.get("sample_id", 0)) == 0
}
for molecule_id in ids:
    if molecule_id not in references:
        raise SystemExit(f"missing PBE Hessian reference for {molecule_id}")
    reference = references[molecule_id]
    cache = Path(reference.get("cache_path") or reference["pbe_hessian_path"])
    if not cache.is_file():
        raise SystemExit(f"missing PBE Hessian cache for {molecule_id}: {cache}")
    label = zarr.open(
        dataset / "labels" / f"{molecule_id}.0000000.zarr.zip", mode="r"
    )
    atomic_numbers = np.asarray(label["geometry/atomic_numbers"], dtype=np.int64)
    positions = np.asarray(label["geometry/atom_pos"], dtype=np.float64)
    direction_path = Path(
        "/home/shenwei01/xzh_node02_20260724/artifacts/"
        "graphformer_hybrid_relaxed_hvp_rebuild_v1/directions/train20/artifacts"
    ) / f"{molecule_id}.0000000.npz"
    direction = np.load(direction_path)
    if not np.array_equal(atomic_numbers, direction["atomic_numbers"]):
        raise SystemExit(f"atomic-number mismatch for {molecule_id}")
    if not np.allclose(positions, direction["positions_bohr"], atol=1e-12, rtol=0.0):
        raise SystemExit(f"center-geometry mismatch for {molecule_id}")
print(json.dumps({"geometry_preflight": "passed", "molecules": ids}, indent=2))
PY

for molecule_id in "${IDS[@]}"; do
  molecule_out="$OUT_ROOT/per_molecule/$molecule_id"
  hessian_npz="$molecule_out/${RUN_NAME}_${molecule_id}_0000000_total_hessian.npz"
  if [[ -f "$molecule_out/summary.json" && -f "$hessian_npz" ]]; then
    echo "Skipping completed molecule $molecule_id"
    continue
  fi
  mkdir -p "$molecule_out"
  /usr/bin/time -v -o "$OUT_ROOT/logs/${molecule_id}.time.txt" \
    python scripts/qm9_total_ofdft_hessian_audit.py \
      --dataset-dir "$DATASET" \
      --reference-dir "$REFERENCE_DIR" \
      --run "${RUN_NAME}=${TRAIN_ROOT}=${CHECKPOINT}" \
      --molecules "$molecule_id" \
      --sample-id 0 \
      --output-dir "$molecule_out" \
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
      >"$OUT_ROOT/logs/${molecule_id}.log" 2>&1
done

python - "$OUT_ROOT" "$RUN_NAME" "${IDS[@]}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
run_name = sys.argv[2]
ids = sys.argv[3:]
metric_rows = []
point_rows = []
sources = []
for molecule_id in ids:
    path = out / "per_molecule" / molecule_id / "summary.json"
    payload = json.loads(path.read_text())
    metric_rows.extend(payload["metric_rows"])
    point_rows.extend(payload["point_rows"])
    sources.append(str(path))
if len(metric_rows) != len(ids):
    raise SystemExit(f"expected {len(ids)} metric rows, found {len(metric_rows)}")
merged = {
    "definition": "Merged strict density-relaxed total-OFDFT EGFH10 Hessian evaluation.",
    "run": run_name,
    "molecules": ids,
    "metric_rows": metric_rows,
    "point_rows": point_rows,
    "source_summaries": sources,
    "test100_accessed": False,
    "test100_evaluations_used": 0,
}
(out / "merged_hessian_summary.json").write_text(
    json.dumps(merged, indent=2, sort_keys=True) + "\n"
)
PY

python scripts/qm9_hessian_vibrational_metrics.py \
  --manifest-json "$REFERENCE_MANIFEST" \
  --dataset-dir "$DATASET" \
  --result-json "${RUN_NAME}=$OUT_ROOT/merged_hessian_summary.json" \
  --output-dir "$OUT_ROOT/vibrational" \
  >"$OUT_ROOT/logs/vibrational.log" 2>&1

python - "$OUT_ROOT" "$EXPECTED_CHECKPOINT_SHA" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
checkpoint_sha = sys.argv[2]
hessian = json.loads((out / "merged_hessian_summary.json").read_text())
vibration = json.loads((out / "vibrational" / "summary.json").read_text())
rows = hessian["metric_rows"]
relative = np.asarray([float(row["relative_fro_error"]) for row in rows])
vibrational_rows = vibration["rows"]
per_molecule_mae = np.asarray(
    [float(row["frequency_mae_cm-1"]) for row in vibrational_rows]
)
matched_frequency_errors = []
for row in vibrational_rows:
    modes = np.load(row["matched_frequencies_npz"])
    matched_frequency_errors.append(
        np.asarray(modes["matched_model_frequencies_cm"], dtype=np.float64)
        - np.asarray(modes["matched_pbe_frequencies_cm"], dtype=np.float64)
    )
frequency_errors = np.concatenate(matched_frequency_errors)
final = {
    "definition": "Strict density-relaxed total-OFDFT Hessian and vibrational-frequency metrics.",
    "checkpoint_sha256": checkpoint_sha,
    "molecule_count": len(rows),
    "relative_frobenius": {
        "mean": float(relative.mean()),
        "median": float(np.median(relative)),
        "p90": float(np.quantile(relative, 0.9)),
        "max": float(relative.max()),
    },
    "frequency_error_cm-1": {
        "pooled_mode_mae": float(np.mean(np.abs(frequency_errors))),
        "pooled_mode_rmse": float(np.sqrt(np.mean(frequency_errors**2))),
        "pooled_mode_max_abs": float(np.max(np.abs(frequency_errors))),
        "per_molecule_mae_mean": float(per_molecule_mae.mean()),
        "per_molecule_mae_median": float(np.median(per_molecule_mae)),
        "per_molecule_mae_p90": float(np.quantile(per_molecule_mae, 0.9)),
        "per_molecule_mae_max": float(per_molecule_mae.max()),
    },
    "vibrational_summary": vibration["summaries"],
    "per_molecule_hessian_metrics": rows,
    "per_molecule_vibrational_metrics": vibration["rows"],
    "test100_accessed": False,
    "test100_evaluations_used": 0,
}
(out / "final_summary.json").write_text(
    json.dumps(final, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(final, indent=2, sort_keys=True))
print(
    "final_summary_sha256="
    + hashlib.sha256((out / "final_summary.json").read_bytes()).hexdigest()
)
PY
