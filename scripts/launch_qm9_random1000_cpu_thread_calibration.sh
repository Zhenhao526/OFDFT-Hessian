#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
source "${ENV_FILE:-/scratch/xzh/env.sh}"
export DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
export DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET_DIR="${DFT_DATA}/QM9PBEForceRandom1000"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-${DFT_MODELS}/eval/qm9_random1000_hessian_protocol_scan/20260714_194632/representative_manifest_8mol.json}"
EGF_RUN_DIR="${EGF_RUN_DIR:-${DFT_MODELS}/train/runs/qm9_random1000_egf_lam1_e10_20260713_162829}"
EGF_CKPT="${EGF_CKPT:-${EGF_RUN_DIR}/checkpoints/epoch_009.ckpt}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${DFT_MODELS}/eval/qm9_random1000_cpu_thread_calibration/${RUN_STAMP}}"
mkdir -p "${OUT_DIR}/logs"

"${PYTHON_BIN}" - "${SOURCE_MANIFEST}" "${OUT_DIR}/manifest.json" <<'PY'
import json
import sys

rows = json.load(open(sys.argv[1]))
row = next(row for row in rows if row["molecule_id"] == "0040728")
open(sys.argv[2], "w").write(json.dumps([row], indent=2) + "\n")
PY

run_case() {
  local gpu="$1" threads="$2" name="threads${2}" root="${OUT_DIR}/threads${2}"
  mkdir -p "${root}/hessians"
  local start
  start="$(date +%s)"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  OMP_NUM_THREADS="${threads}" MKL_NUM_THREADS="${threads}" \
  OPENBLAS_NUM_THREADS="${threads}" NUMEXPR_NUM_THREADS="${threads}" \
    "${PYTHON_BIN}" scripts/qm9_hessian_density_relaxed_eval.py \
      --manifest-json "${OUT_DIR}/manifest.json" --dataset-dir "${DATASET_DIR}" \
      --run "EGF_w0p1=${EGF_RUN_DIR}=${EGF_CKPT}" \
      --output-json "${root}/summary.json" --output-csv "${root}/molecules.csv" \
      --optimization-csv "${root}/optimizations.csv" --hessian-npz-dir "${root}/hessians" \
      --max-molecules 1 --displacement 1e-3 \
      --initialization sad_default --base-density-warm-start \
      --optimizer adam --lr 1e-3 --max-cycle 1000 --convergence-tolerance 1e-2 \
      --fallback-optimizer adam --fallback-lr 3e-4 --fallback-max-cycle 10000 \
      --fallback-convergence-tolerance 1e-4 --fallback-always --device cuda:0 \
      >"${OUT_DIR}/logs/${name}.log" 2>&1
  echo "$(($(date +%s)-start))" > "${root}/wall_seconds.txt"
}

pids=()
for spec in "0:1" "1:4" "2:8"; do
  IFS=: read -r gpu threads <<<"${spec}"
  run_case "${gpu}" "${threads}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done

"${PYTHON_BIN}" - "${OUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
rows = []
reference = None
for threads in (1, 4, 8):
    case = root / f"threads{threads}"
    payload = json.load(open(case / "summary.json"))
    row = payload["rows"][0]
    hessian = np.load(row["hessian_npz"])["density_relaxed_hessian"]
    if reference is None:
        reference = hessian
    rows.append(
        {
            "threads": threads,
            "wall_s": int((case / "wall_seconds.txt").read_text()),
            "success": row["success"],
            "strict_points": row["n_converged"],
            "optimization_points": row["n_optimizations"],
            "mae": row["mae"],
            "relative_fro_vs_threads1": float(
                np.linalg.norm(hessian - reference) / np.linalg.norm(reference)
            ),
        }
    )
baseline = rows[0]["wall_s"]
for row in rows:
    row["speedup_vs_threads1"] = baseline / row["wall_s"]
result = {
    "definition": "same EGF_w0p1/0040728 workload with 1, 4, or 8 BLAS/OpenMP threads",
    "rows": rows,
}
(root / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
PY

echo "CPU-thread calibration complete: ${OUT_DIR}"
