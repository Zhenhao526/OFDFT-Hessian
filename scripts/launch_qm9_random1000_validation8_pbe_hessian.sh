#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
source "${ENV_FILE:-/scratch/xzh/env.sh}"
export DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
export DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIR="${DFT_DATA}/QM9PBEForceRandom1000"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${DFT_MODELS}/eval/qm9_random1000_validation8_pbe_hessian/${RUN_STAMP}}"
CACHE_DIR="${CACHE_DIR:-${DFT_MODELS}/eval/qm9_random1000_pbe_hessian_refs/validation8_pbe_hessians}"
mkdir -p "${OUT_DIR}/chunks" "${OUT_DIR}/logs" "${CACHE_DIR}"

"${PYTHON_BIN}" scripts/select_qm9_split_representatives.py \
  --split-file "${DATASET_DIR}/split.pkl" \
  --data-dir "${DFT_DATA}" \
  --split val --count 8 \
  --output-json "${OUT_DIR}/validation_representatives_8.json" \
  --output-ids "${OUT_DIR}/validation_representatives_8.ids" \
  >"${OUT_DIR}/logs/select.log" 2>&1

mapfile -t molecule_ids < <(tr ', ' '\n\n' < "${OUT_DIR}/validation_representatives_8.ids" | sed '/^$/d')
if [[ "${#molecule_ids[@]}" != 8 ]]; then
  echo "ERROR: expected 8 validation representatives, got ${#molecule_ids[@]}" >&2
  exit 1
fi

pids=()
for gpu in "${!molecule_ids[@]}"; do
  molecule_id="${molecule_ids[$gpu]}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    /usr/bin/time -v -o "${OUT_DIR}/logs/${molecule_id}.time.txt" \
      "${PYTHON_BIN}" scripts/qm9_pbe_hessian_reference_set.py \
        --dataset-dir "${DATASET_DIR}" \
        --output-dir "${CACHE_DIR}" \
        --manifest-json "${OUT_DIR}/chunks/${molecule_id}.json" \
        --manifest-csv "${OUT_DIR}/chunks/${molecule_id}.csv" \
        --molecules "${molecule_id}" --max-molecules 1 --sample-id 0 \
        --split val --workers 1 --backend gpu4pyscf \
        >"${OUT_DIR}/logs/${molecule_id}.log" 2>&1
  ) &
  pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failures=$((failures + 1)); fi
done
if [[ "${failures}" != 0 ]]; then
  echo "ERROR: ${failures} validation PBE Hessian workers failed" >&2
  exit 1
fi

"${PYTHON_BIN}" - "${OUT_DIR}" <<'PY'
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for path in sorted((out / "chunks").glob("*.json")):
    rows.extend(json.load(open(path)))
rows.sort(key=lambda row: (int(row["natoms"]), row["molecule_id"]))
(out / "pbe_hessian_manifest_validation8_gpu4pyscf.json").write_text(
    json.dumps(rows, indent=2, sort_keys=True) + "\n"
)
fields = sorted({key for row in rows for key in row})
with (out / "pbe_hessian_manifest_validation8_gpu4pyscf.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
summary = {
    "split": "val",
    "requested": len(rows),
    "success": sum(bool(row.get("success")) for row in rows),
    "failed": sum(not bool(row.get("success")) for row in rows),
    "elapsed_sum_s": sum(float(row.get("elapsed_s") or 0.0) for row in rows),
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
if summary["requested"] != 8 or summary["failed"]:
    raise SystemExit(1)
PY

echo "Validation PBE Hessians complete: ${OUT_DIR}"
