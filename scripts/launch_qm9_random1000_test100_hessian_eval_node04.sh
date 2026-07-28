#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f "${ENV_FILE:-/scratch/xzh/env.sh}" ]]; then
  # shellcheck source=/dev/null
  source "${ENV_FILE:-/scratch/xzh/env.sh}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

export DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
export DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"
export DATASET_NAME="${DATASET_NAME:-QM9PBEForceRandom1000}"
export CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${DFT_MODELS}/eval/qm9_random1000_test100_hessian_node04/${RUN_STAMP}}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
CHUNK_DIR="${OUT_DIR}/hessian_chunks"
mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${CHUNK_DIR}"

DATASET_DIR="${DFT_DATA}/${DATASET_NAME}"
EG_RUN_DIR="${EG_RUN_DIR:-${DFT_MODELS}/train/runs/qm9_random1000_eg_e10_20260713_162829}"
EG_CKPT="${EG_CKPT:-${EG_RUN_DIR}/checkpoints/epoch_009.ckpt}"
EGF_RUN_DIR="${EGF_RUN_DIR:-${DFT_MODELS}/train/runs/qm9_random1000_egf_lam1_e10_20260713_162829}"
EGF_CKPT="${EGF_CKPT:-${EGF_RUN_DIR}/checkpoints/epoch_009.ckpt}"

HESSIAN_REF_MOLECULES="${HESSIAN_REF_MOLECULES:-100}"
HESSIAN_REF_WORKERS="${HESSIAN_REF_WORKERS:-24}"
HESSIAN_MODEL_SHARDS="${HESSIAN_MODEL_SHARDS:-8}"
HESSIAN_DISPLACEMENT="${HESSIAN_DISPLACEMENT:-1e-3}"
HESSIAN_SCF_ITERATION="${HESSIAN_SCF_ITERATION:-1}"
PBE_HESSIAN_CACHE_DIR="${PBE_HESSIAN_CACHE_DIR:-${DFT_MODELS}/eval/qm9_random1000_pbe_hessian_refs/test100_pbe_hessians}"
PREVIOUS_PBE_HESSIAN_CACHE_DIR="${PREVIOUS_PBE_HESSIAN_CACHE_DIR:-${DFT_MODELS}/eval/qm9_random1000_eg_egf_metrics/20260713_190645/pbe_hessians}"

PBE_MANIFEST_JSON="${OUT_DIR}/pbe_hessian_manifest_test100.json"
PBE_MANIFEST_CSV="${OUT_DIR}/pbe_hessian_manifest_test100.csv"
HESSIAN_JSON="${OUT_DIR}/hessian_fixed_density_test100_parallel.json"
HESSIAN_CSV="${OUT_DIR}/hessian_fixed_density_test100_parallel.csv"

run_timed() {
  local name="$1"
  shift
  echo
  echo "===== ${name} ====="
  echo "command: $*"
  /usr/bin/time -v -o "${LOG_DIR}/${name}.time.txt" "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
}

echo "host: $(hostname)"
echo "date: $(date)"
echo "root: ${ROOT_DIR}"
echo "out_dir: ${OUT_DIR}"
echo "dataset_dir: ${DATASET_DIR}"
echo "pbe_hessian_cache_dir: ${PBE_HESSIAN_CACHE_DIR}"
echo "eg_run_dir: ${EG_RUN_DIR}"
echo "eg_ckpt: ${EG_CKPT}"
echo "egf_run_dir: ${EGF_RUN_DIR}"
echo "egf_ckpt: ${EGF_CKPT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "HESSIAN_REF_MOLECULES=${HESSIAN_REF_MOLECULES}"
echo "HESSIAN_REF_WORKERS=${HESSIAN_REF_WORKERS}"
echo "HESSIAN_MODEL_SHARDS=${HESSIAN_MODEL_SHARDS}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | tee "${OUT_DIR}/gpu_status_before.csv"
fi

for path in "${DATASET_DIR}" "${EG_RUN_DIR}" "${EG_CKPT}" "${EGF_RUN_DIR}" "${EGF_CKPT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: missing required path: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${PBE_HESSIAN_CACHE_DIR}"
if [[ -d "${PREVIOUS_PBE_HESSIAN_CACHE_DIR}" ]]; then
  echo "Seeding reusable PBE Hessian cache from ${PREVIOUS_PBE_HESSIAN_CACHE_DIR}"
  find "${PREVIOUS_PBE_HESSIAN_CACHE_DIR}" -maxdepth 1 -type f -name "pbe_hessian_*.npz" \
    -exec cp -n {} "${PBE_HESSIAN_CACHE_DIR}/" \;
fi

run_timed "01_pbe_hessian_reference_test100" \
  "${PYTHON_BIN}" scripts/qm9_pbe_hessian_reference_set.py \
  --dataset-dir "${DATASET_DIR}" \
  --output-dir "${PBE_HESSIAN_CACHE_DIR}" \
  --manifest-json "${PBE_MANIFEST_JSON}" \
  --manifest-csv "${PBE_MANIFEST_CSV}" \
  --max-molecules "${HESSIAN_REF_MOLECULES}" \
  --sample-id 0 \
  --workers "${HESSIAN_REF_WORKERS}"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

manifest_path = Path("${PBE_MANIFEST_JSON}")
chunk_dir = Path("${CHUNK_DIR}")
chunk_dir.mkdir(parents=True, exist_ok=True)
rows = [row for row in json.loads(manifest_path.read_text()) if row.get("success")]
rows = sorted(rows, key=lambda row: (int(row["natoms"]), row["molecule_id"]), reverse=True)
shards = int("${HESSIAN_MODEL_SHARDS}")
chunks = [[] for _ in range(shards)]
for idx, row in enumerate(rows):
    chunks[idx % shards].append(row)
for idx, chunk in enumerate(chunks):
    (chunk_dir / f"manifest_chunk_{idx:02d}.json").write_text(
        json.dumps(chunk, indent=2, sort_keys=True) + "\\n"
    )
print(json.dumps({
    "references": len(rows),
    "shards": shards,
    "chunk_sizes": [len(chunk) for chunk in chunks],
}, indent=2, sort_keys=True))
PY

echo
echo "===== 02_fixed_density_hessian_test100_parallel ====="
stage_start="$(date +%s)"
pids=()
for shard in $(seq 0 $((HESSIAN_MODEL_SHARDS - 1))); do
  manifest_chunk="${CHUNK_DIR}/manifest_chunk_$(printf "%02d" "${shard}").json"
  if [[ ! -s "${manifest_chunk}" ]]; then
    continue
  fi
  (
    export CUDA_VISIBLE_DEVICES="${shard}"
    "${PYTHON_BIN}" scripts/qm9_hessian_eval_reference_set.py \
      --manifest-json "${manifest_chunk}" \
      --run "EG=${EG_RUN_DIR}=${EG_CKPT}" \
      --run "EGF_lam1=${EGF_RUN_DIR}=${EGF_CKPT}" \
      --output-json "${CHUNK_DIR}/hessian_chunk_$(printf "%02d" "${shard}").json" \
      --output-csv "${CHUNK_DIR}/hessian_chunk_$(printf "%02d" "${shard}").csv" \
      --scf-iteration "${HESSIAN_SCF_ITERATION}" \
      --displacement "${HESSIAN_DISPLACEMENT}" \
      --num-workers 0 \
      --device cuda:0
  ) >"${LOG_DIR}/02_hessian_chunk_$(printf "%02d" "${shard}").log" 2>&1 &
  pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failures=$((failures + 1))
  fi
done
stage_end="$(date +%s)"
{
  echo "Elapsed wall seconds: $((stage_end - stage_start))"
  echo "Failed shard processes: ${failures}"
} | tee "${LOG_DIR}/02_fixed_density_hessian_test100_parallel.time.txt"
if [[ "${failures}" != "0" ]]; then
  echo "ERROR: ${failures} Hessian shard process(es) failed" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<PY
import csv
import json
from pathlib import Path
from statistics import mean

chunk_dir = Path("${CHUNK_DIR}")
out_json = Path("${HESSIAN_JSON}")
out_csv = Path("${HESSIAN_CSV}")
chunk_paths = sorted(chunk_dir.glob("hessian_chunk_*.json"))
rows = []
for path in chunk_paths:
    data = json.loads(path.read_text())
    rows.extend(data.get("rows", []))

def _mean(values):
    values = [value for value in values if value is not None]
    return mean(values) if values else None

summaries = {}
for run in sorted({row.get("run") for row in rows}):
    run_rows = [row for row in rows if row.get("run") == run]
    ok = [row for row in run_rows if row.get("success")]
    failed = [row for row in run_rows if not row.get("success")]
    first = ok[0] if ok else (run_rows[0] if run_rows else {})
    summaries[run] = {
        "run_dir": first.get("run_dir"),
        "ckpt": first.get("ckpt"),
        "n_success": len(ok),
        "n_failed": len(failed),
        "mean_mae": _mean([row.get("mae") for row in ok]),
        "mean_rmse": _mean([row.get("rmse") for row in ok]),
        "mean_relative_fro_error": _mean([row.get("relative_fro_error") for row in ok]),
        "mean_model_symmetry_max_abs_error": _mean(
            [row.get("model_symmetry_max_abs_error") for row in ok]
        ),
    }

result = {
    "manifest_json": "${PBE_MANIFEST_JSON}",
    "device": "8-way parallel cuda:0 with per-shard CUDA_VISIBLE_DEVICES",
    "model_displacement": float("${HESSIAN_DISPLACEMENT}"),
    "scf_iteration": int("${HESSIAN_SCF_ITERATION}"),
    "n_references": len({(row.get("molecule_id"), row.get("sample_id")) for row in rows}),
    "runs": summaries,
    "rows": sorted(rows, key=lambda row: (row.get("run", ""), int(row.get("natoms", 0)), row.get("molecule_id", ""))),
}
out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\\n")

if rows:
    fieldnames = [
        "run", "molecule_id", "sample_id", "scf_iteration", "natoms",
        "success", "mae", "rmse", "max_abs_error", "relative_fro_error",
        "model_symmetry_max_abs_error", "model_hessian_max_abs",
        "reference_cache", "error",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["rows"])

print(json.dumps(result["runs"], indent=2, sort_keys=True))
PY

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

out = Path("${OUT_DIR}")
manifest = json.loads(Path("${PBE_MANIFEST_JSON}").read_text())
hess = json.loads(Path("${HESSIAN_JSON}").read_text())
summary = {
    "out_dir": out.as_posix(),
    "dataset": "${DATASET_NAME}",
    "pbe_reference": {
        "requested": len(manifest),
        "success": sum(1 for row in manifest if row.get("success")),
        "failed": sum(1 for row in manifest if not row.get("success")),
        "workers": int("${HESSIAN_REF_WORKERS}"),
        "cache_dir": "${PBE_HESSIAN_CACHE_DIR}",
    },
    "fixed_density_hessian": hess.get("runs", {}),
}
(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | tee "${OUT_DIR}/gpu_status_after.csv"
fi

echo "Evaluation complete"
echo "out_dir=${OUT_DIR}"
