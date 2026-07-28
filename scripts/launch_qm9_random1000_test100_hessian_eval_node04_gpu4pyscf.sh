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
OUT_DIR="${OUT_DIR:-${DFT_MODELS}/eval/qm9_random1000_test100_hessian_node04_gpu4pyscf/${RUN_STAMP}}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
PBE_CHUNK_DIR="${OUT_DIR}/pbe_chunks"
HESSIAN_CHUNK_DIR="${OUT_DIR}/hessian_chunks"
mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${PBE_CHUNK_DIR}" "${HESSIAN_CHUNK_DIR}"

DATASET_DIR="${DFT_DATA}/${DATASET_NAME}"
EG_RUN_DIR="${EG_RUN_DIR:-${DFT_MODELS}/train/runs/qm9_random1000_eg_e10_20260713_162829}"
EG_CKPT="${EG_CKPT:-${EG_RUN_DIR}/checkpoints/epoch_009.ckpt}"
EGF_RUN_DIR="${EGF_RUN_DIR:-${DFT_MODELS}/train/runs/qm9_random1000_egf_lam1_e10_20260713_162829}"
EGF_CKPT="${EGF_CKPT:-${EGF_RUN_DIR}/checkpoints/epoch_009.ckpt}"

HESSIAN_REF_MOLECULES="${HESSIAN_REF_MOLECULES:-100}"
GPU_REF_SHARDS="${GPU_REF_SHARDS:-8}"
HESSIAN_MODEL_SHARDS="${HESSIAN_MODEL_SHARDS:-8}"
HESSIAN_DISPLACEMENT="${HESSIAN_DISPLACEMENT:-1e-3}"
HESSIAN_SCF_ITERATION="${HESSIAN_SCF_ITERATION:-1}"
PBE_HESSIAN_CACHE_DIR="${PBE_HESSIAN_CACHE_DIR:-${DFT_MODELS}/eval/qm9_random1000_pbe_hessian_refs/test100_pbe_hessians}"

PBE_MANIFEST_JSON="${OUT_DIR}/pbe_hessian_manifest_test100_gpu4pyscf.json"
PBE_MANIFEST_CSV="${OUT_DIR}/pbe_hessian_manifest_test100_gpu4pyscf.csv"
HESSIAN_JSON="${OUT_DIR}/hessian_fixed_density_test100_parallel.json"
HESSIAN_CSV="${OUT_DIR}/hessian_fixed_density_test100_parallel.csv"

echo "host: $(hostname)"
echo "date: $(date)"
echo "root: ${ROOT_DIR}"
echo "out_dir: ${OUT_DIR}"
echo "dataset_dir: ${DATASET_DIR}"
echo "pbe_hessian_cache_dir: ${PBE_HESSIAN_CACHE_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "GPU_REF_SHARDS=${GPU_REF_SHARDS}"
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

"${PYTHON_BIN}" - <<PY
import json
import pickle
from collections import defaultdict
from pathlib import Path

import zarr

dataset_dir = Path("${DATASET_DIR}")
with (dataset_dir / "split.pkl").open("rb") as f:
    split = pickle.load(f)
rows = []
seen = set()
for _, label_name, n_scf_iter in split["test"]:
    molecule_id, sample_text, *_ = label_name.removesuffix(".zarr.zip").split(".")
    if molecule_id in seen or int(sample_text) != 0:
        continue
    seen.add(molecule_id)
    root = zarr.open(dataset_dir / "labels" / label_name, mode="r")
    rows.append({
        "molecule_id": molecule_id,
        "sample_id": int(sample_text),
        "natoms": int(root["geometry/atomic_numbers"].shape[0]),
    })
rows = sorted(rows, key=lambda row: (int(row["natoms"]), row["molecule_id"]), reverse=True)
rows = rows[: int("${HESSIAN_REF_MOLECULES}")]
shards = int("${GPU_REF_SHARDS}")
chunks = [[] for _ in range(shards)]
for idx, row in enumerate(rows):
    chunks[idx % shards].append(row)
chunk_dir = Path("${PBE_CHUNK_DIR}")
for idx, chunk in enumerate(chunks):
    molecules = ",".join(row["molecule_id"] for row in chunk)
    (chunk_dir / f"molecules_chunk_{idx:02d}.txt").write_text(molecules + "\\n")
print(json.dumps({
    "references": len(rows),
    "gpu_ref_shards": shards,
    "chunk_sizes": [len(chunk) for chunk in chunks],
}, indent=2, sort_keys=True))
PY

echo
echo "===== 01_pbe_hessian_reference_test100_gpu4pyscf ====="
stage_start="$(date +%s)"
pids=()
for shard in $(seq 0 $((GPU_REF_SHARDS - 1))); do
  molecules_file="${PBE_CHUNK_DIR}/molecules_chunk_$(printf "%02d" "${shard}").txt"
  molecules="$(cat "${molecules_file}")"
  if [[ -z "${molecules}" ]]; then
    continue
  fi
  chunk_size="$(awk -F, '{print NF}' "${molecules_file}")"
  (
    export CUDA_VISIBLE_DEVICES="${shard}"
    /usr/bin/time -v -o "${LOG_DIR}/01_pbe_gpu_chunk_$(printf "%02d" "${shard}").time.txt" \
      "${PYTHON_BIN}" scripts/qm9_pbe_hessian_reference_set.py \
        --dataset-dir "${DATASET_DIR}" \
        --output-dir "${PBE_HESSIAN_CACHE_DIR}" \
        --manifest-json "${PBE_CHUNK_DIR}/pbe_manifest_chunk_$(printf "%02d" "${shard}").json" \
        --manifest-csv "${PBE_CHUNK_DIR}/pbe_manifest_chunk_$(printf "%02d" "${shard}").csv" \
        --molecules "${molecules}" \
        --max-molecules "${chunk_size}" \
        --sample-id 0 \
        --workers 1 \
        --backend gpu4pyscf
  ) >"${LOG_DIR}/01_pbe_gpu_chunk_$(printf "%02d" "${shard}").log" 2>&1 &
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
} | tee "${LOG_DIR}/01_pbe_hessian_reference_test100_gpu4pyscf.time.txt"
if [[ "${failures}" != "0" ]]; then
  echo "ERROR: ${failures} GPU4PySCF PBE Hessian shard process(es) failed" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<PY
import csv
import json
from pathlib import Path

chunk_dir = Path("${PBE_CHUNK_DIR}")
rows = []
for path in sorted(chunk_dir.glob("pbe_manifest_chunk_*.json")):
    rows.extend(json.loads(path.read_text()))
rows = sorted(rows, key=lambda row: (int(row["natoms"]), row["molecule_id"]))
Path("${PBE_MANIFEST_JSON}").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\\n")
fieldnames = [
    "molecule_id", "sample_id", "natoms", "backend", "chk_path", "cache_path",
    "success", "cached", "elapsed_s", "shape", "finite",
    "symmetry_max_abs_error", "max_abs", "error",
]
with Path("${PBE_MANIFEST_CSV}").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({
    "requested": len(rows),
    "success": sum(1 for row in rows if row.get("success")),
    "failed": sum(1 for row in rows if not row.get("success")),
    "cached": sum(1 for row in rows if row.get("cached")),
}, indent=2, sort_keys=True))
if any(not row.get("success") for row in rows):
    raise SystemExit(1)
PY

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

manifest_path = Path("${PBE_MANIFEST_JSON}")
chunk_dir = Path("${HESSIAN_CHUNK_DIR}")
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
    "model_shards": shards,
    "chunk_sizes": [len(chunk) for chunk in chunks],
}, indent=2, sort_keys=True))
PY

echo
echo "===== 02_fixed_density_hessian_test100_parallel ====="
stage_start="$(date +%s)"
pids=()
for shard in $(seq 0 $((HESSIAN_MODEL_SHARDS - 1))); do
  manifest_chunk="${HESSIAN_CHUNK_DIR}/manifest_chunk_$(printf "%02d" "${shard}").json"
  if [[ ! -s "${manifest_chunk}" ]]; then
    continue
  fi
  (
    export CUDA_VISIBLE_DEVICES="${shard}"
    "${PYTHON_BIN}" scripts/qm9_hessian_eval_reference_set.py \
      --manifest-json "${manifest_chunk}" \
      --run "EG=${EG_RUN_DIR}=${EG_CKPT}" \
      --run "EGF_lam1=${EGF_RUN_DIR}=${EGF_CKPT}" \
      --output-json "${HESSIAN_CHUNK_DIR}/hessian_chunk_$(printf "%02d" "${shard}").json" \
      --output-csv "${HESSIAN_CHUNK_DIR}/hessian_chunk_$(printf "%02d" "${shard}").csv" \
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

chunk_dir = Path("${HESSIAN_CHUNK_DIR}")
out_json = Path("${HESSIAN_JSON}")
out_csv = Path("${HESSIAN_CSV}")
rows = []
for path in sorted(chunk_dir.glob("hessian_chunk_*.json")):
    rows.extend(json.loads(path.read_text()).get("rows", []))

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
        "backend": "gpu4pyscf",
        "gpu_shards": int("${GPU_REF_SHARDS}"),
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
