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
OUT_DIR="${OUT_DIR:-${DFT_MODELS}/eval/qm9_random1000_test100_density_relaxed_hessian_node04/${RUN_STAMP}}"
LOG_DIR="${OUT_DIR}/logs"
CHUNK_DIR="${OUT_DIR}/chunks"
HESSIAN_NPZ_DIR="${OUT_DIR}/hessians"
mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${CHUNK_DIR}" "${HESSIAN_NPZ_DIR}"

DATASET_DIR="${DFT_DATA}/${DATASET_NAME}"
PBE_MANIFEST="${PBE_MANIFEST:-${DFT_MODELS}/eval/qm9_random1000_test100_hessian_node04_gpu4pyscf/20260714_135600/pbe_hessian_manifest_test100_gpu4pyscf.json}"
EG_RUN_DIR="${EG_RUN_DIR:-${DFT_MODELS}/train/runs/qm9_random1000_eg_e10_20260713_162829}"
EG_CKPT="${EG_CKPT:-${EG_RUN_DIR}/checkpoints/epoch_009.ckpt}"
EGF_RUN_DIR="${EGF_RUN_DIR:-${DFT_MODELS}/train/runs/qm9_random1000_egf_lam1_e10_20260713_162829}"
EGF_CKPT="${EGF_CKPT:-${EGF_RUN_DIR}/checkpoints/epoch_009.ckpt}"

NUM_SHARDS="${NUM_SHARDS:-8}"
MAX_MOLECULES="${MAX_MOLECULES:-100}"
DISPLACEMENT="${DISPLACEMENT:-1e-3}"

echo "host: $(hostname)"
echo "date: $(date)"
echo "out_dir: ${OUT_DIR}"
echo "dataset_dir: ${DATASET_DIR}"
echo "pbe_manifest: ${PBE_MANIFEST}"
echo "num_shards: ${NUM_SHARDS}"
echo "max_molecules: ${MAX_MOLECULES}"

for path in "${DATASET_DIR}" "${PBE_MANIFEST}" "${EG_RUN_DIR}" "${EG_CKPT}" "${EGF_RUN_DIR}" "${EGF_CKPT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: missing required path: ${path}" >&2
    exit 1
  fi
done

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | tee "${OUT_DIR}/gpu_status_before.csv"
fi

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

rows = [
    row
    for row in json.loads(Path("${PBE_MANIFEST}").read_text())
    if row.get("success") and Path(row["cache_path"]).exists()
]
rows = sorted(rows, key=lambda row: (int(row["natoms"]), row["molecule_id"]), reverse=True)
rows = rows[: int("${MAX_MOLECULES}")]
shards = int("${NUM_SHARDS}")
chunks = [[] for _ in range(shards)]
loads = [0 for _ in range(shards)]
for row in rows:
    shard = min(range(shards), key=lambda idx: loads[idx])
    chunks[shard].append(row)
    loads[shard] += 6 * int(row["natoms"])
chunk_dir = Path("${CHUNK_DIR}")
for idx, chunk in enumerate(chunks):
    (chunk_dir / f"manifest_chunk_{idx:02d}.json").write_text(
        json.dumps(chunk, indent=2, sort_keys=True) + "\n"
    )
print(json.dumps({
    "references": len(rows),
    "chunk_sizes": [len(chunk) for chunk in chunks],
    "coordinate_point_loads": loads,
}, indent=2, sort_keys=True))
if len(rows) != int("${MAX_MOLECULES}"):
    raise SystemExit(f"Expected ${MAX_MOLECULES} references, found {len(rows)}")
PY

stage_start="$(date +%s)"
pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  chunk="${CHUNK_DIR}/manifest_chunk_$(printf "%02d" "${shard}").json"
  chunk_size="$("${PYTHON_BIN}" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "${chunk}")"
  if [[ "${chunk_size}" == "0" ]]; then
    continue
  fi
  (
    export CUDA_VISIBLE_DEVICES="${shard}"
    /usr/bin/time -v -o "${LOG_DIR}/shard_$(printf "%02d" "${shard}").time.txt" \
      "${PYTHON_BIN}" scripts/qm9_hessian_density_relaxed_eval.py \
        --manifest-json "${chunk}" \
        --dataset-dir "${DATASET_DIR}" \
        --run "EG=${EG_RUN_DIR}=${EG_CKPT}" \
        --run "EGF_lam1=${EGF_RUN_DIR}=${EGF_CKPT}" \
        --output-json "${CHUNK_DIR}/result_chunk_$(printf "%02d" "${shard}").json" \
        --output-csv "${CHUNK_DIR}/molecules_chunk_$(printf "%02d" "${shard}").csv" \
        --optimization-csv "${CHUNK_DIR}/optimizations_chunk_$(printf "%02d" "${shard}").csv" \
        --hessian-npz-dir "${HESSIAN_NPZ_DIR}" \
        --max-molecules "${chunk_size}" \
        --optimizer adam \
        --lr 1e-3 \
        --max-cycle 1000 \
        --convergence-tolerance 1e-2 \
        --fallback-optimizer adam \
        --fallback-lr 3e-4 \
        --fallback-max-cycle 10000 \
        --fallback-convergence-tolerance 1e-4 \
        --fallback-always \
        --base-density-warm-start \
        --initialization sad_default \
        --displacement "${DISPLACEMENT}" \
        --device cuda:0
  ) >"${LOG_DIR}/shard_$(printf "%02d" "${shard}").log" 2>&1 &
  pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failures=$((failures + 1))
  fi
done
stage_end="$(date +%s)"
wall_seconds=$((stage_end - stage_start))
{
  echo "Elapsed wall seconds: ${wall_seconds}"
  echo "Failed shard processes: ${failures}"
} | tee "${OUT_DIR}/density_relaxed_stage.time.txt"
if [[ "${failures}" != "0" ]]; then
  echo "ERROR: ${failures} density-relaxed shard process(es) failed" >&2
  exit 1
fi

WALL_SECONDS="${wall_seconds}" "${PYTHON_BIN}" - <<PY
import csv
import json
import os
import re
from pathlib import Path
from statistics import mean, median

out_dir = Path("${OUT_DIR}")
chunk_dir = Path("${CHUNK_DIR}")
rows = []
optimization_rows = []
base_rows = []
for path in sorted(chunk_dir.glob("result_chunk_*.json")):
    result = json.loads(path.read_text())
    rows.extend(result.get("rows", []))
    optimization_rows.extend(result.get("optimization_rows", []))
    base_rows.extend(result.get("base_optimization_rows", []))

def avg(values):
    valid = [float(value) for value in values if value is not None]
    return mean(valid) if valid else None

def maximum(values):
    valid = [float(value) for value in values if value is not None]
    return max(valid) if valid else None

def write_csv(path, records):
    if not records:
        return
    fields = sorted({key for record in records for key in record})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

summaries = {}
for run in ("EG", "EGF_lam1"):
    run_rows = [row for row in rows if row.get("run") == run]
    ok = [row for row in run_rows if row.get("success")]
    failed = [row for row in run_rows if not row.get("success")]
    opts = [row for row in optimization_rows if row.get("run") == run]
    bases = [row for row in base_rows if row.get("run") == run]
    cycles = [int(row["cycles"]) for row in opts if row.get("cycles") is not None]
    final_grads = [
        float(row["final_gradient_norm"])
        for row in opts
        if row.get("final_gradient_norm") is not None
    ]
    summaries[run] = {
        "n_success": len(ok),
        "n_failed": len(failed),
        "mean_mae": avg([row.get("mae") for row in ok]),
        "mean_rmse": avg([row.get("rmse") for row in ok]),
        "mean_relative_fro_error": avg([row.get("relative_fro_error") for row in ok]),
        "mean_model_symmetry_max_abs_error": avg(
            [row.get("model_symmetry_max_abs_error") for row in ok]
        ),
        "molecule_total_elapsed_s": sum(float(row.get("elapsed_s", 0.0)) for row in ok),
        "mean_molecule_elapsed_s": avg([row.get("elapsed_s") for row in ok]),
        "optimization_points": len(opts),
        "strict_converged": sum(
            1
            for row in opts
            if row.get("converged")
            and row.get("final_gradient_norm") is not None
            and float(row["final_gradient_norm"]) < 1e-4
        ),
        "medium_converged": sum(
            1
            for row in opts
            if row.get("final_gradient_norm") is not None
            and float(row["final_gradient_norm"]) < 1e-3
        ),
        "fallback_triggers": sum(1 for row in opts if row.get("used_fallback")),
        "mean_cycles": avg(cycles),
        "median_cycles": median(cycles) if cycles else None,
        "max_cycles": max(cycles) if cycles else None,
        "mean_final_gradient_norm": avg(final_grads),
        "max_final_gradient_norm": maximum(final_grads),
        "base_optimizations": len(bases),
        "base_strict_converged": sum(
            1
            for row in bases
            if row.get("converged")
            and row.get("final_gradient_norm") is not None
            and float(row["final_gradient_norm"]) < 1e-4
        ),
        "base_total_elapsed_s": sum(float(row.get("elapsed_s", 0.0)) for row in bases),
        "timing_breakdown_per_displacement_point_s": {
            "sample_build_mean": avg([row.get("sample_build_elapsed_s") for row in opts]),
            "density_optimization_mean": avg(
                [row.get("density_optimization_elapsed_s") for row in opts]
            ),
            "force_autograd_mean": avg([row.get("force_autograd_elapsed_s") for row in opts]),
            "total_mean": avg([row.get("total_point_elapsed_s") for row in opts]),
        },
    }

time_rows = []
for path in sorted((out_dir / "logs").glob("shard_*.time.txt")):
    text = path.read_text()
    rss_match = re.search(r"Maximum resident set size \(kbytes\):\s+(\d+)", text)
    elapsed_match = re.search(r"Elapsed \(wall clock\) time.*:\s+(.+)", text)
    time_rows.append({
        "shard": path.stem.removesuffix(".time"),
        "max_rss_kb": int(rss_match.group(1)) if rss_match else None,
        "elapsed_text": elapsed_match.group(1).strip() if elapsed_match else None,
    })

result = {
    "definition": (
        "density-relaxed derived-force finite-difference Hessian; every displaced geometry "
        "is density-optimized and force is derived from model scalar energy"
    ),
    "limitation": (
        "not a full analytic total-OFDFT Hessian because classical integral/nuclear terms "
        "are not differentiated with respect to nuclear coordinates"
    ),
    "dataset_dir": "${DATASET_DIR}",
    "pbe_manifest": "${PBE_MANIFEST}",
    "displacement": float("${DISPLACEMENT}"),
    "n_references": len({(row.get("molecule_id"), row.get("sample_id")) for row in rows}),
    "num_gpu_shards": int("${NUM_SHARDS}"),
    "wall_seconds": int(os.environ["WALL_SECONDS"]),
    "runs": summaries,
    "shard_times": time_rows,
    "rows": sorted(rows, key=lambda row: (row.get("run", ""), row.get("molecule_id", ""))),
    "optimization_rows": optimization_rows,
    "base_optimization_rows": base_rows,
}
(out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
write_csv(out_dir / "molecule_metrics.csv", result["rows"])
write_csv(out_dir / "optimization_points.csv", optimization_rows)
write_csv(out_dir / "base_optimizations.csv", base_rows)
print(json.dumps({
    "out_dir": out_dir.as_posix(),
    "n_references": result["n_references"],
    "wall_seconds": result["wall_seconds"],
    "runs": result["runs"],
    "shard_times": result["shard_times"],
}, indent=2, sort_keys=True))
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | tee "${OUT_DIR}/gpu_status_after.csv"
fi

echo "Density-relaxed Hessian evaluation complete"
echo "out_dir=${OUT_DIR}"
