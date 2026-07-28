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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

DATASET_DIR="${DFT_DATA}/QM9PBEForceRandom1000"
PBE_MANIFEST="${PBE_MANIFEST:-${DFT_MODELS}/eval/qm9_random1000_test100_hessian_node04_gpu4pyscf/20260714_135600/pbe_hessian_manifest_test100_gpu4pyscf.json}"
BASELINE_DIR="${BASELINE_DIR:-${DFT_MODELS}/eval/qm9_random1000_test100_density_relaxed_hessian_node04/20260714_150546}"
EG_RUN_DIR="${EG_RUN_DIR:-${DFT_MODELS}/train/runs/qm9_random1000_eg_e10_20260713_162829}"
EG_CKPT="${EG_CKPT:-${EG_RUN_DIR}/checkpoints/epoch_009.ckpt}"
EGF_RUN_DIR="${EGF_RUN_DIR:-${DFT_MODELS}/train/runs/qm9_random1000_egf_lam1_e10_20260713_162829}"
EGF_CKPT="${EGF_CKPT:-${EGF_RUN_DIR}/checkpoints/epoch_009.ckpt}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${DFT_MODELS}/eval/qm9_random1000_hessian_protocol_scan/${RUN_STAMP}}"
SELECTED_MANIFEST="${OUT_DIR}/representative_manifest_8mol.json"
mkdir -p "${OUT_DIR}/conditions" "${OUT_DIR}/logs"

for path in "${DATASET_DIR}" "${PBE_MANIFEST}" "${BASELINE_DIR}" "${EG_CKPT}" "${EGF_CKPT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: missing required path: ${path}" >&2
    exit 1
  fi
done

"${PYTHON_BIN}" - "${PBE_MANIFEST}" "${SELECTED_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
selected_ids = [
    "0000777",  # smallest and EGF MAE outlier
    "0043905",  # 14 atoms, central-error sample
    "0056566",  # 16 atoms, EG median-error sample
    "0072895",  # 17 atoms, EGF median-error sample
    "0054659",  # 18 atoms, central-error sample
    "0093887",  # 19 atoms, elevated relaxed-force asymmetry
    "0040728",  # 21 atoms, dominant EG error/asymmetry outlier
    "0060531",  # 27 atoms, largest-size representative
]
rows = json.loads(source.read_text())
by_id = {row["molecule_id"]: row for row in rows if row.get("success")}
missing = [molecule_id for molecule_id in selected_ids if molecule_id not in by_id]
if missing:
    raise SystemExit(f"Missing selected PBE references: {missing}")
selected = [by_id[molecule_id] for molecule_id in selected_ids]
target.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
print(json.dumps({
    "selected_ids": selected_ids,
    "natoms": [int(row["natoms"]) for row in selected],
    "manifest": target.as_posix(),
}, indent=2))
PY

# h=1e-3, tolerance=1e-4 already exists in BASELINE_DIR and is deliberately not rerun.
conditions=(
  "h3em3_tol1em4:3e-3:1e-4"
  "h3em3_tol3em5:3e-3:3e-5"
  "h3em3_tol1em5:3e-3:1e-5"
  "h1em3_tol3em5:1e-3:3e-5"
  "h1em3_tol1em5:1e-3:1e-5"
  "h3em4_tol1em4:3e-4:1e-4"
  "h3em4_tol3em5:3e-4:3e-5"
  "h3em4_tol1em5:3e-4:1e-5"
)

printf '%s\n' \
  "host=$(hostname)" \
  "date=$(date --iso-8601=seconds)" \
  "out_dir=${OUT_DIR}" \
  "baseline_dir=${BASELINE_DIR}" \
  "selected_manifest=${SELECTED_MANIFEST}" \
  "conditions=${conditions[*]}" | tee "${OUT_DIR}/run_config.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | tee "${OUT_DIR}/gpu_status_before.csv"
fi

stage_start="$(date +%s)"
pids=()
for gpu in "${!conditions[@]}"; do
  IFS=: read -r name displacement tolerance <<<"${conditions[$gpu]}"
  condition_dir="${OUT_DIR}/conditions/${name}"
  mkdir -p "${condition_dir}/hessians" "${condition_dir}/optimization_traces"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    /usr/bin/time -v -o "${OUT_DIR}/logs/${name}.time.txt" \
      "${PYTHON_BIN}" scripts/qm9_hessian_density_relaxed_eval.py \
        --manifest-json "${SELECTED_MANIFEST}" \
        --dataset-dir "${DATASET_DIR}" \
        --run "EG=${EG_RUN_DIR}=${EG_CKPT}" \
        --run "EGF_lam1=${EGF_RUN_DIR}=${EGF_CKPT}" \
        --output-json "${condition_dir}/summary.json" \
        --output-csv "${condition_dir}/molecule_metrics.csv" \
        --optimization-csv "${condition_dir}/optimization_points.csv" \
        --optimization-trace-dir "${condition_dir}/optimization_traces" \
        --hessian-npz-dir "${condition_dir}/hessians" \
        --max-molecules 8 \
        --optimizer adam \
        --lr 1e-3 \
        --max-cycle 1000 \
        --convergence-tolerance 1e-2 \
        --fallback-optimizer adam \
        --fallback-lr 3e-4 \
        --fallback-max-cycle 10000 \
        --fallback-convergence-tolerance "${tolerance}" \
        --fallback-always \
        --base-density-warm-start \
        --initialization sad_default \
        --displacement "${displacement}" \
        --device cuda:0
  ) >"${OUT_DIR}/logs/${name}.log" 2>&1 &
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

printf '%s\n' \
  "wall_seconds=${wall_seconds}" \
  "failed_condition_processes=${failures}" | tee "${OUT_DIR}/scan_status.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | tee "${OUT_DIR}/gpu_status_after.csv"
fi

if [[ "${failures}" != "0" ]]; then
  exit 1
fi

echo "Protocol scan complete: ${OUT_DIR}"
