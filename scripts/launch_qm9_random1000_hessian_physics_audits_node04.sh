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
BASELINE_HESSIANS="${BASELINE_DIR}/hessians"
EG_RUN_DIR="${EG_RUN_DIR:-${DFT_MODELS}/train/runs/qm9_random1000_eg_e10_20260713_162829}"
EG_CKPT="${EG_CKPT:-${EG_RUN_DIR}/checkpoints/epoch_009.ckpt}"
EGF_RUN_DIR="${EGF_RUN_DIR:-${DFT_MODELS}/train/runs/qm9_random1000_egf_lam1_e10_20260713_162829}"
EGF_CKPT="${EGF_CKPT:-${EGF_RUN_DIR}/checkpoints/epoch_009.ckpt}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${DFT_MODELS}/eval/qm9_random1000_hessian_physics_audits/${RUN_STAMP}}"
mkdir -p "${OUT_DIR}/logs"

for path in "${DATASET_DIR}" "${PBE_MANIFEST}" "${BASELINE_HESSIANS}" "${EG_CKPT}" "${EGF_CKPT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: missing required path: ${path}" >&2
    exit 1
  fi
done

"${PYTHON_BIN}" - "${PBE_MANIFEST}" "${OUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
rows = json.loads(source.read_text())
by_id = {row["molecule_id"]: row for row in rows if row.get("success")}
selections = {
    "loop_manifest.json": ["0000777", "0072895", "0040728"],
    "anomaly_manifest.json": ["0000777", "0040728"],
    "small_energy_manifest.json": ["0000777", "0021889", "0000242"],
}
for filename, molecule_ids in selections.items():
    missing = [molecule_id for molecule_id in molecule_ids if molecule_id not in by_id]
    if missing:
        raise SystemExit(f"Missing references for {filename}: {missing}")
    selected = [by_id[molecule_id] for molecule_id in molecule_ids]
    (out_dir / filename).write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
print(json.dumps(selections, indent=2))
PY

printf '%s\n' \
  "host=$(hostname)" \
  "date=$(date --iso-8601=seconds)" \
  "out_dir=${OUT_DIR}" \
  "baseline_dir=${BASELINE_DIR}" | tee "${OUT_DIR}/run_config.txt"

common_args=(
  --dataset-dir "${DATASET_DIR}"
  --baseline-hessian-dir "${BASELINE_HESSIANS}"
  --run "EG=${EG_RUN_DIR}=${EG_CKPT}"
  --run "EGF_lam1=${EGF_RUN_DIR}=${EGF_CKPT}"
  --optimizer adam
  --lr 1e-3
  --max-cycle 1000
  --convergence-tolerance 1e-2
  --fallback-optimizer adam
  --fallback-lr 3e-4
  --fallback-max-cycle 10000
  --fallback-always
  --base-density-warm-start
  --initialization sad_default
  --device cuda:0
)

pids=()
launch_loop() {
  local gpu="$1"
  local name="$2"
  local manifest="$3"
  local half_width="$4"
  local tolerance="$5"
  local max_molecules="$6"
  local output="${OUT_DIR}/${name}"
  mkdir -p "${output}/optimization_traces"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    /usr/bin/time -v -o "${OUT_DIR}/logs/${name}.time.txt" \
      "${PYTHON_BIN}" scripts/qm9_force_conservativity_loop_audit.py \
        --manifest-json "${manifest}" \
        --output-dir "${output}" \
        --optimization-trace-dir "${output}/optimization_traces" \
        --half-width "${half_width}" \
        --fallback-convergence-tolerance "${tolerance}" \
        --max-molecules "${max_molecules}" \
        "${common_args[@]}"
  ) >"${OUT_DIR}/logs/${name}.log" 2>&1 &
  pids+=("$!")
}

launch_loop 0 loop_h3em3_tol1em4 "${OUT_DIR}/loop_manifest.json" 3e-3 1e-4 3
launch_loop 1 loop_h3em3_tol1em5 "${OUT_DIR}/loop_manifest.json" 3e-3 1e-5 3
launch_loop 2 loop_h1em3_tol1em4 "${OUT_DIR}/loop_manifest.json" 1e-3 1e-4 3
launch_loop 3 loop_h1em3_tol1em5 "${OUT_DIR}/loop_manifest.json" 1e-3 1e-5 3
launch_loop 4 loop_h3em4_tol1em4 "${OUT_DIR}/loop_manifest.json" 3e-4 1e-4 3
launch_loop 5 loop_h3em4_tol1em5 "${OUT_DIR}/loop_manifest.json" 3e-4 1e-5 3
launch_loop 6 loop_h1em3_tol1em6_anomalies "${OUT_DIR}/anomaly_manifest.json" 1e-3 1e-6 2

launch_energy_pair() {
  local gpu="$1"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    local name displacement output
    for spec in "energy_block_h3em3_tol1em5:3e-3" "energy_block_h1em3_tol1em5:1e-3"; do
      IFS=: read -r name displacement <<<"${spec}"
      output="${OUT_DIR}/${name}"
      mkdir -p "${output}/optimization_traces"
      /usr/bin/time -v -o "${OUT_DIR}/logs/${name}.time.txt" \
        "${PYTHON_BIN}" scripts/qm9_relaxed_scalar_energy_hessian_audit.py \
          --manifest-json "${OUT_DIR}/small_energy_manifest.json" \
          --dataset-dir "${DATASET_DIR}" \
          --force-hessian-dir "${BASELINE_HESSIANS}" \
          --run "EG=${EG_RUN_DIR}=${EG_CKPT}" \
          --run "EGF_lam1=${EGF_RUN_DIR}=${EGF_CKPT}" \
          --output-dir "${output}" \
          --optimization-trace-dir "${output}/optimization_traces" \
          --displacement "${displacement}" \
          --max-molecules 3 \
          --optimizer adam \
          --lr 1e-3 \
          --max-cycle 1000 \
          --convergence-tolerance 1e-2 \
          --fallback-optimizer adam \
          --fallback-lr 3e-4 \
          --fallback-max-cycle 10000 \
          --fallback-convergence-tolerance 1e-5 \
          --fallback-always \
          --base-density-warm-start \
          --initialization sad_default \
          --device cuda:0 \
          >"${OUT_DIR}/logs/${name}.log" 2>&1
    done
  ) &
  pids+=("$!")
}

launch_energy_pair 7

failures=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failures=$((failures + 1))
  fi
done
printf '%s\n' \
  "date=$(date --iso-8601=seconds)" \
  "failed_processes=${failures}" | tee "${OUT_DIR}/audit_status.txt"
if [[ "${failures}" != "0" ]]; then
  exit 1
fi
"${PYTHON_BIN}" scripts/qm9_hessian_physics_audit_analysis.py \
  --root "${OUT_DIR}" \
  --output-dir "${OUT_DIR}/analysis" \
  >"${OUT_DIR}/logs/analysis.log" 2>&1
echo "Physics audits complete: ${OUT_DIR}"
