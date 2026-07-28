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
PBE_MANIFEST="${PBE_MANIFEST:-${DFT_MODELS}/eval/qm9_random1000_test100_hessian_node04_gpu4pyscf/20260714_135600/pbe_hessian_manifest_test100_gpu4pyscf.json}"
EG_RUN_DIR="${EG_RUN_DIR:-${DFT_MODELS}/train/runs/qm9_random1000_eg_e10_20260713_162829}"
EG_CKPT="${EG_CKPT:-${EG_RUN_DIR}/checkpoints/epoch_009.ckpt}"
EGF_RUN_DIR="${EGF_RUN_DIR:-${DFT_MODELS}/train/runs/qm9_random1000_egf_lam1_e10_20260713_162829}"
EGF_CKPT="${EGF_CKPT:-${EGF_RUN_DIR}/checkpoints/epoch_009.ckpt}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${DFT_MODELS}/eval/qm9_random1000_anomaly_tol1e6_full_hessian/${RUN_STAMP}}"
MANIFEST="${OUT_DIR}/anomaly_manifest_2mol.json"
mkdir -p "${OUT_DIR}/conditions" "${OUT_DIR}/logs"

for path in "${DATASET_DIR}" "${PBE_MANIFEST}" "${EG_CKPT}" "${EGF_CKPT}"; do
  [[ -e "${path}" ]] || { echo "ERROR: missing input ${path}" >&2; exit 1; }
done

"${PYTHON_BIN}" - "${PBE_MANIFEST}" "${MANIFEST}" <<'PY'
import json,sys
from pathlib import Path
source,target=map(Path,sys.argv[1:])
by_id={row["molecule_id"]:row for row in json.load(open(source)) if row.get("success")}
ids=["0000777","0040728"]
missing=[x for x in ids if x not in by_id]
if missing: raise SystemExit(f"missing PBE references: {missing}")
target.write_text(json.dumps([by_id[x] for x in ids],indent=2,sort_keys=True)+"\n")
PY

models=(
  "EG=${EG_RUN_DIR}=${EG_CKPT}"
  "EGF_lam1=${EGF_RUN_DIR}=${EGF_CKPT}"
)
steps=("h3em3=3e-3" "h1em3=1e-3" "h3em4=3e-4")
allocated_gpu_tokens=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a allocated_gpu_tokens <<<"${CUDA_VISIBLE_DEVICES}"
fi

printf '%s\n' \
  "definition=full density-relaxed incomplete-derived-force Hessian anomaly tolerance audit" \
  "molecules=0000777,0040728" \
  "steps=3e-3,1e-3,3e-4" \
  "target_density_tolerance=1e-6" \
  "force_definition=-partial_E_model/partial_R_at_optimized_density" \
  "host=$(hostname)" "date=$(date --iso-8601=seconds)" >"${OUT_DIR}/run_config.txt"

pids=()
gpu=0
for model_spec in "${models[@]}"; do
  model_name="${model_spec%%=*}"
  for step_spec in "${steps[@]}"; do
    step_name="${step_spec%%=*}"
    displacement="${step_spec#*=}"
    name="${model_name}_${step_name}_tol1em6"
    condition_dir="${OUT_DIR}/conditions/${name}"
    mkdir -p "${condition_dir}/hessians" "${condition_dir}/optimization_traces"
    gpu_token="${gpu}"
    if [[ "${#allocated_gpu_tokens[@]}" -ge 6 ]]; then
      gpu_token="${allocated_gpu_tokens[$gpu]}"
    fi
    (
      export CUDA_VISIBLE_DEVICES="${gpu_token}"
      /usr/bin/time -v -o "${OUT_DIR}/logs/${name}.time.txt" \
        "${PYTHON_BIN}" scripts/qm9_hessian_density_relaxed_eval.py \
          --manifest-json "${MANIFEST}" --dataset-dir "${DATASET_DIR}" \
          --run "${model_spec}" \
          --output-json "${condition_dir}/summary.json" \
          --output-csv "${condition_dir}/molecule_metrics.csv" \
          --optimization-csv "${condition_dir}/optimization_points.csv" \
          --optimization-trace-dir "${condition_dir}/optimization_traces" \
          --hessian-npz-dir "${condition_dir}/hessians" \
          --max-molecules 2 --displacement "${displacement}" \
          --initialization sad_default --base-density-warm-start \
          --optimizer adam --lr 1e-3 --max-cycle 1000 --convergence-tolerance 1e-2 \
          --fallback-optimizer adam --fallback-lr 3e-4 --fallback-max-cycle 10000 \
          --fallback-convergence-tolerance 1e-6 --fallback-always --device cuda:0 \
          >"${OUT_DIR}/logs/${name}.log" 2>&1
    ) &
    pids+=("$!")
    gpu=$((gpu + 1))
  done
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failures=$((failures + 1)); fi
done

"${PYTHON_BIN}" - "${OUT_DIR}" "${failures}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); failures=int(sys.argv[2]); rows=[]
for path in sorted((root/"conditions").glob("*/summary.json")):
    payload=json.load(open(path))
    for row in payload.get("rows",[]):
        rows.append({"condition":path.parent.name,**row})
result={
    "definition":"full anomaly Hessian at density threshold 1e-6; non-converged points are retained",
    "failed_processes":failures,
    "condition_summaries":6,
    "molecule_rows":len(rows),
    "successful_molecule_rows":sum(bool(row.get("success")) for row in rows),
    "rows":rows,
}
(root/"combined_summary.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
print(json.dumps({k:v for k,v in result.items() if k != "rows"},indent=2))
PY

[[ "${failures}" == 0 ]] || exit 1
echo "Anomaly tolerance-1e-6 full-Hessian audit complete: ${OUT_DIR}"
