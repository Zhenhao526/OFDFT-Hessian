#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
source "${ENV_FILE:-/scratch/xzh/env.sh}"

export DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
export DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_SHARDS="${NUM_SHARDS:-8}"
CPU_THREADS="${CPU_THREADS:-8}"
INPUT_CSV="${INPUT_CSV:-${DFT_MODELS}/eval/qm9_random1000_hessian_protocol_scan/20260714_194632/analysis_pre_rescue_20260715/per_displacement_optimization.csv}"
DATASET_DIR="${DFT_DATA}/QM9PBEForceRandom1000"
EG_RUN_DIR="${DFT_MODELS}/train/runs/qm9_random1000_eg_e10_20260713_162829"
EG_CKPT="${EG_RUN_DIR}/checkpoints/epoch_009.ckpt"
EGF_RUN_DIR="${DFT_MODELS}/train/runs/qm9_random1000_egf_lam1_e10_20260713_162829"
EGF_CKPT="${EGF_RUN_DIR}/checkpoints/epoch_009.ckpt"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${DFT_MODELS}/eval/qm9_random1000_hessian_protocol_bad_point_rescue_parallel/${RUN_STAMP}}"
mkdir -p "${OUT_DIR}/shards" "${OUT_DIR}/curves" "${OUT_DIR}/logs" "${OUT_DIR}/analysis"

for path in "${INPUT_CSV}" "${DATASET_DIR}" "${EG_CKPT}" "${EGF_CKPT}"; do
  [[ -e "${path}" ]] || { echo "ERROR: missing input ${path}" >&2; exit 1; }
done

expected_points="$(${PYTHON_BIN} - "${INPUT_CSV}" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1])))
points={
 (r['run'],r['molecule_id'],int(r['sample_id']),int(r['coord_idx']),r['side'])
 for r in rows
 if r.get('converged')!='True'
 and abs(float(r.get('displacement') or 0)-1e-3)<1e-12
 and abs(float(r.get('tolerance') or 0)-1e-5)<1e-12
 and r['run'] in {'EG','EGF_lam1'}
}
print(len(points))
PY
)"
[[ "${expected_points}" -gt 0 ]] || { echo "ERROR: no rescue points" >&2; exit 1; }

printf '%s\n' \
  "input_csv=${INPUT_CSV}" "expected_points=${expected_points}" \
  "num_shards=${NUM_SHARDS}" "cpu_threads_per_worker=${CPU_THREADS}" \
  "threshold=1e-5" "host=$(hostname)" "date=$(date --iso-8601=seconds)" \
  >"${OUT_DIR}/run_config.txt"

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  (
    export CUDA_VISIBLE_DEVICES="${shard}"
    export OMP_NUM_THREADS="${CPU_THREADS}" MKL_NUM_THREADS="${CPU_THREADS}"
    export OPENBLAS_NUM_THREADS="${CPU_THREADS}" NUMEXPR_NUM_THREADS="${CPU_THREADS}"
    /usr/bin/time -v -o "${OUT_DIR}/logs/shard_${shard}.time.txt" \
      "${PYTHON_BIN}" scripts/qm9_density_relaxed_bad_displacement_rescue.py \
        --previous-optimization-csv "${INPUT_CSV}" --dataset-dir "${DATASET_DIR}" \
        --run "EG=${EG_RUN_DIR}=${EG_CKPT}" \
        --run "EGF_lam1=${EGF_RUN_DIR}=${EGF_CKPT}" \
        --variant "label_adam3e4:label:adam:3e-4:20000" \
        --variant "sad_adam3e4_slsqp:sad_default:adam:3e-4:10000:slsqp:1.0:2000" \
        --variant "label_adam3e4_slsqp:label:adam:3e-4:10000:slsqp:1.0:2000" \
        --source-displacement 1e-3 --source-tolerance 1e-5 --threshold 1e-5 \
        --num-shards "${NUM_SHARDS}" --shard-index "${shard}" \
        --output-json "${OUT_DIR}/shards/result_${shard}.json" \
        --output-csv "${OUT_DIR}/shards/points_${shard}.csv" \
        --summary-csv "${OUT_DIR}/shards/summary_${shard}.csv" \
        --curve-dir "${OUT_DIR}/curves/shard_${shard}" --device cuda:0 \
        >"${OUT_DIR}/logs/shard_${shard}.log" 2>&1
  ) &
  pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failures=$((failures + 1)); fi
done
[[ "${failures}" == 0 ]] || { echo "ERROR: ${failures} rescue shards failed" >&2; exit 1; }

"${PYTHON_BIN}" scripts/qm9_bad_point_rescue_merge.py \
  --shard-dir "${OUT_DIR}/shards" --output-dir "${OUT_DIR}/analysis" \
  --expected-points "${expected_points}" --expected-variants 3 \
  >"${OUT_DIR}/logs/merge.log" 2>&1

echo "Parallel bad-point rescue complete: ${OUT_DIR}"
