#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
source "${ENV_FILE:-/scratch/xzh/env.sh}"
export DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
export DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_DENSITY_SHARDS="${NUM_DENSITY_SHARDS:-16}"
DENSITY_CPU_THREADS="${DENSITY_CPU_THREADS:-8}"

DATASET_DIR="${DFT_DATA}/QM9PBEForceRandom1000"
PBE_MANIFEST="${PBE_MANIFEST:-${DFT_MODELS}/eval/qm9_random1000_test100_hessian_node04_gpu4pyscf/20260714_135600/pbe_hessian_manifest_test100_gpu4pyscf.json}"
PBE_CACHE_DIR="$(dirname "$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))[0]["cache_path"])' "${PBE_MANIFEST}")")"
TIER2_SUMMARY="${TIER2_SUMMARY:-$(find "${DFT_MODELS}/eval/qm9_random1000_validation_tier2_strict_hessian" -type f -path '*/analysis/tier2_model_summary.json' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)}"
BASELINE_DENSITY_SUMMARY="${BASELINE_DENSITY_SUMMARY:-${DFT_MODELS}/eval/qm9_random1000_test100_density_relaxed_hessian_node04/20260714_150546/summary.json}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${DFT_MODELS}/eval/qm9_random1000_frozen_test100/${RUN_STAMP}}"
mkdir -p "${OUT_DIR}/force" "${OUT_DIR}/fixed_hessian" "${OUT_DIR}/density_relaxed/chunks" \
  "${OUT_DIR}/density_relaxed/hessians" "${OUT_DIR}/logs"

for path in "${DATASET_DIR}" "${PBE_MANIFEST}" "${TIER2_SUMMARY}" "${BASELINE_DENSITY_SUMMARY}"; do
  if [[ -z "${path}" || ! -e "${path}" ]]; then
    echo "ERROR: missing frozen Test100 input: ${path:-<empty>}" >&2
    exit 1
  fi
done

"${PYTHON_BIN}" - "${TIER2_SUMMARY}" "${OUT_DIR}/model_manifest.json" "${OUT_DIR}/candidate_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

tier2_path, all_output, candidate_output = map(Path, sys.argv[1:])
tier2 = json.load(open(tier2_path))
by_name = {row["name"]: row for row in tier2["models"]}
names = []
for name in ["EG", "EGF_w0p1", *tier2["recommended_for_test100"]]:
    if name not in names:
        names.append(name)
models = [
    {
        "name": name,
        "run_dir": by_name[name]["run_dir"],
        "checkpoint": by_name[name]["checkpoint"],
    }
    for name in names
]
candidates = [
    row for row in models if row["name"] not in {"EG", "EGF_w0p1"}
]
all_output.write_text(json.dumps(models, indent=2, sort_keys=True) + "\n")
candidate_output.write_text(json.dumps(candidates, indent=2, sort_keys=True) + "\n")
print(json.dumps({"all_models": models, "new_density_candidates": candidates}, indent=2))
PY

mapfile -t all_models < <("${PYTHON_BIN}" - "${OUT_DIR}/model_manifest.json" <<'PY'
import json,sys
for row in json.load(open(sys.argv[1])):
    print("\t".join((row["name"],row["run_dir"],row["checkpoint"])))
PY
)

# Ground-state Test100 energy/force for every frozen comparison model.
pids=()
for index in "${!all_models[@]}"; do
  IFS=$'\t' read -r name run_dir ckpt <<<"${all_models[$index]}"
  (
    export CUDA_VISIBLE_DEVICES="${index}"
    /usr/bin/time -v -o "${OUT_DIR}/logs/force_${name}.time.txt" \
      "${PYTHON_BIN}" scripts/qm9_force_eval.py \
        --run-dir "${run_dir}" --ckpt "${ckpt}" \
        --output-json "${OUT_DIR}/force/${name}.json" \
        --worst-csv "${OUT_DIR}/force/${name}_worst.csv" \
        --split test --ground-state-only --batch-size 4 --num-workers 0 --device cuda:0 \
        >"${OUT_DIR}/logs/force_${name}.log" 2>&1
  ) &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done

# Fixed-density scalar-energy autograd Hessian/HVP is the cheap first Test100 diagnostic. Models
# are independent here, so run one worker per model instead of leaving seven A100s idle.
molecules="$(${PYTHON_BIN} - "${PBE_MANIFEST}" <<'PY'
import json,sys
print(",".join(row["molecule_id"] for row in json.load(open(sys.argv[1])) if row.get("success")))
PY
)"
mkdir -p "${OUT_DIR}/fixed_hessian/chunks"
fixed_start="$(date +%s)"
pids=()
fixed_merge_args=()
for index in "${!all_models[@]}"; do
  IFS=$'\t' read -r name run_dir ckpt <<<"${all_models[$index]}"
  chunk_dir="${OUT_DIR}/fixed_hessian/chunks/${name}"
  mkdir -p "${chunk_dir}"
  fixed_merge_args+=(--input-json "${chunk_dir}/summary.json")
  (
    export CUDA_VISIBLE_DEVICES="$((index % 8))"
    /usr/bin/time -v -o "${OUT_DIR}/logs/fixed_hessian_${name}.time.txt" \
      "${PYTHON_BIN}" scripts/qm9_second_order_autograd_hessian_audit.py \
        --run "${name}=${run_dir}=${ckpt}" \
        --molecules "${molecules}" --split test --scf-iteration -1 \
        --num-workers 0 --device cuda:0 --model-dtype float64 \
        --output-json "${chunk_dir}/summary.json" --output-dir "${chunk_dir}" \
        --reference-dir "${PBE_CACHE_DIR}" --fd-displacement 1e-3 --hvp-eps 1e-3 \
        --directional-sample-ids 1 2 3 --compare-pbe-force-secant --run-hvp \
        --no-run-self-edge-diagnostic --no-run-unrolled \
        >"${OUT_DIR}/logs/fixed_hessian_${name}.log" 2>&1
  ) &
  pids+=("$!")
done
fixed_failures=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then fixed_failures=$((fixed_failures + 1)); fi
done
if [[ "${fixed_failures}" != 0 ]]; then
  echo "ERROR: ${fixed_failures} fixed-density model workers failed" >&2
  exit 1
fi
fixed_wall_seconds=$(($(date +%s) - fixed_start))
"${PYTHON_BIN}" scripts/qm9_second_order_audit_merge.py \
  "${fixed_merge_args[@]}" --output-json "${OUT_DIR}/fixed_hessian/summary.json" \
  >"${OUT_DIR}/logs/fixed_hessian_merge.log" 2>&1
printf '{"model_workers": %d, "wall_seconds": %d}\n' \
  "${#all_models[@]}" "${fixed_wall_seconds}" >"${OUT_DIR}/fixed_hessian/timing.json"

# Split Test100 by coordinate cost. Only new frozen candidates are recomputed; EG and historical
# EGF strict results are reused from the immutable baseline run.
"${PYTHON_BIN}" - "${PBE_MANIFEST}" "${OUT_DIR}/density_relaxed/chunks" "${NUM_DENSITY_SHARDS}" <<'PY'
import json,sys
from pathlib import Path
manifest, out = Path(sys.argv[1]), Path(sys.argv[2])
num_shards = int(sys.argv[3])
rows = [row for row in json.load(open(manifest)) if row.get("success")]
rows.sort(key=lambda row:(int(row["natoms"]),row["molecule_id"]), reverse=True)
chunks=[[] for _ in range(num_shards)]; loads=[0]*num_shards
for row in rows:
    shard=min(range(num_shards),key=lambda i:loads[i])
    chunks[shard].append(row); loads[shard]+=6*int(row["natoms"])
for i,chunk in enumerate(chunks):
    (out/f"manifest_{i:02d}.json").write_text(json.dumps(chunk,indent=2)+"\n")
print(json.dumps({"sizes":list(map(len,chunks)),"coordinate_loads":loads},indent=2))
PY

mapfile -t candidates < <("${PYTHON_BIN}" - "${OUT_DIR}/candidate_manifest.json" <<'PY'
import json,sys
for row in json.load(open(sys.argv[1])):
    print("\t".join((row["name"],row["run_dir"],row["checkpoint"])))
PY
)

if [[ "${#candidates[@]}" -gt 0 ]]; then
  candidate_run_args=()
  for line in "${candidates[@]}"; do
    IFS=$'\t' read -r name run_dir ckpt <<<"${line}"
    candidate_run_args+=(--run "${name}=${run_dir}=${ckpt}")
  done
  stage_start="$(date +%s)"
  pids=()
  for shard in $(seq 0 $((NUM_DENSITY_SHARDS - 1))); do
    chunk="${OUT_DIR}/density_relaxed/chunks/manifest_$(printf '%02d' "${shard}").json"
    chunk_size="$(${PYTHON_BIN} -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "${chunk}")"
    (
      export CUDA_VISIBLE_DEVICES="$((shard % 8))"
      OMP_NUM_THREADS="${DENSITY_CPU_THREADS}" MKL_NUM_THREADS="${DENSITY_CPU_THREADS}" \
      OPENBLAS_NUM_THREADS="${DENSITY_CPU_THREADS}" NUMEXPR_NUM_THREADS="${DENSITY_CPU_THREADS}" \
        /usr/bin/time -v -o "${OUT_DIR}/logs/density_shard_$(printf '%02d' "${shard}").time.txt" \
        "${PYTHON_BIN}" scripts/qm9_hessian_density_relaxed_eval.py \
          --manifest-json "${chunk}" --dataset-dir "${DATASET_DIR}" \
          "${candidate_run_args[@]}" \
          --output-json "${OUT_DIR}/density_relaxed/chunks/result_$(printf '%02d' "${shard}").json" \
          --output-csv "${OUT_DIR}/density_relaxed/chunks/molecules_$(printf '%02d' "${shard}").csv" \
          --optimization-csv "${OUT_DIR}/density_relaxed/chunks/optimizations_$(printf '%02d' "${shard}").csv" \
          --hessian-npz-dir "${OUT_DIR}/density_relaxed/hessians" \
          --max-molecules "${chunk_size}" --displacement 1e-3 \
          --initialization sad_default --base-density-warm-start \
          --optimizer adam --lr 1e-3 --max-cycle 1000 --convergence-tolerance 1e-2 \
          --fallback-optimizer adam --fallback-lr 3e-4 --fallback-max-cycle 10000 \
          --fallback-convergence-tolerance 1e-4 --fallback-always \
          --share-prepared-geometry-across-runs --device cuda:0 \
          >"${OUT_DIR}/logs/density_shard_$(printf '%02d' "${shard}").log" 2>&1
    ) &
    pids+=("$!")
  done
  failures=0
  for pid in "${pids[@]}"; do if ! wait "${pid}"; then failures=$((failures+1)); fi; done
  if [[ "${failures}" != 0 ]]; then echo "ERROR: ${failures} Test100 shards failed" >&2; exit 1; fi
  wall_seconds=$(($(date +%s)-stage_start))
  WALL_SECONDS="${wall_seconds}" NUM_DENSITY_SHARDS="${NUM_DENSITY_SHARDS}" \
    DENSITY_CPU_THREADS="${DENSITY_CPU_THREADS}" \
    "${PYTHON_BIN}" - "${OUT_DIR}/density_relaxed" <<'PY'
import csv,json,os,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]); rows=[]; opts=[]; bases=[]
for path in sorted((root/'chunks').glob('result_*.json')):
    payload=json.load(open(path)); rows+=payload.get('rows',[]); opts+=payload.get('optimization_rows',[]); bases+=payload.get('base_optimization_rows',[])
def mean(values):
    values=[float(x) for x in values if x is not None]
    return float(np.mean(values)) if values else None
runs={}
for name in sorted({row['run'] for row in rows}):
    rr=[row for row in rows if row['run']==name and row.get('success')]
    oo=[row for row in opts if row['run']==name]
    bb=[row for row in bases if row['run']==name]
    cycles=[int(row['cycles']) for row in oo if row.get('cycles') is not None]
    runs[name]={
      'n_success':len(rr),'n_failed':sum(row['run']==name and not row.get('success') for row in rows),
      'mean_mae':mean([row.get('mae') for row in rr]),'mean_rmse':mean([row.get('rmse') for row in rr]),
      'mean_relative_fro_error':mean([row.get('relative_fro_error') for row in rr]),
      'mean_model_symmetry_max_abs_error':mean([row.get('model_symmetry_max_abs_error') for row in rr]),
      'optimization_points':len(oo),'strict_converged':sum(row.get('final_gradient_norm') is not None and float(row['final_gradient_norm'])<1e-4 for row in oo),
      'fallback_triggers':sum(bool(row.get('used_fallback')) for row in oo),'mean_cycles':mean(cycles),'max_cycles':max(cycles) if cycles else None,
      'molecule_total_elapsed_s':sum(float(row.get('elapsed_s') or 0) for row in rr),
      'base_optimizations':len(bb),'base_strict_converged':sum(row.get('final_gradient_norm') is not None and float(row['final_gradient_norm'])<1e-4 for row in bb),
    }
result={'definition':'frozen Test100 density-relaxed incomplete-derived-force Hessian','displacement':1e-3,'density_threshold':1e-4,'num_shards':int(os.environ['NUM_DENSITY_SHARDS']),'workers_per_gpu':int(os.environ['NUM_DENSITY_SHARDS'])/8,'density_cpu_threads_per_worker':int(os.environ['DENSITY_CPU_THREADS']),'wall_seconds':int(os.environ['WALL_SECONDS']),'runs':runs,'rows':rows,'optimization_rows':opts,'base_optimization_rows':bases}
(root/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
def write(path,data):
    if not data:return
    fields=sorted({k for row in data for k in row})
    with path.open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(data)
write(root/'molecule_metrics.csv',rows);write(root/'optimization_points.csv',opts)
print(json.dumps({'wall_seconds':result['wall_seconds'],'runs':runs},indent=2))
PY
else
  printf '{"runs": {}, "rows": [], "optimization_rows": []}\n' > "${OUT_DIR}/density_relaxed/summary.json"
fi

"${PYTHON_BIN}" scripts/qm9_hessian_vibrational_metrics.py \
  --manifest-json "${PBE_MANIFEST}" --dataset-dir "${DATASET_DIR}" \
  --result-json "baseline=${BASELINE_DENSITY_SUMMARY}" \
  --result-json "candidates=${OUT_DIR}/density_relaxed/summary.json" \
  --output-dir "${OUT_DIR}/vibrational" >"${OUT_DIR}/logs/vibrational.log" 2>&1

"${PYTHON_BIN}" scripts/qm9_test100_funnel_analysis.py \
  --tier2-summary "${TIER2_SUMMARY}" --baseline-density-summary "${BASELINE_DENSITY_SUMMARY}" \
  --candidate-density-summary "${OUT_DIR}/density_relaxed/summary.json" \
  --force-dir "${OUT_DIR}/force" --fixed-summary "${OUT_DIR}/fixed_hessian/summary.json" \
  --vibration-summary "${OUT_DIR}/vibrational/summary.json" --output-dir "${OUT_DIR}/analysis" \
  >"${OUT_DIR}/logs/analysis.log" 2>&1
echo "Frozen Test100 evaluation complete: ${OUT_DIR}"
