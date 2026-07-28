#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
source "${ENV_FILE:-/scratch/xzh/env.sh}"
export DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
export DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHON_BIN="${PYTHON_BIN:-python}"
HESSIAN_CPU_THREADS="${HESSIAN_CPU_THREADS:-8}"

DATASET_DIR="${DFT_DATA}/QM9PBEForceRandom1000"
TIER1_SUMMARY="${TIER1_SUMMARY:-$(find "${DFT_MODELS}/eval/qm9_random1000_lambda_tier1" -type f -path '*/analysis/tier1_model_summary.json' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)}"
PBE_MANIFEST="${PBE_MANIFEST:-$(find "${DFT_MODELS}/eval/qm9_random1000_validation8_pbe_hessian" -type f -name pbe_hessian_manifest_validation8_gpu4pyscf.json -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)}"
PAIRED_RUN_DIR="${PAIRED_RUN_DIR:-$(find "${DFT_MODELS}/train/runs" -mindepth 1 -maxdepth 1 -type d -name 'qm9_random1000_paired_egf_forcew*_computematched_*' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${DFT_MODELS}/eval/qm9_random1000_validation_tier2_strict_hessian/${RUN_STAMP}}"
mkdir -p "${OUT_DIR}/models" "${OUT_DIR}/logs"

for path in "${DATASET_DIR}" "${TIER1_SUMMARY}" "${PBE_MANIFEST}" "${PAIRED_RUN_DIR}"; do
  if [[ -z "${path}" || ! -e "${path}" ]]; then
    echo "ERROR: missing Tier-2 input: ${path:-<empty>}" >&2
    exit 1
  fi
done

"${PYTHON_BIN}" - "${TIER1_SUMMARY}" "${PAIRED_RUN_DIR}" "${OUT_DIR}/model_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

tier1_path, paired_dir, output = map(Path, sys.argv[1:])
tier1 = json.load(open(tier1_path))
by_name = {row["run"]: row for row in tier1["models"]}
names = ["EG", "EGF_w0p1", *tier1["recommended_for_tier2"]]
models = []
seen = set()
for name in names:
    if name in seen:
        continue
    seen.add(name)
    row = by_name[name]
    models.append({
        "name": name,
        "run_dir": row["run_dir"],
        "checkpoint": row["checkpoint"],
        "source": "tier1",
    })
paired_ckpt = paired_dir / "checkpoints" / "epoch_009.ckpt"
if not paired_ckpt.exists():
    paired_ckpt = paired_dir / "checkpoints" / "last.ckpt"
if not paired_ckpt.exists():
    raise SystemExit(f"Missing paired checkpoint in {paired_dir}")
models.append({
    "name": "EGF_paired",
    "run_dir": str(paired_dir),
    "checkpoint": str(paired_ckpt),
    "source": "train-parent-only paired augmentation",
})
output.write_text(json.dumps(models, indent=2, sort_keys=True) + "\n")
print(json.dumps(models, indent=2, sort_keys=True))
PY

mapfile -t model_lines < <("${PYTHON_BIN}" - "${OUT_DIR}/model_manifest.json" <<'PY'
import json
import sys
for row in json.load(open(sys.argv[1])):
    print("\t".join((row["name"], row["run_dir"], row["checkpoint"])))
PY
)

"${PYTHON_BIN}" scripts/qm9_training_ablation_analysis.py \
  --model-manifest "${OUT_DIR}/model_manifest.json" \
  --output-dir "${OUT_DIR}/training_analysis" \
  >"${OUT_DIR}/logs/training_analysis.log" 2>&1

pids=()
for index in "${!model_lines[@]}"; do
  IFS=$'\t' read -r name run_dir ckpt <<<"${model_lines[$index]}"
  model_dir="${OUT_DIR}/models/${name}"
  mkdir -p "${model_dir}/hessians" "${model_dir}/optimization_traces"
  (
    export CUDA_VISIBLE_DEVICES="${index}"
    "${PYTHON_BIN}" scripts/qm9_force_eval.py \
      --run-dir "${run_dir}" --ckpt "${ckpt}" \
      --output-json "${model_dir}/force.json" \
      --worst-csv "${model_dir}/force_worst.csv" \
      --split val --ground-state-only --batch-size 4 --num-workers 0 --device cuda:0 \
      >"${OUT_DIR}/logs/${name}_force.log" 2>&1
    OMP_NUM_THREADS="${HESSIAN_CPU_THREADS}" MKL_NUM_THREADS="${HESSIAN_CPU_THREADS}" \
    OPENBLAS_NUM_THREADS="${HESSIAN_CPU_THREADS}" NUMEXPR_NUM_THREADS="${HESSIAN_CPU_THREADS}" \
      /usr/bin/time -v -o "${OUT_DIR}/logs/${name}_hessian.time.txt" \
      "${PYTHON_BIN}" scripts/qm9_hessian_density_relaxed_eval.py \
        --manifest-json "${PBE_MANIFEST}" --dataset-dir "${DATASET_DIR}" \
        --run "${name}=${run_dir}=${ckpt}" \
        --output-json "${model_dir}/hessian_summary.json" \
        --output-csv "${model_dir}/hessian_metrics.csv" \
        --optimization-csv "${model_dir}/optimization_points.csv" \
        --optimization-trace-dir "${model_dir}/optimization_traces" \
        --hessian-npz-dir "${model_dir}/hessians" \
        --max-molecules 8 --displacement 1e-3 \
        --initialization sad_default --base-density-warm-start \
        --optimizer adam --lr 1e-3 --max-cycle 1000 --convergence-tolerance 1e-2 \
        --fallback-optimizer adam --fallback-lr 3e-4 --fallback-max-cycle 10000 \
        --fallback-convergence-tolerance 1e-4 --fallback-always \
        --device cuda:0 \
        >"${OUT_DIR}/logs/${name}_hessian.log" 2>&1
  ) &
  pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failures=$((failures + 1)); fi
done
if [[ "${failures}" != 0 ]]; then
  echo "ERROR: ${failures} Tier-2 model workers failed" >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/qm9_density_relaxed_tier2_analysis.py \
  --root "${OUT_DIR}" --output-dir "${OUT_DIR}/analysis" \
  >"${OUT_DIR}/logs/analysis.log" 2>&1
echo "Validation Tier-2 strict Hessian complete: ${OUT_DIR}"
