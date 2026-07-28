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
export CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${DFT_MODELS}/eval/qm9_random1000_eg_egf_metrics/${RUN_STAMP}}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

DATASET_DIR="${DFT_DATA}/${DATASET_NAME}"
EG_RUN_DIR="${EG_RUN_DIR:-${DFT_MODELS}/train/runs/qm9_random1000_eg_e10_20260713_162829}"
EG_CKPT="${EG_CKPT:-${EG_RUN_DIR}/checkpoints/epoch_009.ckpt}"
EGF_RUN_DIR="${EGF_RUN_DIR:-${DFT_MODELS}/train/runs/qm9_random1000_egf_lam1_e10_20260713_162829}"
EGF_CKPT="${EGF_CKPT:-${EGF_RUN_DIR}/checkpoints/epoch_009.ckpt}"

ENERGY_FORCE_BATCH_SIZE="${ENERGY_FORCE_BATCH_SIZE:-8}"
ENERGY_FORCE_WORKERS="${ENERGY_FORCE_WORKERS:-4}"
HESSIAN_REF_MOLECULES="${HESSIAN_REF_MOLECULES:-10}"
HESSIAN_REF_WORKERS="${HESSIAN_REF_WORKERS:-8}"
HESSIAN_DEVICE="${HESSIAN_DEVICE:-cuda:0}"
HESSIAN_DISPLACEMENT="${HESSIAN_DISPLACEMENT:-1e-3}"
HESSIAN_SCF_ITERATION="${HESSIAN_SCF_ITERATION:-1}"

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
echo "eg_run_dir: ${EG_RUN_DIR}"
echo "eg_ckpt: ${EG_CKPT}"
echo "egf_run_dir: ${EGF_RUN_DIR}"
echo "egf_ckpt: ${EGF_CKPT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

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

run_timed "01_energy_force_eg" \
  "${PYTHON_BIN}" scripts/qm9_force_eval.py \
  --run-dir "${EG_RUN_DIR}" \
  --ckpt "${EG_CKPT}" \
  --output-json "${OUT_DIR}/energy_force_eg.json" \
  --worst-csv "${OUT_DIR}/energy_force_eg_worst.csv" \
  --batch-size "${ENERGY_FORCE_BATCH_SIZE}" \
  --num-workers "${ENERGY_FORCE_WORKERS}" \
  --device cuda:0

run_timed "02_energy_force_egf_lam1" \
  "${PYTHON_BIN}" scripts/qm9_force_eval.py \
  --run-dir "${EGF_RUN_DIR}" \
  --ckpt "${EGF_CKPT}" \
  --output-json "${OUT_DIR}/energy_force_egf_lam1.json" \
  --worst-csv "${OUT_DIR}/energy_force_egf_lam1_worst.csv" \
  --batch-size "${ENERGY_FORCE_BATCH_SIZE}" \
  --num-workers "${ENERGY_FORCE_WORKERS}" \
  --device cuda:0

run_timed "03_pbe_hessian_reference_10mol" \
  "${PYTHON_BIN}" scripts/qm9_pbe_hessian_reference_set.py \
  --dataset-dir "${DATASET_DIR}" \
  --output-dir "${OUT_DIR}/pbe_hessians" \
  --manifest-json "${OUT_DIR}/pbe_hessian_manifest_10mol.json" \
  --manifest-csv "${OUT_DIR}/pbe_hessian_manifest_10mol.csv" \
  --max-molecules "${HESSIAN_REF_MOLECULES}" \
  --sample-id 0 \
  --workers "${HESSIAN_REF_WORKERS}"

run_timed "04_fixed_density_hessian_10mol" \
  "${PYTHON_BIN}" scripts/qm9_hessian_eval_reference_set.py \
  --manifest-json "${OUT_DIR}/pbe_hessian_manifest_10mol.json" \
  --run "EG=${EG_RUN_DIR}=${EG_CKPT}" \
  --run "EGF_lam1=${EGF_RUN_DIR}=${EGF_CKPT}" \
  --output-json "${OUT_DIR}/hessian_fixed_density_10mol.json" \
  --output-csv "${OUT_DIR}/hessian_fixed_density_10mol.csv" \
  --scf-iteration "${HESSIAN_SCF_ITERATION}" \
  --displacement "${HESSIAN_DISPLACEMENT}" \
  --num-workers 0 \
  --device "${HESSIAN_DEVICE}"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

out = Path("${OUT_DIR}")
eg = json.loads((out / "energy_force_eg.json").read_text())
egf = json.loads((out / "energy_force_egf_lam1.json").read_text())
hess = json.loads((out / "hessian_fixed_density_10mol.json").read_text())
manifest = json.loads((out / "pbe_hessian_manifest_10mol.json").read_text())
summary = {
    "out_dir": out.as_posix(),
    "dataset": "${DATASET_NAME}",
    "energy_force": {
        "EG": {k: eg.get(k) for k in [
            "energy_mae", "energy_rmse", "energy_max_abs_error",
            "force_component_mae", "force_component_rmse", "force_vector_mae",
            "max_component_abs_error", "samples_expected", "samples_evaluated",
            "energy_samples_evaluated", "atoms_evaluated",
        ]},
        "EGF_lam1": {k: egf.get(k) for k in [
            "energy_mae", "energy_rmse", "energy_max_abs_error",
            "force_component_mae", "force_component_rmse", "force_vector_mae",
            "max_component_abs_error", "samples_expected", "samples_evaluated",
            "energy_samples_evaluated", "atoms_evaluated",
        ]},
    },
    "hessian_reference": {
        "requested": len(manifest),
        "success": sum(1 for row in manifest if row.get("success")),
        "failed": sum(1 for row in manifest if not row.get("success")),
        "molecules": [
            {
                "molecule_id": row.get("molecule_id"),
                "sample_id": row.get("sample_id"),
                "natoms": row.get("natoms"),
                "elapsed_s": row.get("elapsed_s"),
                "success": row.get("success"),
                "error": row.get("error"),
            }
            for row in manifest
        ],
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
