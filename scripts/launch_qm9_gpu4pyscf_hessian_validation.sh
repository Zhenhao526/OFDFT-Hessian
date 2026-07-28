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
export CUDA_VISIBLE_DEVICES="${GPU4PYSCF_VALIDATE_CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${DFT_MODELS}/eval/qm9_random1000_gpu4pyscf_hessian_validation/${RUN_STAMP}}"
GPU_REF_DIR="${OUT_DIR}/gpu_hessians"
CPU_REF_DIR="${CPU_REF_DIR:-${DFT_MODELS}/eval/qm9_random1000_pbe_hessian_refs/test100_pbe_hessians}"
MOLECULES="${MOLECULES:-0000777,0021889,0000242,0044411,0004063}"
mkdir -p "${OUT_DIR}" "${GPU_REF_DIR}"

echo "host: $(hostname)"
echo "date: $(date)"
echo "out_dir: ${OUT_DIR}"
echo "cpu_ref_dir: ${CPU_REF_DIR}"
echo "gpu_ref_dir: ${GPU_REF_DIR}"
echo "molecules: ${MOLECULES}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | tee "${OUT_DIR}/gpu_status_before.csv"
fi

/usr/bin/time -v -o "${OUT_DIR}/gpu_hessian.time.txt" \
  "${PYTHON_BIN}" scripts/qm9_pbe_hessian_reference_set.py \
  --dataset-dir "${DFT_DATA}/${DATASET_NAME}" \
  --output-dir "${GPU_REF_DIR}" \
  --manifest-json "${OUT_DIR}/gpu_hessian_manifest.json" \
  --manifest-csv "${OUT_DIR}/gpu_hessian_manifest.csv" \
  --molecules "${MOLECULES}" \
  --sample-id 0 \
  --workers 1 \
  --backend gpu4pyscf \
  --recompute \
  2>&1 | tee "${OUT_DIR}/gpu_hessian.log"

"${PYTHON_BIN}" - <<PY
import json
import numpy as np
from pathlib import Path

out = Path("${OUT_DIR}")
cpu_dir = Path("${CPU_REF_DIR}")
gpu_dir = Path("${GPU_REF_DIR}")
manifest = json.loads((out / "gpu_hessian_manifest.json").read_text())
rows = []
for rec in manifest:
    molecule_id = rec["molecule_id"]
    sample_id = int(rec["sample_id"])
    cpu_path = cpu_dir / f"pbe_hessian_{molecule_id}_{sample_id:07d}.npz"
    gpu_path = gpu_dir / f"pbe_hessian_{molecule_id}_{sample_id:07d}.npz"
    row = {
        "molecule_id": molecule_id,
        "sample_id": sample_id,
        "natoms": int(rec["natoms"]),
        "gpu_success": bool(rec["success"]),
        "gpu_elapsed_s": rec["elapsed_s"],
        "cpu_path": cpu_path.as_posix(),
        "gpu_path": gpu_path.as_posix(),
        "cpu_exists": cpu_path.exists(),
        "gpu_exists": gpu_path.exists(),
        "error": rec.get("error"),
    }
    if cpu_path.exists() and gpu_path.exists():
        cpu = np.load(cpu_path)["pbe_hessian"]
        gpu = np.load(gpu_path)["pbe_hessian"]
        diff = gpu - cpu
        ref_norm = float(np.linalg.norm(cpu))
        row.update({
            "finite": bool(np.isfinite(gpu).all()),
            "symmetry_max_abs_error": float(np.max(np.abs(gpu - gpu.T))),
            "gpu_vs_cpu_mae": float(np.mean(np.abs(diff))),
            "gpu_vs_cpu_rmse": float(np.sqrt(np.mean(diff * diff))),
            "gpu_vs_cpu_max_abs": float(np.max(np.abs(diff))),
            "gpu_vs_cpu_relative_fro": float(np.linalg.norm(diff) / ref_norm) if ref_norm else None,
        })
    rows.append(row)

threshold_rel = float("${GPU4PYSCF_VALIDATE_MAX_REL_FRO:-1e-3}")
threshold_max = float("${GPU4PYSCF_VALIDATE_MAX_ABS:-2e-3}")
passed = all(
    row.get("gpu_success")
    and row.get("cpu_exists")
    and row.get("gpu_exists")
    and row.get("finite")
    and row.get("gpu_vs_cpu_relative_fro", 1.0) <= threshold_rel
    and row.get("gpu_vs_cpu_max_abs", 1.0) <= threshold_max
    for row in rows
)
summary = {
    "passed": passed,
    "threshold_relative_fro": threshold_rel,
    "threshold_max_abs": threshold_max,
    "rows": rows,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\\n")
print(json.dumps(summary, indent=2, sort_keys=True))
if not passed:
    raise SystemExit(1)
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | tee "${OUT_DIR}/gpu_status_after.csv"
fi

echo "GPU4PySCF Hessian validation complete"
echo "out_dir=${OUT_DIR}"
