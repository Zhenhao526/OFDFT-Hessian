#!/usr/bin/env bash
#SBATCH --job-name=ctH_resolve
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/shared_resolve_%A_%a.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/shared_resolve_%A_%a.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

ROOT=/scratch/xzh/models/complete_total_capacity/20260717
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) MODE=none; ESCALE=0.03; FSCALE=0.03; TAG=hessian_only ;;
  1) MODE=soft; ESCALE=0.01; FSCALE=0.01; TAG=soft_e0p01_f0p01 ;;
  2) MODE=soft; ESCALE=0.03; FSCALE=0.03; TAG=soft_e0p03_f0p03 ;;
  3) MODE=soft; ESCALE=0.05; FSCALE=0.03; TAG=soft_e0p05_f0p03 ;;
  4) MODE=soft; ESCALE=0.10; FSCALE=0.05; TAG=soft_e0p10_f0p05 ;;
  *) echo "unsupported array task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac
OUT=${ROOT}/five_parent_shared_geometry_resolve/${TAG}
mkdir -p "${OUT}" /scratch/xzh/logs/complete_total_capacity

/usr/bin/time -v -o "${OUT}/resource.time" \
python scripts/qm9_complete_total_geometry_shared_resolve.py \
  --baseline-manifest "${ROOT}/five_parent_baseline_manifest.json" \
  --design-root "${ROOT}/five_parent_shared_geometry_capacity" \
  --output-dir "${OUT}" \
  --anchor-mode "${MODE}" \
  --energy-anchor-scale "${ESCALE}" \
  --force-anchor-scale "${FSCALE}" \
  --lsqr-max-iterations "${SHARED_RESOLVE_MAX_ITERATIONS:-5000}"
