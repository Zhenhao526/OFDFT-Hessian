#!/usr/bin/env bash
#SBATCH --job-name=hvp_curve_strict
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/xzh/logs/hvp_curvature/strict_%A_%a.out
#SBATCH --error=/scratch/xzh/logs/hvp_curvature/strict_%A_%a.err

set -euo pipefail
cd /scratch/xzh/code/structures25
source /scratch/xzh/env.sh
export DFT_DATA=/scratch/xzh/data DFT_MODELS=/scratch/xzh/models
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

TASKS="${HVP_STRICT_TASKS:?Set HVP_STRICT_TASKS}"
line=$((SLURM_ARRAY_TASK_ID + 2))
IFS=$'\t' read -r task run_name variant seed weight floor direction molecule_count molecules < <(
  awk -F'\t' -v line="${line}" 'NR==line' "${TASKS}"
)
molecules="${molecules%$'\r'}"
RUN_DIR="${DFT_MODELS}/train/runs/${run_name}"
CKPT="${RUN_DIR}/checkpoints/last.ckpt"
if [[ ! -f "${CKPT}" ]]; then
  CKPT="$(find "${RUN_DIR}/checkpoints" -maxdepth 1 -name '*.ckpt' | sort | tail -n 1)"
fi
[[ -f "${CKPT}" ]] || { echo "No checkpoint for ${run_name}" >&2; exit 1; }
ROOT="${HVP_STRICT_ROOT:-/scratch/xzh/models/hvp_curvature_v1/20260716/strict_complete_total}"
[[ "${molecule_count}" -eq 1 && "${molecules}" != *,* ]] || {
  echo "Strict tasks must contain exactly one molecule; got ${molecule_count}: ${molecules}" >&2
  exit 2
}
OUT="${ROOT}/${run_name}/direction_${direction}/${molecules}"
mkdir -p "${OUT}"

/usr/bin/time -v -o "${OUT}/resource.time" \
python scripts/qm9_total_ofdft_hvp_audit.py \
  --dataset-dir "${DFT_DATA}/QM9PBEForceRandom1000" \
  --reference-dir /scratch/xzh/models/hvp100/20260716/pbe_hessians/cache \
  --direction-sidecar-dir /scratch/xzh/models/hvp_branch_stability/20260716/formal_val20_v3/analysis_v3/stable_sidecars \
  --run "${run_name}=${RUN_DIR}=${CKPT}" \
  --molecules "${molecules}" --sample-id 0 --direction-index "${direction}" \
  --base-initialization label_reference --displaced-initialization base_continuation \
  --hvp-step 1e-5 --mixed-derivative-step 1e-4 \
  --integral-derivative-step 1e-4 --integral-derivative-workers 4 \
  --model-geometry-derivative autograd \
  --response-solver auto --krylov-tolerance 1e-8 \
  --max-krylov-iterations 1200 --preconditioner-probes 8 \
  --dense-fallback-max-coefficients 4096 \
  --optimizer adam --lr 1e-3 --max-cycle 1000 --convergence-tolerance 1e-2 \
  --fallback-optimizer adam --fallback-lr 3e-4 \
  --fallback-max-cycle 10000 --fallback-convergence-tolerance 1e-5 \
  --fallback-always --lbfgs-refine --newton-refine \
  --device cuda:0 --transform-device cpu --output-dir "${OUT}"
