#!/usr/bin/env bash
#SBATCH --job-name=hvp_curve_fullH
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/xzh/logs/hvp_curvature/fullH_%A_%a.out
#SBATCH --error=/scratch/xzh/logs/hvp_curvature/fullH_%A_%a.err

set -euo pipefail
cd /scratch/xzh/code/structures25
source /scratch/xzh/env.sh
export DFT_DATA=/scratch/xzh/data DFT_MODELS=/scratch/xzh/models
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

TASKS="${HVP_FULL_HESSIAN_TASKS:?Set HVP_FULL_HESSIAN_TASKS}"
line=$((SLURM_ARRAY_TASK_ID + 2))
IFS=$'\t' read -r task role variant weight floor seed run molecule natoms reason < <(
  awk -F'\t' -v line="${line}" 'NR==line' "${TASKS}"
)
reason="${reason%$'\r'}"
RUN_DIR="${DFT_MODELS}/train/runs/${run}"
CKPT="${RUN_DIR}/checkpoints/last.ckpt"
if [[ ! -f "${CKPT}" ]]; then
  CKPT="$(find "${RUN_DIR}/checkpoints" -maxdepth 1 -name '*.ckpt' | sort | tail -n 1)"
fi
[[ -f "${CKPT}" ]] || { echo "No checkpoint for ${run}" >&2; exit 1; }
ROOT=/scratch/xzh/models/hvp_curvature_v1/20260716/full_hessian
OUT="${ROOT}/tasks/task_$(printf '%03d' "${task}")_${run}_${molecule}"
mkdir -p "${OUT}"

/usr/bin/time -v -o "${OUT}/resource.time" \
python scripts/qm9_total_ofdft_hessian_audit.py \
  --dataset-dir "${DFT_DATA}/QM9PBEForceRandom1000" \
  --reference-dir "${DFT_MODELS}/hvp100/20260716/pbe_hessians/cache" \
  --run "${run}=${RUN_DIR}=${CKPT}" --molecules "${molecule}" --sample-id 0 \
  --output-dir "${OUT}" --displacement 1e-5 \
  --base-initialization label_reference \
  --integral-derivative-step 1e-4 --integral-derivative-workers 4 \
  --model-geometry-derivative autograd \
  --optimizer adam --lr 1e-3 --max-cycle 1000 --convergence-tolerance 1e-2 \
  --fallback-optimizer adam --fallback-lr 3e-4 --fallback-max-cycle 10000 \
  --fallback-convergence-tolerance 1e-5 --fallback-always \
  --lbfgs-refine --newton-refine --device cuda:0 --transform-device cpu
