#!/usr/bin/env bash
#SBATCH --job-name=hvp_branch
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-95%40
#SBATCH --output=/scratch/xzh/logs/hvp_branch/%A_%a.out
#SBATCH --error=/scratch/xzh/logs/hvp_branch/%A_%a.err

set -euo pipefail
cd /scratch/xzh/code/structures25
source /scratch/xzh/env.sh
export DFT_DATA=/scratch/xzh/data DFT_MODELS=/scratch/xzh/models
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

ROOT="${HVP_BRANCH_ROOT:-${DFT_MODELS}/hvp_branch_stability/20260716}"
TASKS="${HVP_BRANCH_TASKS:-${ROOT}/validation_diagnostic_tasks.tsv}"
TASK_SUBDIR="${HVP_BRANCH_TASK_SUBDIR:-validation_diagnostic_tasks}"
TASK_OFFSET="${HVP_BRANCH_TASK_OFFSET:-0}"
SIDECARS="${HVP_BRANCH_SIDECARS:-${ROOT}/multidirection_sidecars/sidecars}"
RUN_DIR="${HVP_BRANCH_RUN_DIR:-${DFT_MODELS}/train/runs/qm9_random1000_egf_forcew1p0_e10_20260714_202203}"
CKPT="${HVP_BRANCH_CKPT:-${RUN_DIR}/checkpoints/epoch_009.ckpt}"
task_id=$((SLURM_ARRAY_TASK_ID + TASK_OFFSET))
line=$((task_id + 2))
IFS=$'\t' read -r task molecule direction branch base_init displaced_init < <(
  awk -F'\t' -v line="${line}" 'NR==line' "${TASKS}"
)
displaced_init="${displaced_init%$'\r'}"
[[ -n "${molecule:-}" && -n "${branch:-}" ]] || { echo "Missing task ${task_id}" >&2; exit 1; }
out="${ROOT}/${TASK_SUBDIR}/task_$(printf '%04d' "${task}")_${molecule}_d${direction}_${branch}"
mkdir -p "${out}"

/usr/bin/time -v -o "${out}/resource.time" \
python scripts/qm9_total_ofdft_hvp_audit.py \
  --dataset-dir "${DFT_DATA}/QM9PBEForceRandom1000" \
  --reference-dir "${DFT_MODELS}/hvp100/20260716/pbe_hessians/cache" \
  --direction-sidecar-dir "${SIDECARS}" --direction-index "${direction}" \
  --base-initialization "${base_init}" --displaced-initialization "${displaced_init}" \
  --run "Baseline=${RUN_DIR}=${CKPT}" --molecules "${molecule}" --sample-id 0 \
  --hvp-step 1e-5 --curvature-step 3e-5 --curvature-step 3e-6 --curvature-step 1e-3 \
  --mixed-derivative-step 1e-4 --integral-derivative-step 1e-4 \
  --integral-derivative-workers 4 --model-geometry-derivative autograd \
  --response-solver dense --krylov-tolerance 1e-8 --max-krylov-iterations 1200 \
  --optimizer adam --lr 1e-3 --max-cycle 1000 --convergence-tolerance 1e-2 \
  --fallback-optimizer adam --fallback-lr 3e-4 --fallback-max-cycle 10000 \
  --fallback-convergence-tolerance 1e-5 --fallback-always \
  --lbfgs-refine --lbfgs-tolerance 1e-8 --newton-refine --newton-tolerance 1e-8 \
  --device cuda:0 --transform-device cpu --output-dir "${out}"
