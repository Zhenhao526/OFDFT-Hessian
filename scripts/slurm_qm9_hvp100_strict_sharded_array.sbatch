#!/usr/bin/env bash
#SBATCH --job-name=hvp100_strict_sh
#SBATCH --partition=compute
#SBATCH --nodelist=node01,node02,node04,node05,node06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-239%40
#SBATCH --output=/scratch/xzh/logs/hvp100/%A_%a_strict_sharded.out
#SBATCH --error=/scratch/xzh/logs/hvp100/%A_%a_strict_sharded.err

set -euo pipefail
cd /scratch/xzh/code/structures25
source /scratch/xzh/env.sh
export DFT_DATA=/scratch/xzh/data DFT_MODELS=/scratch/xzh/models
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

ROOT=/scratch/xzh/models/hvp100/20260716_gated25v2
DATA_ROOT=/scratch/xzh/models/hvp100/20260716
TASKS="${ROOT}/analysis/strict_tasks.tsv"
line=$((SLURM_ARRAY_TASK_ID + 2))
IFS=$'\t' read -r task variant weight seed run molecule natoms < <(
  awk -F'\t' -v line="${line}" 'NR==line' "${TASKS}"
)
[[ -n "${run:-}" && -n "${molecule:-}" ]] || {
  echo "No strict task ${SLURM_ARRAY_TASK_ID}" >&2; exit 1;
}
run_dir="${DFT_MODELS}/train/runs/${run}"
ckpt="${run_dir}/checkpoints/last.ckpt"
out="${ROOT}/strict_validation_sharded/tasks/task_$(printf '%03d' "${task}")_${run}_${molecule}"
mkdir -p "${out}"

/usr/bin/time -v -o "${out}/resource.time" \
python scripts/qm9_total_ofdft_hvp_audit.py \
  --dataset-dir "${DFT_DATA}/QM9PBEForceRandom1000" \
  --reference-dir "${DATA_ROOT}/pbe_hessians/cache" \
  --direction-sidecar-dir "${DATA_ROOT}/hvp_sidecars/sidecars" \
  --run "${run}=${run_dir}=${ckpt}" --molecules "${molecule}" \
  --sample-id 0 --direction-coordinate 0 --hvp-step 1e-5 \
  --mixed-derivative-step 1e-4 --integral-derivative-step 1e-4 \
  --integral-derivative-workers 4 --model-geometry-derivative autograd \
  --response-solver auto --krylov-tolerance 1e-8 --max-krylov-iterations 1200 \
  --preconditioner-probes 8 --dense-fallback-max-coefficients 4096 \
  --optimizer adam --lr 1e-3 --max-cycle 1000 --convergence-tolerance 1e-2 \
  --fallback-optimizer adam --fallback-lr 3e-4 --fallback-max-cycle 10000 \
  --fallback-convergence-tolerance 1e-5 --fallback-always \
  --lbfgs-refine --newton-refine --device cuda:0 --transform-device cpu \
  --output-dir "${out}"
