#!/usr/bin/env bash
#SBATCH --job-name=hvp100_fullH
#SBATCH --partition=compute
#SBATCH --nodelist=node01,node02,node04,node05,node06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/xzh/logs/hvp100/%A_%a_full_hessian.out
#SBATCH --error=/scratch/xzh/logs/hvp100/%A_%a_full_hessian.err

set -euo pipefail
cd /scratch/xzh/code/structures25
source /scratch/xzh/env.sh
export DFT_DATA=/scratch/xzh/data DFT_MODELS=/scratch/xzh/models
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

ROOT=/scratch/xzh/models/hvp100/20260716_gated25v2
TASKS="${ROOT}/analysis/full_hessian_tasks.tsv"
line=$((SLURM_ARRAY_TASK_ID + 2))
IFS=$'\t' read -r task role variant weight seed run molecule natoms reason < <(
  awk -F'\t' -v line="${line}" 'NR==line' "${TASKS}"
)
[[ -n "${run:-}" && -n "${molecule:-}" ]] || {
  echo "No full-Hessian task ${SLURM_ARRAY_TASK_ID}" >&2; exit 1;
}
run_dir="${DFT_MODELS}/train/runs/${run}"
ckpt="${run_dir}/checkpoints/last.ckpt"
out="${ROOT}/full_hessian/tasks/task_$(printf '%03d' "${task}")_${run}_${molecule}"
mkdir -p "${out}"

/usr/bin/time -v -o "${out}/resource.time" \
python scripts/qm9_total_ofdft_hessian_audit.py \
  --dataset-dir "${DFT_DATA}/QM9PBEForceRandom1000" \
  --reference-dir "${DFT_MODELS}/hvp100/20260716/pbe_hessians/cache" \
  --run "${run}=${run_dir}=${ckpt}" --molecules "${molecule}" --sample-id 0 \
  --output-dir "${out}" --displacement 1e-5 \
  --integral-derivative-step 1e-4 --integral-derivative-workers 4 \
  --model-geometry-derivative autograd \
  --optimizer adam --lr 1e-3 --max-cycle 1000 --convergence-tolerance 1e-2 \
  --fallback-optimizer adam --fallback-lr 3e-4 --fallback-max-cycle 10000 \
  --fallback-convergence-tolerance 1e-5 --fallback-always \
  --lbfgs-refine --newton-refine --device cuda:0 --transform-device cpu
