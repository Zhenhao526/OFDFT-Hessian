#!/usr/bin/env bash
#SBATCH --job-name=hvp_curve_train
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-19%20
#SBATCH --output=/scratch/xzh/logs/hvp_curvature/%A_%a.out
#SBATCH --error=/scratch/xzh/logs/hvp_curvature/%A_%a.err

set -euo pipefail
cd /scratch/xzh/code/structures25
source /scratch/xzh/env.sh
export DFT_DATA=/scratch/xzh/data DFT_MODELS=/scratch/xzh/models
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

TASKS="${HVP_CURVATURE_TASKS:?Set HVP_CURVATURE_TASKS}"
line=$((SLURM_ARRAY_TASK_ID + 2))
IFS=$'\t' read -r task variant weight floor seed max_steps run_name < <(
  awk -F'\t' -v line="${line}" 'NR==line' "${TASKS}"
)
run_name="${run_name%$'\r'}"
[[ -n "${run_name:-}" ]] || { echo "Missing task ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; }
export HVP_REFERENCE_FLOOR="${floor}"
export MAX_STEPS="${max_steps}"

exec scripts/launch_qm9_hvp_stable_replay_train.sh \
  "${variant}" "${weight}" "${seed}" "${run_name}"
