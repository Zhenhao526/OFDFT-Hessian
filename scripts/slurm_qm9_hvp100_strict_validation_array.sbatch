#!/usr/bin/env bash
#SBATCH --job-name=hvp100_strict
#SBATCH --partition=compute
#SBATCH --nodelist=node01,node04,node05,node06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --array=0-11%12
#SBATCH --output=/scratch/xzh/logs/hvp100/%A_%a_strict.out
#SBATCH --error=/scratch/xzh/logs/hvp100/%A_%a_strict.err

set -euo pipefail
cd /scratch/xzh/code/structures25
RUNS=/scratch/xzh/models/hvp100/20260716/analysis/strict_runs.tsv
run_name="$(awk -F'\t' -v line="$((SLURM_ARRAY_TASK_ID + 2))" 'NR==line {print $4}' "${RUNS}")"
[[ -n "${run_name}" ]] || { echo "No run for task ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; }
exec bash scripts/launch_qm9_hvp100_strict_validation.sh "${run_name}"
