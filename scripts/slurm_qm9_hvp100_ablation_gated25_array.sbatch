#!/usr/bin/env bash
#SBATCH --job-name=hvp100_gate25
#SBATCH --partition=compute
#SBATCH --nodelist=node01,node04,node05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --array=0-23%24
#SBATCH --output=/scratch/xzh/logs/hvp100/%A_%a_gated25_train.out
#SBATCH --error=/scratch/xzh/logs/hvp100/%A_%a_gated25_train.err

set -euo pipefail
cd /scratch/xzh/code/structures25

SEEDS=(676368232 20260716 314159)
VARIANTS=(A B C C C D D D)
WEIGHTS=(0 0 1e-5 1e-4 1e-3 1e-5 1e-4 1e-3)
NAMES=(force force_secant force_hvp_w1e5 force_hvp_w1e4 force_hvp_w1e3 force_secant_hvp_w1e5 force_secant_hvp_w1e4 force_secant_hvp_w1e3)

seed_index=$((SLURM_ARRAY_TASK_ID / 8))
variant_index=$((SLURM_ARRAY_TASK_ID % 8))
seed="${SEEDS[seed_index]}"
variant="${VARIANTS[variant_index]}"
weight="${WEIGHTS[variant_index]}"
run_name="qm9_hvp100_${NAMES[variant_index]}_seed${seed}_s600_gated25v2_20260716"

export MASTER_PORT=$((34000 + SLURM_ARRAY_TASK_ID))
exec bash scripts/launch_qm9_hvp100_train.sh "${variant}" "${weight}" "${seed}" "${run_name}"
