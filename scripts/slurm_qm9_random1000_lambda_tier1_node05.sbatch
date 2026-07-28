#!/usr/bin/env bash
#SBATCH --job-name=qm9_lamtier1
#SBATCH --partition=compute
#SBATCH --nodelist=node05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:a100:8
#SBATCH --mem=600G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/xzh/logs/qm9_random1000_lambda_tier1/%j.out
#SBATCH --error=/scratch/xzh/logs/qm9_random1000_lambda_tier1/%j.err

set -euo pipefail
mkdir -p /scratch/xzh/logs/qm9_random1000_lambda_tier1
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export SWEEP_STAMP="${SWEEP_STAMP:-20260714_202203}"
exec bash scripts/launch_qm9_random1000_lambda_tier1_screen.sh
