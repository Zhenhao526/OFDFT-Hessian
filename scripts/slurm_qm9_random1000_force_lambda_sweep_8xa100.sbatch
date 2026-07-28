#!/usr/bin/env bash
#SBATCH --job-name=qm9_lamsweep
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:a100:8
#SBATCH --mem=900G
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/xzh/logs/qm9_random1000_force_lambda_sweep/%j.out
#SBATCH --error=/scratch/xzh/logs/qm9_random1000_force_lambda_sweep/%j.err

set -euo pipefail
mkdir -p /scratch/xzh/logs/qm9_random1000_force_lambda_sweep
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
exec bash scripts/launch_qm9_random1000_force_lambda_sweep_8xa100.sh
