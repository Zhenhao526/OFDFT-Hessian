#!/usr/bin/env bash
#SBATCH --job-name=qm9_total_hvp100
#SBATCH --partition=compute
#SBATCH --nodelist=node04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=152
#SBATCH --gres=gpu:a100:8
#SBATCH --mem=900G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/xzh/logs/qm9_random1000_total_hvp_test100/%j.out
#SBATCH --error=/scratch/xzh/logs/qm9_random1000_total_hvp_test100/%j.err

set -euo pipefail
mkdir -p /scratch/xzh/logs/qm9_random1000_total_hvp_test100
cd /scratch/xzh/code/structures25
exec bash scripts/launch_qm9_random1000_total_hvp_test100.sh
