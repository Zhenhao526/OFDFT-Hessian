#!/usr/bin/env bash
#SBATCH --job-name=hvp100_pbe
#SBATCH --partition=compute
#SBATCH --nodelist=node02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100:8
#SBATCH --mem=300G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/xzh/logs/hvp100/%j_pbe.out
#SBATCH --error=/scratch/xzh/logs/hvp100/%j_pbe.err

set -euo pipefail
cd /scratch/xzh/code/structures25
exec bash scripts/launch_qm9_hvp100_pbe_hessians.sh
