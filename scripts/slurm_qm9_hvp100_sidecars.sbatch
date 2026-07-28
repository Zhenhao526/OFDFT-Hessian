#!/usr/bin/env bash
#SBATCH --job-name=hvp100_labels
#SBATCH --partition=compute
#SBATCH --nodelist=node02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:a100:8
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/xzh/logs/hvp100/%j_sidecars.out
#SBATCH --error=/scratch/xzh/logs/hvp100/%j_sidecars.err

set -euo pipefail
cd /scratch/xzh/code/structures25
exec bash scripts/launch_qm9_hvp100_sidecars.sh
