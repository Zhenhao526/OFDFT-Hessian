#!/usr/bin/env bash
#SBATCH --job-name=hvp4dir
#SBATCH --partition=compute
#SBATCH --nodelist=node02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:a100:8
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/xzh/logs/hvp_branch/%j_sidecars.out
#SBATCH --error=/scratch/xzh/logs/hvp_branch/%j_sidecars.err

set -euo pipefail
mkdir -p /scratch/xzh/logs/hvp_branch
cd /scratch/xzh/code/structures25
exec bash scripts/launch_qm9_hvp_multidirection_sidecars.sh
