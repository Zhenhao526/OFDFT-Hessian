#!/usr/bin/env bash
#SBATCH --job-name=qm9_valpbe8
#SBATCH --partition=compute
#SBATCH --nodelist=node02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/xzh/logs/qm9_random1000_validation8_pbe_hessian/%j.out
#SBATCH --error=/scratch/xzh/logs/qm9_random1000_validation8_pbe_hessian/%j.err

set -euo pipefail
mkdir -p /scratch/xzh/logs/qm9_random1000_validation8_pbe_hessian
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
exec bash scripts/launch_qm9_random1000_validation8_pbe_hessian.sh
