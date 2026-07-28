#!/usr/bin/env bash
#SBATCH --job-name=qm9_valtier2
#SBATCH --partition=compute
#SBATCH --nodelist=node04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:a100:8
#SBATCH --mem=800G
#SBATCH --time=05:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/xzh/logs/qm9_random1000_validation_tier2/%j.out
#SBATCH --error=/scratch/xzh/logs/qm9_random1000_validation_tier2/%j.err

set -euo pipefail
mkdir -p /scratch/xzh/logs/qm9_random1000_validation_tier2
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
exec bash scripts/launch_qm9_random1000_validation_tier2_strict_hessian.sh
