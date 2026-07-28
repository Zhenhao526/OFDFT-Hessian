#!/usr/bin/env bash
#SBATCH --job-name=qm9_valsmoke
#SBATCH --partition=compute
#SBATCH --nodelist=node06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xzh/logs/qm9_validation_funnel_smoke/%j.out
#SBATCH --error=/scratch/xzh/logs/qm9_validation_funnel_smoke/%j.err

set -euo pipefail
mkdir -p /scratch/xzh/logs/qm9_validation_funnel_smoke
cd /scratch/xzh/code/structures25
exec bash scripts/launch_qm9_validation_funnel_smoke.sh
