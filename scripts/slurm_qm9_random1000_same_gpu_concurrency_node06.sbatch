#!/usr/bin/env bash
#SBATCH --job-name=qm9_hconc
#SBATCH --partition=compute
#SBATCH --nodelist=node06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/xzh/logs/qm9_random1000_same_gpu_concurrency/%j.out
#SBATCH --error=/scratch/xzh/logs/qm9_random1000_same_gpu_concurrency/%j.err

set -euo pipefail
mkdir -p /scratch/xzh/logs/qm9_random1000_same_gpu_concurrency
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
exec bash scripts/launch_qm9_random1000_same_gpu_concurrency_calibration.sh
