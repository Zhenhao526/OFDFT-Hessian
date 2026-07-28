#!/usr/bin/env bash
#SBATCH --job-name=qm9_hthreads
#SBATCH --partition=compute
#SBATCH --nodelist=node06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:a100:3
#SBATCH --mem=300G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/xzh/logs/qm9_random1000_cpu_thread_calibration/%j.out
#SBATCH --error=/scratch/xzh/logs/qm9_random1000_cpu_thread_calibration/%j.err

set -euo pipefail
mkdir -p /scratch/xzh/logs/qm9_random1000_cpu_thread_calibration
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
exec bash scripts/launch_qm9_random1000_cpu_thread_calibration.sh
