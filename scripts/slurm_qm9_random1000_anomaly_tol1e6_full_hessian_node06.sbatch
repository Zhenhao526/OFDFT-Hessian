#!/usr/bin/env bash
#SBATCH --job-name=qm9_hanom1e6
#SBATCH --partition=compute
#SBATCH --nodelist=node06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:a100:6
#SBATCH --mem=700G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/xzh/logs/qm9_random1000_anomaly_tol1e6/%j.out
#SBATCH --error=/scratch/xzh/logs/qm9_random1000_anomaly_tol1e6/%j.err

set -euo pipefail
mkdir -p /scratch/xzh/logs/qm9_random1000_anomaly_tol1e6
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
exec bash scripts/launch_qm9_random1000_anomaly_tol1e6_full_hessian.sh
