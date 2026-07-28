#!/usr/bin/env bash
#SBATCH --job-name=qm9_r1000_hrelax
#SBATCH --partition=compute
#SBATCH --nodelist=node04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:a100:8
#SBATCH --mem=900G
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/xzh/logs/qm9_random1000_test100_density_relaxed_hessian/%j.out
#SBATCH --error=/scratch/xzh/logs/qm9_random1000_test100_density_relaxed_hessian/%j.err

set -euo pipefail

mkdir -p /scratch/xzh/logs/qm9_random1000_test100_density_relaxed_hessian
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

exec bash scripts/launch_qm9_random1000_test100_density_relaxed_hessian_node04.sh
