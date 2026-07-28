#!/usr/bin/env bash
#SBATCH --job-name=qm9_gcache
#SBATCH --partition=compute
#SBATCH --nodelist=node06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/xzh/logs/qm9_prepared_geometry_cache_smoke/%j.out
#SBATCH --error=/scratch/xzh/logs/qm9_prepared_geometry_cache_smoke/%j.err

set -euo pipefail

mkdir -p /scratch/xzh/logs/qm9_prepared_geometry_cache_smoke
cd /scratch/xzh/code/structures25

exec bash scripts/launch_qm9_prepared_geometry_cache_smoke.sh
