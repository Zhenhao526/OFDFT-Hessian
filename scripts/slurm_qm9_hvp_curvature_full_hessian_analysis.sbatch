#!/usr/bin/env bash
#SBATCH --job-name=hvp_fullH_analyze
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/xzh/logs/hvp_curvature/fullH_analysis_%j.out
#SBATCH --error=/scratch/xzh/logs/hvp_curvature/fullH_analysis_%j.err

set -euo pipefail
cd /scratch/xzh/code/structures25
source /scratch/xzh/env.sh
python scripts/qm9_hvp_curvature_full_hessian_analysis.py \
  --protocol configs/audit/qm9_hvp_curvature_validation_v1.yaml \
  --task-manifest /scratch/xzh/models/hvp_curvature_v1/20260716/full_hessian_tasks_v1.json \
  --task-root /scratch/xzh/models/hvp_curvature_v1/20260716/full_hessian/tasks \
  --dataset-dir /scratch/xzh/data/QM9PBEForceRandom1000 \
  --output-dir /scratch/xzh/models/hvp_curvature_v1/20260716/full_hessian/analysis
