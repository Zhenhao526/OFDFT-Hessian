#!/usr/bin/env bash
#SBATCH --job-name=gf_cth_st5_parent
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/xzh/logs/graphformer_relaxed_hvp/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/graphformer_relaxed_hvp/%x_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

root=/scratch/xzh/models/graphformer_relaxed_hvp/20260723/v1
mkdir -p "${root}/manifests" /scratch/xzh/logs/graphformer_relaxed_hvp
python scripts/prepare_qm9_complete_total_capacity_stage1.py \
  --protocol configs/audit/qm9_complete_total_hessian_capacity_v1_stable5_v2.yaml \
  --dataset-dir /scratch/xzh/data/QM9PBEForceRandom1000 \
  --reference-dir /scratch/xzh/models/hvp100/20260716/pbe_hessians/cache \
  --stability-csv /scratch/xzh/models/hvp_branch_stability/20260716/formal_train100_v2/analysis/parent_stability.csv \
  --output "${root}/manifests/stable5_parent_manifest.json"
