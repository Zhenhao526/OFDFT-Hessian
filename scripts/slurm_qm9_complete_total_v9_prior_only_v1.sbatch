#!/usr/bin/env bash
#SBATCH --job-name=ct_v9_prior
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

root=/scratch/xzh/models/complete_total_capacity/20260717/stage4_train20_direction_v10
output_dir="${root}/v9_coefficient_prior_v1/prior_only"
mkdir -p "${output_dir}" /scratch/xzh/logs/complete_total_capacity

/usr/bin/time -v -o "${output_dir}/resource.time" \
  python scripts/qm9_complete_total_local_random_feature_stage2.py fit \
    --protocol configs/audit/qm9_complete_total_hessian_local_random_feature_stage2_v10.yaml \
    --jet-dir "${root}/feature_jets_v1" \
    --output-dir "${output_dir}" \
    --device cuda:0 \
    --v9-prior-only
