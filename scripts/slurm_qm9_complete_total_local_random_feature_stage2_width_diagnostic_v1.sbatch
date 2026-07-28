#!/usr/bin/env bash
#SBATCH --job-name=ct_rf_width
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-5%6
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%A_%a.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%A_%a.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

widths=(0 16 32 64 128 256)
width="${widths[${SLURM_ARRAY_TASK_ID}]}"
root=/scratch/xzh/models/complete_total_capacity/20260717/stage4_train20_direction_v10
output_dir="${root}/width_diagnostic_v1/width_${width}"
mkdir -p "${output_dir}" /scratch/xzh/logs/complete_total_capacity

/usr/bin/time -v -o "${output_dir}/resource.time" \
  python scripts/qm9_complete_total_local_random_feature_stage2.py fit \
    --protocol configs/audit/qm9_complete_total_hessian_local_random_feature_stage2_v10.yaml \
    --jet-dir "${root}/feature_jets_v1" \
    --output-dir "${output_dir}" \
    --device cuda:0 \
    --random-width-per-scale "${width}" \
    --diagnostic-feature-subset

