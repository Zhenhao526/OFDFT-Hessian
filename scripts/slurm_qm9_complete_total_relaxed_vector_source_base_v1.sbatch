#!/usr/bin/env bash
#SBATCH --job-name=ct_v20_base
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --array=0-19%8
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%A_%a.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%A_%a.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

root=/scratch/xzh/models/complete_total_capacity/20260717/stage6_relaxed_vector_train20_v1
run_dir=/scratch/xzh/models/train/runs/qm9_random1000_egf_forcew1p0_e10_20260714_202203
run="vector_source=${run_dir}=${run_dir}/checkpoints/epoch_009.ckpt"
mkdir -p "${root}/source_baselines" /scratch/xzh/logs/complete_total_capacity

python scripts/qm9_complete_total_replay_baseline.py \
  --protocol configs/audit/qm9_complete_total_relaxed_vector_stage2p5_targets_v1.yaml \
  --task-csv "${root}/selection/source_base_tasks.csv" \
  --output-dir "${root}/source_baselines" \
  --run "${run}" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --device cuda:0 \
  --verify-label-hash \
  --integral-derivative-workers 4 \
  --strict-threshold 1.0e-8
