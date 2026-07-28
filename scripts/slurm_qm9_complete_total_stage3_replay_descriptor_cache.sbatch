#!/usr/bin/env bash
#SBATCH --job-name=ctH_s3desc
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/stage3_descriptor_%A_%a.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/stage3_descriptor_%A_%a.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

ROOT=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1
TASK_CSV=${STAGE3_DESCRIPTOR_TASK_CSV:-${ROOT}/replay_baseline_v3_uniform_A_merged/replay_baseline_success.csv}
SCHEMA=${STAGE3_DESCRIPTOR_SCHEMA:-${ROOT}/active_feature_schema_uniform_v3/active_feature_schema_manifest.json}
OUTPUT=${STAGE3_DESCRIPTOR_OUTPUT:-${ROOT}/replay_descriptor_cache_uniform_v3}
SHARD_COUNT=${STAGE3_DESCRIPTOR_SHARD_COUNT:-64}
SHARD_INDEX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}

python scripts/qm9_complete_total_replay_descriptor_cache.py \
  --task-csv "${TASK_CSV}" \
  --schema-manifest "${SCHEMA}" \
  --output-dir "${OUTPUT}" \
  --shard-index "${SHARD_INDEX}" \
  --shard-count "${SHARD_COUNT}" \
  --device cuda:0
