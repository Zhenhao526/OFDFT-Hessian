#!/usr/bin/env bash
#SBATCH --job-name=ctH_s3dcmerge
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/stage3_descriptor_merge_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/stage3_descriptor_merge_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

ROOT=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1
TASK_CSV=${STAGE3_DESCRIPTOR_TASK_CSV:-${ROOT}/replay_baseline_v3_uniform_A_merged/replay_baseline_success.csv}
SCHEMA=${STAGE3_DESCRIPTOR_SCHEMA:-${ROOT}/active_feature_schema_uniform_v3/active_feature_schema_manifest.json}
CACHE_ROOT=${STAGE3_DESCRIPTOR_OUTPUT:-${ROOT}/replay_descriptor_cache_uniform_v3}
OUTPUT=${STAGE3_DESCRIPTOR_MERGED_OUTPUT:-${ROOT}/replay_descriptor_cache_uniform_v3_merged}
SHARD_COUNT=${STAGE3_DESCRIPTOR_SHARD_COUNT:-64}

python scripts/qm9_complete_total_replay_descriptor_cache_merge.py \
  --task-csv "${TASK_CSV}" \
  --schema-manifest "${SCHEMA}" \
  --cache-root "${CACHE_ROOT}" \
  --output-dir "${OUTPUT}" \
  --shard-count "${SHARD_COUNT}" \
  --require-complete
