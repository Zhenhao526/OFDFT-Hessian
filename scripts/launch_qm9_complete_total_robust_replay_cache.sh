#!/usr/bin/env bash
set -euo pipefail

ROOT=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1
TASK_CSV=${ROOT}/replay_baseline_v3_uniform_A_merged/replay_baseline_success.csv
SCHEMA=${ROOT}/active_feature_schema_train800_union_floor1_v2/active_feature_schema_manifest.json
CACHE=${ROOT}/replay_descriptor_cache_train800_union_floor1_v2
MERGED=${ROOT}/replay_descriptor_cache_train800_union_floor1_v2_merged
SHARDS=${SHARDS:-64}
THROTTLE=${THROTTLE:-8}
REPLAY_MERGE_JOB=${REPLAY_MERGE_JOB:-3422}
SCHEMA_JOB=${SCHEMA_JOB:-}

cd /scratch/xzh/code/structures25
source /scratch/xzh/env.sh

DEPENDENCY="afterok:${REPLAY_MERGE_JOB}"
if [[ -n "${SCHEMA_JOB}" ]]; then
  DEPENDENCY+="${DEPENDENCY:+:}${SCHEMA_JOB}"
fi

CACHE_JOB=$(sbatch --parsable --dependency="${DEPENDENCY}" \
  --array="0-$((SHARDS - 1))%${THROTTLE}" \
  --export="ALL,STAGE3_DESCRIPTOR_TASK_CSV=${TASK_CSV},STAGE3_DESCRIPTOR_SCHEMA=${SCHEMA},STAGE3_DESCRIPTOR_OUTPUT=${CACHE},STAGE3_DESCRIPTOR_SHARD_COUNT=${SHARDS}" \
  scripts/slurm_qm9_complete_total_stage3_replay_descriptor_cache.sbatch)

MERGE_JOB=$(sbatch --parsable --dependency="afterok:${CACHE_JOB}" \
  --export="ALL,STAGE3_DESCRIPTOR_TASK_CSV=${TASK_CSV},STAGE3_DESCRIPTOR_SCHEMA=${SCHEMA},STAGE3_DESCRIPTOR_OUTPUT=${CACHE},STAGE3_DESCRIPTOR_MERGED_OUTPUT=${MERGED},STAGE3_DESCRIPTOR_SHARD_COUNT=${SHARDS}" \
  scripts/slurm_qm9_complete_total_stage3_replay_descriptor_cache_merge.sbatch)

printf 'cache_job=%s\ncache_merge_job=%s\n' "${CACHE_JOB}" "${MERGE_JOB}"
