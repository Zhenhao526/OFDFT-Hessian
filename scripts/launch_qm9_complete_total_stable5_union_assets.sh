#!/usr/bin/env bash
set -euo pipefail

ROOT=/scratch/xzh/models/complete_total_capacity/20260717
STAGE3=${ROOT}/stage3_unseen_parent_v1
SOURCE_RUN=${ROOT}/stage1_floored_stable5_v1/floor_1p0/mlp_h128_h10_s30000
SOURCE_DESIGN=${ROOT}/five_parent_shared_geometry_capacity_stable5_v2
SOURCE_SCHEMA=${ROOT}/stage1_floored_stable5_v1/floor_1p0/active_feature_schema_v1
INVENTORY=${STAGE3}/train800_feature_inventory_floor1_v3/train800_feature_inventory_manifest.json
EXPANDED=${ROOT}/stage2_robust_floor1_replay_v2/stable5_expanded_train800_union_h128
BOUND_SCHEMA=${STAGE3}/active_feature_schema_train800_union_floor1_v2

cd /scratch/xzh/code/structures25
source /scratch/xzh/env.sh

SOURCE_SCHEMA_JOB=$(sbatch --parsable \
  --export="ALL,STAGE3_SCHEMA_DESIGN_ROOT=${SOURCE_DESIGN},STAGE3_SCHEMA_CHECKPOINT=${SOURCE_RUN}/best.ckpt,STAGE3_SCHEMA_OUTPUT=${SOURCE_SCHEMA},STAGE3_SCHEMA_COLUMN_NORM_RELATIVE_CUTOFF=1e-8,STAGE3_SCHEMA_FEATURE_SCALE_MODE=floored_column_norm,STAGE3_SCHEMA_FEATURE_SCALE_FLOOR=1.0" \
  scripts/slurm_qm9_complete_total_stage3_active_feature_schema.sbatch)

EXPAND_JOB=$(sbatch --parsable --dependency="afterok:${SOURCE_SCHEMA_JOB}" \
  --export="ALL,STAGE3_SOURCE_CHECKPOINT=${SOURCE_RUN}/best.ckpt,STAGE3_SOURCE_SCHEMA=${SOURCE_SCHEMA}/active_feature_schema_manifest.json,STAGE3_FEATURE_INVENTORY=${INVENTORY},STAGE3_EXPANDED_CHECKPOINT_OUTPUT=${EXPANDED}" \
  scripts/slurm_qm9_complete_total_expand_checkpoint.sbatch)

BIND_JOB=$(sbatch --parsable --dependency="afterok:${EXPAND_JOB}" \
  --export="ALL,STAGE3_FEATURE_INVENTORY=${INVENTORY},STAGE3_FEATURE_CHECKPOINT=${EXPANDED}/initial.ckpt,STAGE3_FEATURE_SCHEMA_OUTPUT=${BOUND_SCHEMA}" \
  scripts/slurm_qm9_complete_total_bind_feature_inventory.sbatch)

printf 'source_schema_job=%s\nexpand_job=%s\nbind_job=%s\n' \
  "${SOURCE_SCHEMA_JOB}" "${EXPAND_JOB}" "${BIND_JOB}"
