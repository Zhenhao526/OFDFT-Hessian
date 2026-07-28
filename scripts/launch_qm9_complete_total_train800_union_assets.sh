#!/usr/bin/env bash
set -euo pipefail

ROOT=/scratch/xzh/models/complete_total_capacity/20260717
STAGE2=${ROOT}/stage2_direction_v1
ROBUST=${ROOT}/stage2_robust_floor1_replay_v2
STAGE3=${ROOT}/stage3_unseen_parent_v1
BASELINE=${STAGE2}/selection_uniform_A_v1/selected_baseline_manifest.json
DIRECTIONS=${ROOT}/stage2_direction_v3_near_full/candidate_direction_manifest.json
DESIGN=${STAGE2}/shared_descriptor_design_v2
INVENTORY=${STAGE3}/train800_feature_inventory_floor1_v3/train800_feature_inventory_manifest.json
SEED_OUTPUT=${ROBUST}/schema_seed_train800_union_h128
SCHEMA_OUTPUT=${STAGE3}/active_feature_schema_train800_union_floor1_v1

cd /scratch/xzh/code/structures25
source /scratch/xzh/env.sh

SEED_JOB=$(sbatch --parsable \
  --export="ALL,STAGE2_MLP_OUTPUT_DIR=${SEED_OUTPUT},STAGE2_MLP_BASELINE_MANIFEST=${BASELINE},STAGE2_MLP_DIRECTION_MANIFEST=${DIRECTIONS},STAGE2_MLP_DESIGN_ROOT=${DESIGN},STAGE2_MLP_FEATURE_INVENTORY_MANIFEST=${INVENTORY},STAGE2_MLP_COLUMN_NORM_RELATIVE_CUTOFF=0,STAGE2_MLP_FEATURE_SCALE_MODE=floored_column_norm,STAGE2_MLP_FEATURE_SCALE_FLOOR=1.0,STAGE2_MLP_HIDDEN_SIZE=128,STAGE2_MLP_SEED=20260720,STAGE2_MLP_STEPS=0,STAGE2_MLP_LR=3e-4,STAGE2_MLP_LOG_INTERVAL=1,STAGE2_MLP_LAMBDA_E=1,STAGE2_MLP_LAMBDA_F=1,STAGE2_MLP_LAMBDA_H=1,STAGE2_MLP_LAMBDA_SPEC=0" \
  scripts/slurm_qm9_complete_total_stage2_direction_mlp.sbatch)

BIND_JOB=$(sbatch --parsable --dependency="afterok:${SEED_JOB}" \
  --export="ALL,STAGE3_FEATURE_INVENTORY=${INVENTORY},STAGE3_FEATURE_CHECKPOINT=${SEED_OUTPUT}/best.ckpt,STAGE3_FEATURE_SCHEMA_OUTPUT=${SCHEMA_OUTPUT}" \
  scripts/slurm_qm9_complete_total_bind_feature_inventory.sbatch)

printf 'schema_seed_job=%s\nbind_job=%s\n' "${SEED_JOB}" "${BIND_JOB}"
