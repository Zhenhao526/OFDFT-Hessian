#!/usr/bin/env bash
set -euo pipefail

PROFILE=${1:-smoke}
SEED=${2:-20260720}
ROOT=/scratch/xzh/models/complete_total_capacity/20260717
STAGE2=${ROOT}/stage2_direction_v1
STAGE3=${ROOT}/stage3_unseen_parent_v1
BASELINE=${STAGE2}/selection_uniform_A_v1/selected_baseline_manifest.json
DIRECTIONS=${ROOT}/stage2_direction_v3_near_full/candidate_direction_manifest.json
DESIGN=${STAGE2}/shared_descriptor_design_v2
INVENTORY=${STAGE3}/train800_feature_inventory_floor1_v3/train800_feature_inventory_manifest.json
SCHEMA_CHECKPOINT=${ROOT}/stage2_robust_floor1_replay_v2/stable5_expanded_train800_union_h128/initial.ckpt
CHECKPOINT=${STAGE3_TRAIN_CHECKPOINT:-${SCHEMA_CHECKPOINT}}
REPLAY=${STAGE3}/replay_descriptor_cache_train800_union_floor1_v2_merged/replay_descriptor_cache_manifest.json
RUN_REPLAY=${REPLAY}
VALIDATION=${STAGE3}/validation_baselines_v1_uniform_A/validation_baseline_manifest.json
PREFLIGHT=${STAGE3}/training_preflight_train800_union_floor1_v2
RESUME_OPTIMIZER=
HIDDEN_SIZE=128
DEEP_HIDDEN_SIZE=0

cd /scratch/xzh/code/structures25
source /scratch/xzh/env.sh

case "${PROFILE}" in
  smoke)
    STEPS=10
    LR=1e-6
    LOG_INTERVAL=1
    REPLAY_BATCH=1
    REPLAY_EVAL=2
    REPLAY_CPU_CACHE=16
    HVP_WARMUP=5
    OUTPUT=${STAGE3}/robust_replay_train_smoke_seed${SEED}
    ;;
  no-replay-smoke)
    STEPS=10
    LR=1e-6
    LOG_INTERVAL=1
    REPLAY_BATCH=1
    REPLAY_EVAL=2
    REPLAY_CPU_CACHE=16
    HVP_WARMUP=5
    LAMBDA_H=1
    PCGRAD_MAX_RATIO=inf
    RUN_REPLAY=
    OUTPUT=${STAGE3}/robust_floor1_no_replay_uncapped_smoke_seed${SEED}
    ;;
  pilot)
    STEPS=5000
    LR=3e-5
    LOG_INTERVAL=100
    REPLAY_BATCH=4
    REPLAY_EVAL=64
    REPLAY_CPU_CACHE=512
    HVP_WARMUP=1000
    LAMBDA_H=1
    PCGRAD_MAX_RATIO=1
    OUTPUT=${STAGE3}/robust_replay_train_h1_seed${SEED}_s5000
    ;;
  continue-pilot)
    STEPS=25000
    LR=3e-5
    LOG_INTERVAL=100
    REPLAY_BATCH=4
    REPLAY_EVAL=64
    REPLAY_CPU_CACHE=512
    HVP_WARMUP=0
    LAMBDA_H=1
    PCGRAD_MAX_RATIO=1
    if [[ -z "${STAGE3_TRAIN_CHECKPOINT:-}" ]]; then
      CHECKPOINT=${STAGE3}/robust_replay_train_h10_seed${SEED}_s5000/best.ckpt
    fi
    RESUME_OPTIMIZER=1
    OUTPUT=${STAGE3}/robust_replay_train_h1_seed${SEED}_s5000_to_s30000
    ;;
  no-replay-control)
    STEPS=30000
    LR=3e-5
    LOG_INTERVAL=100
    REPLAY_BATCH=4
    REPLAY_EVAL=64
    REPLAY_CPU_CACHE=512
    HVP_WARMUP=1000
    LAMBDA_H=1
    PCGRAD_MAX_RATIO=inf
    RUN_REPLAY=
    OUTPUT=${STAGE3}/robust_floor1_no_replay_uncapped_h1_seed${SEED}_s30000
    ;;
  no-replay-h128-restart-control)
    STEPS=30000
    LR=3e-5
    LOG_INTERVAL=100
    REPLAY_BATCH=4
    REPLAY_EVAL=64
    REPLAY_CPU_CACHE=512
    HVP_WARMUP=1000
    LAMBDA_H=1
    PCGRAD_MAX_RATIO=inf
    RUN_REPLAY=
    CHECKPOINT=${STAGE3}/robust_floor1_no_replay_uncapped_h1_seed${SEED}_s30000/best.ckpt
    PREFLIGHT_CHECKPOINT=${SCHEMA_CHECKPOINT}
    OUTPUT=${STAGE3}/robust_floor1_no_replay_h128_restart_from_s29400_h1_seed${SEED}_s30000
    ;;
  no-replay-h512-smoke|no-replay-h512-control|no-replay-h512-paired-smoke|no-replay-h512-paired-control)
    HIDDEN_SIZE=512
    WIDE_SOURCE=${STAGE3}/robust_floor1_no_replay_uncapped_h1_seed${SEED}_s30000/best.ckpt
    EXPANSION_ARGS=()
    if [[ "${PROFILE}" == *paired* ]]; then
      WIDE_LABEL=h512_cancelpair1e3
      WIDE_ROOT=${STAGE3}/robust_floor1_no_replay_${WIDE_LABEL}_seed${SEED}_initial
      EXPANSION_ARGS+=(--new-unit-initialization canceling_pairs --pair-output-magnitude 1e-3)
    else
      WIDE_LABEL=h512
      WIDE_ROOT=${STAGE3}/robust_floor1_no_replay_${WIDE_LABEL}_seed${SEED}_initial
    fi
    if [[ ! -f "${WIDE_ROOT}/initial.ckpt" ]]; then
      python scripts/expand_qm9_complete_total_hidden_width.py \
        --source-checkpoint "${WIDE_SOURCE}" \
        --target-hidden-size "${HIDDEN_SIZE}" \
        --seed "${SEED}" \
        "${EXPANSION_ARGS[@]}" \
        --output-dir "${WIDE_ROOT}"
    fi
    CHECKPOINT=${WIDE_ROOT}/initial.ckpt
    PREFLIGHT_CHECKPOINT=${SCHEMA_CHECKPOINT}
    RUN_REPLAY=
    LAMBDA_H=1
    PCGRAD_MAX_RATIO=inf
    REPLAY_BATCH=4
    REPLAY_EVAL=64
    REPLAY_CPU_CACHE=512
    if [[ "${PROFILE}" == *smoke ]]; then
      STEPS=10
      LR=1e-6
      LOG_INTERVAL=1
      HVP_WARMUP=5
      OUTPUT=${STAGE3}/robust_floor1_no_replay_${WIDE_LABEL}_smoke_seed${SEED}
    else
      STEPS=30000
      LR=3e-5
      LOG_INTERVAL=100
      HVP_WARMUP=1000
      OUTPUT=${STAGE3}/robust_floor1_no_replay_${WIDE_LABEL}_h1_seed${SEED}_s30000
    fi
    ;;
  no-replay-deep-h256-smoke|no-replay-deep-h256-control)
    DEEP_HIDDEN_SIZE=256
    DEEP_ROOT=${STAGE3}/robust_floor1_no_replay_deep_h128x256_cancelpair1e3_seed${SEED}_initial
    DEEP_SOURCE=${STAGE3}/robust_floor1_no_replay_uncapped_h1_seed${SEED}_s30000/best.ckpt
    if [[ ! -f "${DEEP_ROOT}/initial.ckpt" ]]; then
      python scripts/expand_qm9_complete_total_deep_residual.py \
        --source-checkpoint "${DEEP_SOURCE}" \
        --deep-hidden-size "${DEEP_HIDDEN_SIZE}" \
        --seed "${SEED}" \
        --pair-output-magnitude 1e-3 \
        --output-dir "${DEEP_ROOT}"
    fi
    CHECKPOINT=${DEEP_ROOT}/initial.ckpt
    PREFLIGHT_CHECKPOINT=${SCHEMA_CHECKPOINT}
    RUN_REPLAY=
    LAMBDA_H=1
    PCGRAD_MAX_RATIO=inf
    REPLAY_BATCH=4
    REPLAY_EVAL=64
    REPLAY_CPU_CACHE=512
    if [[ "${PROFILE}" == *smoke ]]; then
      STEPS=10
      LR=1e-6
      LOG_INTERVAL=1
      HVP_WARMUP=5
      OUTPUT=${STAGE3}/robust_floor1_no_replay_deep_h128x256_cancelpair1e3_smoke_seed${SEED}
    else
      STEPS=30000
      LR=3e-5
      LOG_INTERVAL=100
      HVP_WARMUP=1000
      OUTPUT=${STAGE3}/robust_floor1_no_replay_deep_h128x256_cancelpair1e3_h1_seed${SEED}_s30000
    fi
    ;;
  formal)
    STEPS=30000
    LR=3e-5
    LOG_INTERVAL=100
    REPLAY_BATCH=4
    REPLAY_EVAL=64
    REPLAY_CPU_CACHE=512
    HVP_WARMUP=3000
    LAMBDA_H=10
    PCGRAD_MAX_RATIO=1
    OUTPUT=${STAGE3}/robust_replay_train_h10_seed${SEED}_s30000
    ;;
  *)
    echo "usage: $0 {smoke|no-replay-smoke|pilot|continue-pilot|no-replay-control|no-replay-h128-restart-control|no-replay-h512-smoke|no-replay-h512-control|no-replay-h512-paired-smoke|no-replay-h512-paired-control|no-replay-deep-h256-smoke|no-replay-deep-h256-control|formal} [seed]" >&2
    exit 2
    ;;
esac

LAMBDA_H=${LAMBDA_H:-10}
PCGRAD_MAX_RATIO=${PCGRAD_MAX_RATIO:-1}
PREFLIGHT_CHECKPOINT=${PREFLIGHT_CHECKPOINT:-${CHECKPOINT}}

python scripts/qm9_complete_total_stage3_training_preflight.py \
  --baseline-manifest "${BASELINE}" \
  --direction-manifest "${DIRECTIONS}" \
  --checkpoint "${PREFLIGHT_CHECKPOINT}" \
  --replay-cache-manifest "${REPLAY}" \
  --validation-baseline-manifest "${VALIDATION}" \
  --output-dir "${PREFLIGHT}"

sbatch --parsable \
  --export="ALL,STAGE2_MLP_OUTPUT_DIR=${OUTPUT},STAGE2_MLP_BASELINE_MANIFEST=${BASELINE},STAGE2_MLP_DIRECTION_MANIFEST=${DIRECTIONS},STAGE2_MLP_DESIGN_ROOT=${DESIGN},STAGE2_MLP_FEATURE_INVENTORY_MANIFEST=${INVENTORY},STAGE2_MLP_CHECKPOINT=${CHECKPOINT},STAGE2_MLP_RESUME_OPTIMIZER=${RESUME_OPTIMIZER},STAGE2_MLP_COLUMN_NORM_RELATIVE_CUTOFF=0,STAGE2_MLP_FEATURE_SCALE_MODE=floored_column_norm,STAGE2_MLP_FEATURE_SCALE_FLOOR=1.0,STAGE2_MLP_HIDDEN_SIZE=${HIDDEN_SIZE},STAGE2_MLP_DEEP_HIDDEN_SIZE=${DEEP_HIDDEN_SIZE},STAGE2_MLP_SEED=${SEED},STAGE2_MLP_STEPS=${STEPS},STAGE2_MLP_LR=${LR},STAGE2_MLP_LOG_INTERVAL=${LOG_INTERVAL},STAGE2_MLP_LAMBDA_E=1,STAGE2_MLP_LAMBDA_F=1,STAGE2_MLP_LAMBDA_H=${LAMBDA_H},STAGE2_MLP_LAMBDA_SPEC=0.1,STAGE2_MLP_DIRECTIONS_PER_PARENT=8,STAGE2_MLP_REPLAY_CACHE_MANIFEST=${RUN_REPLAY},STAGE2_MLP_REPLAY_BATCH_SIZE=${REPLAY_BATCH},STAGE2_MLP_REPLAY_EVAL_SIZE=${REPLAY_EVAL},STAGE2_MLP_REPLAY_CPU_CACHE_SIZE=${REPLAY_CPU_CACHE},STAGE2_MLP_LAMBDA_REPLAY_E=1,STAGE2_MLP_LAMBDA_REPLAY_F=1,STAGE2_MLP_HVP_WARMUP_STEPS=${HVP_WARMUP},STAGE2_MLP_PCGRAD_MAX_CURVATURE_RATIO=${PCGRAD_MAX_RATIO},STAGE2_MLP_ENERGY_MEDIAN_GATE=1e-3,STAGE2_MLP_ENERGY_MAX_GATE=2e-3,STAGE2_MLP_FORCE_MEDIAN_GATE=1e-3,STAGE2_MLP_FORCE_MAX_GATE=3e-3" \
  scripts/slurm_qm9_complete_total_stage2_direction_mlp.sbatch
