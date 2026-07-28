#!/usr/bin/env bash
#SBATCH --job-name=ctH_qnet
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --nodelist=node01

set -euo pipefail

source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

: "${SOURCE_RUN_DIR:?set SOURCE_RUN_DIR}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

BASELINE_MANIFEST=${BASELINE_MANIFEST:-/scratch/xzh/models/complete_total_capacity/20260717/stage2_direction_v1/selection_uniform_A_v1/selected_baseline_manifest.json}
DIRECTION_MANIFEST=${DIRECTION_MANIFEST:-/scratch/xzh/models/complete_total_capacity/20260717/stage2_direction_v3_near_full/candidate_direction_manifest.json}
STEPS=${STEPS:-1000}
LOG_INTERVAL=${LOG_INTERVAL:-50}
SEED=${SEED:-20260721}
HIDDEN_SIZE=${HIDDEN_SIZE:-128}
PARENT_BATCH_SIZE=${PARENT_BATCH_SIZE:-4}
DIRECTIONS_PER_PARENT=${DIRECTIONS_PER_PARENT:-8}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-6}
COEFFICIENT_SCALE=${COEFFICIENT_SCALE:-0.1}
CROSS_ALL_PAIRS=${CROSS_ALL_PAIRS:-0}
GLOBAL_CROSS_FEATURES=${GLOBAL_CROSS_FEATURES:-0}

mkdir -p "${OUTPUT_DIR}"

EXTRA_ARGS=()
if [[ "${CROSS_ALL_PAIRS}" == "1" ]]; then
  EXTRA_ARGS+=(--cross-all-pairs)
fi
if [[ "${GLOBAL_CROSS_FEATURES}" == "1" ]]; then
  EXTRA_ARGS+=(--global-cross-features)
fi

/usr/bin/time -v -o "${OUTPUT_DIR}/resource.time" \
  python scripts/qm9_complete_total_local_quadratic_network_capacity.py \
    --baseline-manifest "${BASELINE_MANIFEST}" \
    --source-run-dir "${SOURCE_RUN_DIR}" \
    --direction-manifest "${DIRECTION_MANIFEST}" \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda:0 \
    --seed "${SEED}" \
    --steps "${STEPS}" \
    --log-interval "${LOG_INTERVAL}" \
    --hidden-size "${HIDDEN_SIZE}" \
    --parent-batch-size "${PARENT_BATCH_SIZE}" \
    --directions-per-parent "${DIRECTIONS_PER_PARENT}" \
    --learning-rate "${LEARNING_RATE}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --coefficient-scale "${COEFFICIENT_SCALE}" \
    "${EXTRA_ARGS[@]}"
