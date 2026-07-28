#!/usr/bin/env bash
#SBATCH --job-name=ctH_local
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --nodelist=node01

set -euo pipefail

source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

: "${SOURCE_RUN_DIR:?set SOURCE_RUN_DIR}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

BASELINE_MANIFEST=${BASELINE_MANIFEST:-/scratch/xzh/models/complete_total_capacity/20260717/stage2_direction_v1/selection_uniform_A_v1/selected_baseline_manifest.json}
DIRECTION_MANIFEST=${DIRECTION_MANIFEST:-/scratch/xzh/models/complete_total_capacity/20260717/stage2_direction_v3_near_full/candidate_direction_manifest.json}
STEPS=${STEPS:-1000}
LOG_INTERVAL=${LOG_INTERVAL:-100}
SEED=${SEED:-20260721}
PARENT_BATCH_SIZE=${PARENT_BATCH_SIZE:-1}
PARENT_LIMIT=${PARENT_LIMIT:-0}
DIRECTIONS_PER_PARENT=${DIRECTIONS_PER_PARENT:-2}
PAIR_HIDDEN_SIZE=${PAIR_HIDDEN_SIZE:-32}
TRIPLET_HIDDEN_SIZE=${TRIPLET_HIDDEN_SIZE:-64}
TORSION_HIDDEN_SIZE=${TORSION_HIDDEN_SIZE:-64}
BOND_SCALE=${BOND_SCALE:-1.25}
CANCELING_OUTPUT=${CANCELING_OUTPUT:-1e-3}
CUTOFF_BOHR=${CUTOFF_BOHR:-8.0}
LEARNING_RATE=${LEARNING_RATE:-3e-5}
TORSION_LEARNING_RATE=${TORSION_LEARNING_RATE:-${LEARNING_RATE}}
HVP_WARMUP_STEPS=${HVP_WARMUP_STEPS:-1000}
LAMBDA_ENERGY=${LAMBDA_ENERGY:-1}
LAMBDA_FORCE=${LAMBDA_FORCE:-1}
LAMBDA_HESSIAN=${LAMBDA_HESSIAN:-1}
LAMBDA_SPECTRUM=${LAMBDA_SPECTRUM:-0}
USE_PCGRAD=${USE_PCGRAD:-1}
USE_GRADNORM=${USE_GRADNORM:-1}
ANCHOR_ENERGY_FORCE_AT_PARENT=${ANCHOR_ENERGY_FORCE_AT_PARENT:-0}

mkdir -p "${OUTPUT_DIR}"

balance_args=()
if [[ "${USE_PCGRAD}" == "1" ]]; then
  balance_args+=(--pcgrad)
fi
if [[ "${USE_GRADNORM}" == "1" ]]; then
  balance_args+=(--gradnorm-balance)
fi
if [[ "${ANCHOR_ENERGY_FORCE_AT_PARENT}" == "1" ]]; then
  balance_args+=(--anchor-energy-force-at-parent)
fi

/usr/bin/time -v -o "${OUTPUT_DIR}/resource.time" \
  python scripts/qm9_complete_total_local_body_order_capacity.py \
    --baseline-manifest "${BASELINE_MANIFEST}" \
    --direction-manifest "${DIRECTION_MANIFEST}" \
    --source-run-dir "${SOURCE_RUN_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda:0 \
    --seed "${SEED}" \
    --steps "${STEPS}" \
    --parent-batch-size "${PARENT_BATCH_SIZE}" \
    --parent-limit "${PARENT_LIMIT}" \
    --directions-per-parent "${DIRECTIONS_PER_PARENT}" \
    --pair-hidden-size "${PAIR_HIDDEN_SIZE}" \
    --triplet-hidden-size "${TRIPLET_HIDDEN_SIZE}" \
    --torsion-hidden-size "${TORSION_HIDDEN_SIZE}" \
    --bond-scale "${BOND_SCALE}" \
    --canceling-output "${CANCELING_OUTPUT}" \
    --cutoff-bohr "${CUTOFF_BOHR}" \
    --learning-rate "${LEARNING_RATE}" \
    --torsion-learning-rate "${TORSION_LEARNING_RATE}" \
    --log-interval "${LOG_INTERVAL}" \
    --hvp-warmup-steps "${HVP_WARMUP_STEPS}" \
    --lambda-energy "${LAMBDA_ENERGY}" \
    --lambda-force "${LAMBDA_FORCE}" \
    --lambda-hessian "${LAMBDA_HESSIAN}" \
    --lambda-spectrum "${LAMBDA_SPECTRUM}" \
    --pcgrad-max-curvature-to-replay-norm-ratio 10 \
    "${balance_args[@]}"
