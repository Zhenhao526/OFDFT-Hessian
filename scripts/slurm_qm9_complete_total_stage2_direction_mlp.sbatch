#!/usr/bin/env bash
#SBATCH --job-name=ctH_s2mlp
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/stage2_mlp_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/stage2_mlp_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

ROOT=/scratch/xzh/models/complete_total_capacity/20260717/stage2_direction_v1
OUT=${STAGE2_MLP_OUTPUT_DIR:-${ROOT}/mlp_h128_warm}
mkdir -p "${OUT}" /scratch/xzh/logs/complete_total_capacity

CHECKPOINT_ARGS=()
if [[ -n "${STAGE2_MLP_CHECKPOINT:-}" ]]; then
  CHECKPOINT_ARGS+=(--checkpoint "${STAGE2_MLP_CHECKPOINT}")
fi
if [[ -n "${STAGE2_MLP_RESUME_OPTIMIZER:-}" ]]; then
  CHECKPOINT_ARGS+=(--resume-optimizer)
fi
BALANCE_ARGS=()
if [[ -n "${STAGE2_MLP_PCGRAD:-1}" ]]; then
  BALANCE_ARGS+=(--pcgrad)
fi
ARCHITECTURE_ARGS=()
if [[ -n "${STAGE2_MLP_ATOM_COUNT_EXTENSIVE:-}" ]]; then
  ARCHITECTURE_ARGS+=(--atom-count-extensive)
fi
DIRECTION_ARGS=()
if [[ -z "${STAGE2_MLP_FULL_HESSIAN_CAPACITY:-}" ]]; then
  DIRECTION_ARGS+=(
    --direction-manifest
    "${STAGE2_MLP_DIRECTION_MANIFEST:-${ROOT}/candidate_direction_manifest.json}"
  )
fi
INITIAL_ARGS=()
if [[ -n "${STAGE2_MLP_INITIAL_ROOT:-}" ]]; then
  INITIAL_ARGS+=(--initial-root "${STAGE2_MLP_INITIAL_ROOT}")
else
  INITIAL_ARGS+=(--zero-linear-initialization)
fi
REPLAY_ARGS=()
if [[ -n "${STAGE2_MLP_REPLAY_CACHE_MANIFEST:-}" ]]; then
  REPLAY_ARGS+=(
    --replay-cache-manifest "${STAGE2_MLP_REPLAY_CACHE_MANIFEST}"
    --replay-batch-size "${STAGE2_MLP_REPLAY_BATCH_SIZE:-4}"
    --replay-eval-size "${STAGE2_MLP_REPLAY_EVAL_SIZE:-32}"
    --replay-cpu-cache-size "${STAGE2_MLP_REPLAY_CPU_CACHE_SIZE:-256}"
    --lambda-replay-energy "${STAGE2_MLP_LAMBDA_REPLAY_E:-1}"
    --lambda-replay-force "${STAGE2_MLP_LAMBDA_REPLAY_F:-1}"
  )
fi
INVENTORY_ARGS=()
if [[ -n "${STAGE2_MLP_FEATURE_INVENTORY_MANIFEST:-}" ]]; then
  INVENTORY_ARGS+=(
    --feature-inventory-manifest "${STAGE2_MLP_FEATURE_INVENTORY_MANIFEST}"
  )
fi

/usr/bin/time -v -o "${OUT}/resource.time" \
python scripts/qm9_complete_total_geometry_mlp_capacity.py \
  --baseline-manifest "${STAGE2_MLP_BASELINE_MANIFEST:-${ROOT}/selection_v2/selected_baseline_manifest.json}" \
  --design-root "${STAGE2_MLP_DESIGN_ROOT:-${ROOT}/shared_descriptor_design_v2}" \
  "${INITIAL_ARGS[@]}" \
  "${DIRECTION_ARGS[@]}" \
  --output-dir "${OUT}" \
  "${CHECKPOINT_ARGS[@]}" \
  "${BALANCE_ARGS[@]}" \
  "${ARCHITECTURE_ARGS[@]}" \
  "${REPLAY_ARGS[@]}" \
  "${INVENTORY_ARGS[@]}" \
  --hidden-size "${STAGE2_MLP_HIDDEN_SIZE:-128}" \
  --deep-hidden-size "${STAGE2_MLP_DEEP_HIDDEN_SIZE:-0}" \
  --seed "${STAGE2_MLP_SEED:-20260720}" \
  --steps "${STAGE2_MLP_STEPS:-5000}" \
  --learning-rate "${STAGE2_MLP_LR:-3e-4}" \
  --gradient-clip "${STAGE2_MLP_GRADIENT_CLIP:-10}" \
  --pcgrad-max-curvature-to-replay-norm-ratio "${STAGE2_MLP_PCGRAD_MAX_CURVATURE_RATIO:-inf}" \
  --column-norm-relative-cutoff "${STAGE2_MLP_COLUMN_NORM_RELATIVE_CUTOFF:-1e-8}" \
  --feature-scale-mode "${STAGE2_MLP_FEATURE_SCALE_MODE:-column_norm}" \
  --feature-scale-floor "${STAGE2_MLP_FEATURE_SCALE_FLOOR:-0}" \
  --log-interval "${STAGE2_MLP_LOG_INTERVAL:-100}" \
  --hvp-warmup-steps "${STAGE2_MLP_HVP_WARMUP_STEPS:-0}" \
  --lambda-energy "${STAGE2_MLP_LAMBDA_E:-1}" \
  --lambda-force "${STAGE2_MLP_LAMBDA_F:-1}" \
  --lambda-hessian "${STAGE2_MLP_LAMBDA_H:-1}" \
  --lambda-spectrum "${STAGE2_MLP_LAMBDA_SPEC:-0.1}" \
  --directions-per-parent "${STAGE2_MLP_DIRECTIONS_PER_PARENT:-4}" \
  --absolute-hvp-scale "${STAGE2_MLP_ABSOLUTE_HVP_SCALE:-0.1}" \
  --hvp-relative-loss-fraction "${STAGE2_MLP_RELATIVE_FRACTION:-0.5}" \
  --hvp-reference-floor "${STAGE2_MLP_REFERENCE_FLOOR:-0.05}" \
  --hvp-tail-fraction "${STAGE2_MLP_HVP_TAIL_FRACTION:-0.2}" \
  --hvp-tail-weight "${STAGE2_MLP_HVP_TAIL_WEIGHT:-0}" \
  --spectrum-reference-floor "${STAGE2_MLP_SPECTRUM_FLOOR:-0.01}" \
  --wrong-curvature-multiplier "${STAGE2_MLP_WRONG_CURVATURE_MULTIPLIER:-2}" \
  --energy-median-gate "${STAGE2_MLP_ENERGY_MEDIAN_GATE:-inf}" \
  --energy-max-gate "${STAGE2_MLP_ENERGY_MAX_GATE:-inf}" \
  --force-median-gate "${STAGE2_MLP_FORCE_MEDIAN_GATE:-inf}" \
  --force-max-gate "${STAGE2_MLP_FORCE_MAX_GATE:-inf}" \
  --training-direction-relative-gate 0.05 \
  --heldout-direction-relative-gate 0.15 \
  --full-hessian-median-relative-gate 0.15
