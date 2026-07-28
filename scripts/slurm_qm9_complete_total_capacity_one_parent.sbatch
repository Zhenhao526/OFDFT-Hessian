#!/usr/bin/env bash
#SBATCH --job-name=ctH_cap_1mol
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=220G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/one_parent_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/one_parent_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export DFT_DATA=/scratch/xzh/data
export DFT_MODELS=/scratch/xzh/models
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

ROOT=/scratch/xzh/models/complete_total_capacity/20260717
PROTOCOL=${CAPACITY_PROTOCOL:-configs/audit/qm9_complete_total_hessian_capacity_v1.yaml}
MANIFEST=${CAPACITY_MANIFEST:-${ROOT}/stage1_manifest.json}
BASE_RUN=${CAPACITY_BASE_RUN:-qm9_hvp_curvature_v1_A_w0ep00_f1em02_seed314159_s1200}
RUN_DIR=${CAPACITY_RUN_DIR:-${DFT_MODELS}/train/runs/${BASE_RUN}}
CKPT=${CAPACITY_CKPT:-${RUN_DIR}/checkpoints/last.ckpt}
MOLECULE=${CAPACITY_MOLECULE:-0028399}
MAX_STEPS=${CAPACITY_MAX_STEPS:-50}
LR=${CAPACITY_LR:-3e-6}
ADAM_BETA1=${CAPACITY_ADAM_BETA1:-0.9}
ADAM_BETA2=${CAPACITY_ADAM_BETA2:-0.999}
LAMBDA_E=${CAPACITY_LAMBDA_E:-1}
LAMBDA_F=${CAPACITY_LAMBDA_F:-1}
LAMBDA_RHO=${CAPACITY_LAMBDA_RHO:-0.1}
LAMBDA_H=${CAPACITY_LAMBDA_H:-10}
LAMBDA_Q=${CAPACITY_LAMBDA_Q:-0}
LAMBDA_SPEC=${CAPACITY_LAMBDA_SPEC:-0.1}
DIRECTIONS_PER_STEP=${CAPACITY_DIRECTIONS_PER_STEP:-2}
DIRECTION_LIMIT=${CAPACITY_DIRECTION_LIMIT:-}
EVAL_INTERVAL=${CAPACITY_EVAL_INTERVAL:-${MAX_STEPS}}
DENSITY_REFRESH_INTERVAL=${CAPACITY_DENSITY_REFRESH_INTERVAL:-100}
UNROLL_STEPS=${CAPACITY_UNROLL_STEPS:-0}
UNROLL_LR=${CAPACITY_UNROLL_LR:-1e-3}
IMPLICIT_TOLERANCE=${CAPACITY_IMPLICIT_TOLERANCE:-3e-5}
IMPLICIT_MAX_ITERATIONS=${CAPACITY_IMPLICIT_MAX_ITERATIONS:-2500}
IMPLICIT_DAMPING=${CAPACITY_IMPLICIT_DAMPING:-1e-8}
IMPLICIT_DIAGONAL_PROBES=${CAPACITY_IMPLICIT_DIAGONAL_PROBES:-0}
HVP_LR=${CAPACITY_HVP_LR:-${LR}}
HVP_UPDATE_PERIOD=${CAPACITY_HVP_UPDATE_PERIOD:-2}
REPLAY_UPDATE_PERIOD=${CAPACITY_REPLAY_UPDATE_PERIOD:-}
HVP_LOSS_NORM=${CAPACITY_HVP_LOSS_NORM:-l1}
LOSS_BALANCE_MODE=${CAPACITY_LOSS_BALANCE_MODE:-none}
RESIDUAL_TAG=
if [[ -n "${CAPACITY_THREE_BODY_RESIDUAL:-}" ]]; then
  RESIDUAL_TAG=_threebody
fi
OUT="${ROOT}/one_parent_${MOLECULE}_s${MAX_STEPS}_lr${LR}_h${LAMBDA_H}_q${LAMBDA_Q}${RESIDUAL_TAG}_job${SLURM_JOB_ID}"
mkdir -p "${OUT}" /scratch/xzh/logs/complete_total_capacity

DIRECTION_LIMIT_ARG=()
if [[ -n "${DIRECTION_LIMIT}" ]]; then
  DIRECTION_LIMIT_ARG=(--direction-limit "${DIRECTION_LIMIT}")
fi
CONNECT_MULTIPLIER_ARG=()
if [[ -n "${CAPACITY_CONNECT_MULTIPLIER:-}" ]]; then
  CONNECT_MULTIPLIER_ARG=(--connect-lagrange-multiplier-response)
fi
IMPLICIT_RESPONSE_ARG=()
if [[ -n "${CAPACITY_IMPLICIT_RESPONSE:-}" ]]; then
  IMPLICIT_RESPONSE_ARG=(--implicit-density-parameter-response)
fi
IMPLICIT_WARM_START_ARG=()
if [[ -n "${CAPACITY_IMPLICIT_WARM_START:-}" ]]; then
  IMPLICIT_WARM_START_ARG=(--implicit-response-warm-start)
fi
RESUME_OPTIMIZER_ARG=()
if [[ -n "${CAPACITY_RESUME_OPTIMIZER:-}" ]]; then
  RESUME_OPTIMIZER_ARG=(--resume-capacity-optimizer)
fi
ALTERNATING_HVP_ARG=()
if [[ -n "${CAPACITY_ALTERNATING_HVP:-}" ]]; then
  ALTERNATING_HVP_ARG=(--alternating-hvp-updates)
fi
REPLAY_UPDATE_PERIOD_ARG=()
if [[ -n "${REPLAY_UPDATE_PERIOD}" ]]; then
  REPLAY_UPDATE_PERIOD_ARG=(--replay-update-period "${REPLAY_UPDATE_PERIOD}")
fi
THREE_BODY_ARG=()
if [[ -n "${CAPACITY_THREE_BODY_RESIDUAL:-}" ]]; then
  THREE_BODY_ARG=(
    --three-body-geometry-residual
    --three-body-center-min "${CAPACITY_THREE_BODY_CENTER_MIN:-0.5}"
    --three-body-center-max "${CAPACITY_THREE_BODY_CENTER_MAX:-8.0}"
    --three-body-center-count "${CAPACITY_THREE_BODY_CENTER_COUNT:-4}"
    --three-body-sigma "${CAPACITY_THREE_BODY_SIGMA:-0.5}"
    --three-body-angular-order "${CAPACITY_THREE_BODY_ANGULAR_ORDER:-3}"
  )
fi
if [[ -n "${CAPACITY_THREE_BODY_INITIAL_COEFFICIENTS:-}" ]]; then
  THREE_BODY_ARG+=(
    --three-body-initial-coefficients "${CAPACITY_THREE_BODY_INITIAL_COEFFICIENTS}"
  )
fi
if [[ -n "${CAPACITY_FREEZE_BASE_MODEL:-}" ]]; then
  THREE_BODY_ARG+=(--freeze-base-model)
fi

/usr/bin/time -v -o "${OUT}/resource.time" \
python scripts/qm9_complete_total_capacity_train.py \
  --protocol "${PROTOCOL}" \
  --manifest "${MANIFEST}" \
  --run "${BASE_RUN}=${RUN_DIR}=${CKPT}" \
  --output-dir "${OUT}" \
  --molecules "${MOLECULE}" \
  --device cuda:0 \
  --max-steps "${MAX_STEPS}" \
  --learning-rate "${LR}" \
  --adam-beta1 "${ADAM_BETA1}" \
  --adam-beta2 "${ADAM_BETA2}" \
  "${THREE_BODY_ARG[@]}" \
  "${RESUME_OPTIMIZER_ARG[@]}" \
  "${ALTERNATING_HVP_ARG[@]}" \
  --hvp-learning-rate "${HVP_LR}" \
  --hvp-update-period "${HVP_UPDATE_PERIOD}" \
  "${REPLAY_UPDATE_PERIOD_ARG[@]}" \
  --hvp-loss-norm "${HVP_LOSS_NORM}" \
  --loss-balance-mode "${LOSS_BALANCE_MODE}" \
  --lambda-e "${LAMBDA_E}" \
  --lambda-f "${LAMBDA_F}" \
  --lambda-rho "${LAMBDA_RHO}" \
  --lambda-h "${LAMBDA_H}" \
  --lambda-q "${LAMBDA_Q}" \
  --lambda-spec "${LAMBDA_SPEC}" \
  --directions-per-step "${DIRECTIONS_PER_STEP}" \
  --direction-repeat-steps 8 \
  "${DIRECTION_LIMIT_ARG[@]}" \
  --eval-interval "${EVAL_INTERVAL}" \
  --gradient-diagnostics-interval "${CAPACITY_GRADIENT_DIAGNOSTICS_INTERVAL:-10}" \
  --density-refresh-interval "${DENSITY_REFRESH_INTERVAL}" \
  --density-drift-trigger 1e-4 \
  --density-response-unroll-steps "${UNROLL_STEPS}" \
  --density-response-unroll-lr "${UNROLL_LR}" \
  "${CONNECT_MULTIPLIER_ARG[@]}" \
  "${IMPLICIT_RESPONSE_ARG[@]}" \
  --implicit-response-tolerance "${IMPLICIT_TOLERANCE}" \
  --implicit-response-max-iterations "${IMPLICIT_MAX_ITERATIONS}" \
  --implicit-response-damping "${IMPLICIT_DAMPING}" \
  --implicit-response-diagonal-probes "${IMPLICIT_DIAGONAL_PROBES}" \
  "${IMPLICIT_WARM_START_ARG[@]}" \
  --integral-derivative-workers 4 \
  --integral-cache-entries 8

printf '%s\n' "${OUT}" > "${ROOT}/latest_one_parent_output.txt"
