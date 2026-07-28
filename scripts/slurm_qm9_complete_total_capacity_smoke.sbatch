#!/usr/bin/env bash
#SBATCH --job-name=ctH_cap_smoke
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/smoke_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/smoke_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export DFT_DATA=/scratch/xzh/data
export DFT_MODELS=/scratch/xzh/models
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

ROOT=/scratch/xzh/models/complete_total_capacity/20260717
BASE_RUN=qm9_hvp_curvature_v1_A_w0ep00_f1em02_seed314159_s1200
RUN_DIR="${DFT_MODELS}/train/runs/${BASE_RUN}"
CKPT="${RUN_DIR}/checkpoints/last.ckpt"
OUT="${ROOT}/smoke_0028399_job${SLURM_JOB_ID}"
mkdir -p "${OUT}" /scratch/xzh/logs/complete_total_capacity

/usr/bin/time -v -o "${OUT}/resource.time" \
python scripts/qm9_complete_total_capacity_train.py \
  --protocol configs/audit/qm9_complete_total_hessian_capacity_v1.yaml \
  --manifest "${ROOT}/stage1_manifest.json" \
  --run "${BASE_RUN}=${RUN_DIR}=${CKPT}" \
  --output-dir "${OUT}" \
  --molecules 0028399 \
  --device cuda:0 \
  --max-steps 1 \
  --learning-rate "${SMOKE_LR:-3e-6}" \
  --lambda-e "${SMOKE_LAMBDA_E:-1}" \
  --lambda-f "${SMOKE_LAMBDA_F:-1}" \
  --lambda-rho "${SMOKE_LAMBDA_RHO:-0.1}" \
  --lambda-h "${SMOKE_LAMBDA_H:-10}" \
  --lambda-q "${SMOKE_LAMBDA_Q:-0}" \
  --lambda-spec "${SMOKE_LAMBDA_SPEC:-0.1}" \
  --directions-per-step 1 \
  --direction-limit 1 \
  --eval-interval 1 \
  --gradient-diagnostics-interval "${SMOKE_GRADIENT_DIAGNOSTICS_INTERVAL:-1}" \
  --density-refresh-interval 25 \
  --density-drift-trigger 1e-4 \
  --density-response-unroll-steps "${SMOKE_UNROLL_STEPS:-0}" \
  --density-response-unroll-lr "${SMOKE_UNROLL_LR:-1e-3}" \
  ${SMOKE_CONNECT_MULTIPLIER:+--connect-lagrange-multiplier-response} \
  ${SMOKE_IMPLICIT_RESPONSE:+--implicit-density-parameter-response} \
  --implicit-response-tolerance "${SMOKE_IMPLICIT_TOLERANCE:-1e-8}" \
  --implicit-response-max-iterations "${SMOKE_IMPLICIT_MAX_ITERATIONS:-500}" \
  --implicit-response-damping "${SMOKE_IMPLICIT_DAMPING:-1e-8}" \
  --implicit-response-diagonal-probes "${SMOKE_IMPLICIT_DIAGONAL_PROBES:-0}" \
  --integral-derivative-workers 4 \
  --integral-cache-entries 4

printf '%s\n' "${OUT}" > "${ROOT}/latest_smoke_output.txt"
