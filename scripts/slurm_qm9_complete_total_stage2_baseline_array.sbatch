#!/usr/bin/env bash
#SBATCH --job-name=ctH_s2base
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/stage2_baseline_%A_%a.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/stage2_baseline_%A_%a.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export DFT_DATA=/scratch/xzh/data
export DFT_MODELS=/scratch/xzh/models
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

ROOT=/scratch/xzh/models/complete_total_capacity/20260717/stage2_direction_v1
MANIFEST=${ROOT}/candidate_direction_manifest.json
PROTOCOL=configs/audit/qm9_complete_total_hessian_direction_generalization_v1.yaml
CANDIDATES=${STAGE2_BASELINE_CANDIDATES:-0124009,0132608,0016142,0061724,0096630,0024609,0049017,0052472,0057862,0082761,0015047,0038511,0057270,0091176,0123863,0091752,0083824,0036972,0053662,0070824,0080472}
CANDIDATES=${CANDIDATES//:/,}
IFS=, read -r -a MOLECULES <<< "${CANDIDATES}"
MOLECULE=${MOLECULES[${SLURM_ARRAY_TASK_ID}]}
BASE_RUN=qm9_hvp_curvature_v1_A_w0ep00_f1em02_seed314159_s1200
RUN_DIR=${DFT_MODELS}/train/runs/${BASE_RUN}
CKPT=${RUN_DIR}/checkpoints/last.ckpt
OUT=${ROOT}/baseline_runs/${MOLECULE}_job${SLURM_JOB_ID}
mkdir -p "${OUT}" /scratch/xzh/logs/complete_total_capacity
LOCAL_CKPT=${SLURM_TMPDIR:-/tmp}/stage2_source_${SLURM_JOB_ID}.ckpt
cp "${CKPT}" "${LOCAL_CKPT}"
CKPT=${LOCAL_CKPT}

/usr/bin/time -v -o "${OUT}/resource.time" \
python scripts/qm9_complete_total_capacity_train.py \
  --protocol "${PROTOCOL}" \
  --manifest "${MANIFEST}" \
  --run "${BASE_RUN}=${RUN_DIR}=${CKPT}" \
  --output-dir "${OUT}" \
  --molecules "${MOLECULE}" \
  --device cuda:0 \
  --max-steps 0 \
  --learning-rate 0 \
  --lambda-e 0 --lambda-f 0 --lambda-rho 0 --lambda-h 0 --lambda-q 0 --lambda-spec 0 \
  --directions-per-step 1 \
  --direction-repeat-steps 1 \
  --eval-interval 1 \
  --gradient-diagnostics-interval 1000000 \
  --density-refresh-interval 100 \
  --density-drift-trigger 1e-4 \
  --density-response-unroll-steps 0 \
  --density-response-unroll-lr 1e-3 \
  --implicit-response-tolerance 3e-5 \
  --implicit-response-max-iterations 2500 \
  --implicit-response-damping 1e-8 \
  --implicit-response-diagonal-probes 0 \
  --integral-derivative-workers 4 \
  --integral-cache-entries 8
