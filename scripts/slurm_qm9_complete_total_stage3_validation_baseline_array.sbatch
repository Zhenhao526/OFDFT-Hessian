#!/usr/bin/env bash
#SBATCH --job-name=ctH_s3valbase
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/stage3_validation_baseline_%A_%a.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/stage3_validation_baseline_%A_%a.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export DFT_DATA=/scratch/xzh/data
export DFT_MODELS=/scratch/xzh/models
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

STAGE3_ROOT=${STAGE3_ROOT:-/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1}
ASSET_DIR=${STAGE3_ASSET_DIR:-${STAGE3_ROOT}/assets_v3_uniform_A}
MANIFEST=${STAGE3_VALIDATION_MANIFEST:-${ASSET_DIR}/validation_capacity_manifest.json}
PROTOCOL=${STAGE3_VALIDATION_PROTOCOL:-configs/audit/qm9_complete_total_hessian_direction_generalization_v1.yaml}
OUTPUT_ROOT=${STAGE3_VALIDATION_OUTPUT_ROOT:-${STAGE3_ROOT}/validation_baseline_runs_v1_uniform_A}
BASE_RUN=${STAGE3_BASE_RUN:-qm9_hvp_curvature_v1_A_w0ep00_f1em02_seed314159_s1200}
RUN_DIR=${DFT_MODELS}/train/runs/${BASE_RUN}
SOURCE_CKPT=${RUN_DIR}/checkpoints/last.ckpt
export MANIFEST

MOLECULE=$(python -c 'import json, os; from pathlib import Path; rows=json.loads(Path(os.environ["MANIFEST"]).read_text())["parents"]; print(rows[int(os.environ["SLURM_ARRAY_TASK_ID"])]["molecule_id"])')
OUT=${OUTPUT_ROOT}/${MOLECULE}_job${SLURM_JOB_ID}
mkdir -p "${OUT}" /scratch/xzh/logs/complete_total_capacity

LOCAL_CKPT=${SLURM_TMPDIR:-/tmp}/stage3_validation_source_${SLURM_JOB_ID}.ckpt
cp "${SOURCE_CKPT}" "${LOCAL_CKPT}"

/usr/bin/time -v -o "${OUT}/resource.time" \
python scripts/qm9_complete_total_capacity_train.py \
  --protocol "${PROTOCOL}" \
  --manifest "${MANIFEST}" \
  --run "${BASE_RUN}=${RUN_DIR}=${LOCAL_CKPT}" \
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
