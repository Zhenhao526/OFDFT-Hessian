#!/usr/bin/env bash
#SBATCH --job-name=ctH_sp_ext
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/spectral_external_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/spectral_external_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

ROOT=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1
PROTOCOL=${SPECTRAL_EXTERNAL_PROTOCOL:-/scratch/xzh/code/structures25/configs/audit/qm9_complete_total_hessian_unseen_parent_shared_operator_v1.yaml}
BASELINE=${SPECTRAL_EXTERNAL_BASELINE:-${ROOT}/validation_baselines_v1_uniform_A/validation_baseline_manifest.json}
SOURCE=${SPECTRAL_EXTERNAL_SOURCE:-${ROOT}/validation_external_eval_h512_source_v1}
CANDIDATE=${SPECTRAL_EXTERNAL_CANDIDATE:-${ROOT}/spectral_block_parent_conditioned_symmetric_completion_v5_v1}
OUTPUT=${SPECTRAL_EXTERNAL_OUTPUT:-${ROOT}/validation_spectral_block_shared_operator_v1}
mkdir -p "${OUTPUT}" /scratch/xzh/logs/complete_total_capacity

/usr/bin/time -v -o "${OUTPUT}/resource.time" \
python scripts/qm9_complete_total_spectral_operator_external_eval.py \
  --protocol "${PROTOCOL}" \
  --baseline-manifest "${BASELINE}" \
  --source-run-dir "${SOURCE}" \
  --candidate-run-dir "${CANDIDATE}" \
  --output-dir "${OUTPUT}"
