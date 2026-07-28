#!/usr/bin/env bash
#SBATCH --job-name=ctH_s3valmerge
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/stage3_validation_merge_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/stage3_validation_merge_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

ROOT=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1
ASSETS=${ROOT}/assets_v3_uniform_A
RUN_ROOT=${ROOT}/validation_baseline_runs_v1_uniform_A
OUTPUT=${ROOT}/validation_baselines_v1_uniform_A
PROVENANCE=${OUTPUT}/provenance
SOURCE_RUN=qm9_hvp_curvature_v1_A_w0ep00_f1em02_seed314159_s1200
SOURCE_CKPT=/scratch/xzh/models/train/runs/${SOURCE_RUN}/checkpoints/last.ckpt

python scripts/qm9_complete_total_stage2_provenance_audit.py \
  --source-checkpoint "${SOURCE_CKPT}" \
  --expected-source-run-name "${SOURCE_RUN}" \
  --run-root "${RUN_ROOT}" \
  --output-dir "${PROVENANCE}" \
  --require-all-match

python scripts/qm9_complete_total_validation_baseline_merge.py \
  --protocol configs/audit/qm9_complete_total_hessian_unseen_parent_v1.yaml \
  --capacity-manifest "${ASSETS}/validation_capacity_manifest.json" \
  --source-split /scratch/xzh/data/QM9PBEForceRandom1000/split.pkl \
  --run-root "${RUN_ROOT}" \
  --provenance-audit "${PROVENANCE}/provenance_audit.json" \
  --output-dir "${OUTPUT}" \
  --require-complete
