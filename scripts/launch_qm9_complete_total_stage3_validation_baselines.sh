#!/usr/bin/env bash
set -euo pipefail

source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

STAGE3_ROOT=${STAGE3_ROOT:-/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1}
ASSET_DIR=${STAGE3_ASSET_DIR:-${STAGE3_ROOT}/assets_v3_uniform_A}
MANIFEST=${STAGE3_VALIDATION_MANIFEST:-${ASSET_DIR}/validation_capacity_manifest.json}
OUTPUT_ROOT=${STAGE3_VALIDATION_OUTPUT_ROOT:-${STAGE3_ROOT}/validation_baseline_runs_v1_uniform_A}
export MANIFEST STAGE3_ROOT STAGE3_ASSET_DIR="${ASSET_DIR}" STAGE3_VALIDATION_OUTPUT_ROOT="${OUTPUT_ROOT}"

PARENT_COUNT=$(python -c 'import json, os; from pathlib import Path; print(len(json.loads(Path(os.environ["MANIFEST"]).read_text())["parents"]))')
if (( PARENT_COUNT < 1 )); then
  echo "validation manifest contains no parents" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" /scratch/xzh/logs/complete_total_capacity
sbatch --array="0-$((PARENT_COUNT - 1))%${STAGE3_VALIDATION_CONCURRENCY:-7}" \
  --export=ALL,MANIFEST="${MANIFEST}",STAGE3_ROOT="${STAGE3_ROOT}",STAGE3_ASSET_DIR="${ASSET_DIR}",STAGE3_VALIDATION_OUTPUT_ROOT="${OUTPUT_ROOT}" \
  scripts/slurm_qm9_complete_total_stage3_validation_baseline_array.sbatch
