#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/scratch/xzh/models/complete_total_capacity/20260717}
MODE=${1:-smoke}
cd /scratch/xzh/code/structures25
source /scratch/xzh/env.sh
mkdir -p "${ROOT}" /scratch/xzh/logs/complete_total_capacity

python scripts/prepare_qm9_complete_total_capacity_stage1.py \
  --protocol configs/audit/qm9_complete_total_hessian_capacity_v1.yaml \
  --dataset-dir /scratch/xzh/data/QM9PBEForceRandom1000 \
  --reference-dir /scratch/xzh/models/hvp100/20260716/pbe_hessians/cache \
  --stability-csv /scratch/xzh/models/hvp_branch_stability/20260716/formal_train100_v2/analysis/parent_stability.csv \
  --output "${ROOT}/stage1_manifest.json"

case "${MODE}" in
  smoke)
    sbatch scripts/slurm_qm9_complete_total_capacity_smoke.sbatch
    ;;
  one-parent)
    sbatch scripts/slurm_qm9_complete_total_capacity_one_parent.sbatch
    ;;
  *)
    printf 'Unsupported mode: %s\n' "${MODE}" >&2
    exit 2
    ;;
esac
