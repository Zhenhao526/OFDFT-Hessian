#!/usr/bin/env bash
#SBATCH --job-name=ct_cang4_vib
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail

source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

run_dir=${V7_RUN_DIR:-/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/local_central_angular_four_body_capacity_v7/exact_ridge_v1}
summary=${run_dir}/summary.json
baseline=/scratch/xzh/models/complete_total_capacity/20260717/five_parent_baseline_manifest_stable5_v2.json

python - "${summary}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
if payload.get("gate", {}).get("passed") is not True:
    raise SystemExit(f"refusing vibration closure: stable5 gate did not pass in {path}")
if payload.get("validation_accessed") is not False:
    raise SystemExit(f"refusing vibration closure: validation access not frozen in {path}")
if payload.get("test100_accessed") is not False:
    raise SystemExit(f"refusing vibration closure: Test100 access not frozen in {path}")
if int(payload.get("test100_evaluations_used", 0)) != 0:
    raise SystemExit(f"refusing vibration closure: Test100 evaluation count is nonzero in {path}")
PY

python scripts/qm9_hessian_vibrational_metrics.py \
  --manifest-json "${baseline}" \
  --dataset-dir /scratch/xzh/data/QM9PBEForceRandom1000 \
  --baseline-result "original_A=${baseline}" \
  --capacity-result "central_angular_four_body_v7=${run_dir}" \
  --output-dir "${run_dir}/vibrational_v1"
