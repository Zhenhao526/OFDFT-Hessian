#!/usr/bin/env bash
#SBATCH --job-name=ctH_pair_scan
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/pair_scan_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/pair_scan_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

ROOT=/scratch/xzh/models/complete_total_capacity/20260717
MANIFEST=${ROOT}/stage1_manifest.json
MOLECULE=${PAIR_SCAN_MOLECULE:-0028399}
CAPACITY_ARRAY=${PAIR_SCAN_CAPACITY_ARRAY:-${ROOT}/one_parent_0028399_s50_lr1e-6_h1_q0_job3175/hessian_arrays/step_0000200_0028399.npz}
OUTPUT_ROOT=${PAIR_SCAN_OUTPUT_ROOT:-${ROOT}/geometry_residual_pair_scan_step200}
mkdir -p "${OUTPUT_ROOT}" /scratch/xzh/logs/complete_total_capacity

for center_count in ${PAIR_SCAN_CENTER_COUNTS:-12 20 32 48 64}; do
  for sigma in ${PAIR_SCAN_SIGMAS:-0.15 0.25 0.35 0.50 0.75 1.00}; do
    output_dir="${OUTPUT_ROOT}/n${center_count}_s${sigma}"
    python scripts/qm9_complete_total_geometry_residual_capacity.py \
      --manifest "${MANIFEST}" \
      --molecule-id "${MOLECULE}" \
      --capacity-array "${CAPACITY_ARRAY}" \
      --output-dir "${output_dir}" \
      --center-count "${center_count}" \
      --sigma "${sigma}"
  done
done

python - "${OUTPUT_ROOT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = [json.loads(path.read_text()) for path in root.glob("*/geometry_residual_capacity.json")]
rows.sort(key=lambda row: row["corrected_relative_frobenius"])
(root / "scan_summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
with (root / "scan_summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(rows[:5], indent=2, sort_keys=True))
PY
