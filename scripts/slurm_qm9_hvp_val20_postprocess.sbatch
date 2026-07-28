#!/usr/bin/env bash
#SBATCH --job-name=hvp_val_post
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/xzh/logs/hvp_curvature/val20_post_%j.out
#SBATCH --error=/scratch/xzh/logs/hvp_curvature/val20_post_%j.err

set -euo pipefail
cd /scratch/xzh/code/structures25
source /scratch/xzh/env.sh

ROOT=/scratch/xzh/models/hvp_branch_stability/20260716
VAL8="${ROOT}/formal_v2/validation_diagnostic_tasks"
VAL12="${ROOT}/formal_val20_remaining_v2/validation_remaining_tasks"
OUTPUT="${ROOT}/formal_val20_v3"
EXPECTED="${OUTPUT}/validation20.ids"
mkdir -p "${OUTPUT}"
python - <<'PY'
from pathlib import Path
import json
source = Path('/scratch/xzh/models/eval/qm9_random1000_validation_protocol/validation_representatives_20.json')
records = json.loads(source.read_text())
ids = sorted(str(record['molecule_id']).zfill(7) for record in records)
assert len(ids) == 20 and len(set(ids)) == 20
Path('/scratch/xzh/models/hvp_branch_stability/20260716/formal_val20_v3/validation20.ids').write_text('\n'.join(ids) + '\n')
PY

SUMMARY_COUNT="$(( $(find "${VAL8}" -name summary.json | wc -l) + $(find "${VAL12}" -name summary.json | wc -l) ))"
[[ "${SUMMARY_COUNT}" -eq 240 ]] || {
  echo "Expected 240 validation20 summaries, found ${SUMMARY_COUNT}" >&2
  exit 1
}

python scripts/qm9_hvp_branch_stability_analysis.py \
  --protocol configs/audit/qm9_hvp_branch_stability_v2.yaml \
  --task-root "${VAL8}" --task-root "${VAL12}" \
  --output-dir "${OUTPUT}/analysis_v2" \
  --expected-molecules "${EXPECTED}" --expected-directions 4

python scripts/qm9_hvp_branch_stability_analysis.py \
  --protocol configs/audit/qm9_hvp_branch_stability_v3.yaml \
  --task-root "${VAL8}" --task-root "${VAL12}" \
  --output-dir "${OUTPUT}/analysis_v3" \
  --expected-molecules "${EXPECTED}" --expected-directions 4 \
  --input-sidecar-dir "${ROOT}/multidirection_sidecars/sidecars" \
  --output-sidecar-dir "${OUTPUT}/analysis_v3/stable_sidecars"
