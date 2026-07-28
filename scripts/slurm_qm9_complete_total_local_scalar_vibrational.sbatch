#!/usr/bin/env bash
#SBATCH --job-name=ct_local_vib
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

root=${LOCAL_SCALAR_VIB_ROOT:?LOCAL_SCALAR_VIB_ROOT is required}
selection=${LOCAL_SCALAR_VIB_SELECTION:-${root}/smoke_analysis_v1/summary.json}
output=${LOCAL_SCALAR_VIB_OUTPUT:-${root}/formal_vibrational_v1}
baseline=/scratch/xzh/models/complete_total_capacity/20260717/five_parent_baseline_manifest_stable5_v2.json
dataset=/scratch/xzh/data/QM9PBEForceRandom1000
baseline_only=${LOCAL_SCALAR_VIB_BASELINE_ONLY:-0}

arms=()
if [[ "${baseline_only}" != 1 ]]; then
  mapfile -t arms < <(python - "${selection}" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
if result.get("test100_accessed") is not False:
    raise SystemExit("smoke selection does not certify frozen Test100")
if int(result.get("test100_evaluations_used", -1)) != 0:
    raise SystemExit("smoke selection Test100 count is not zero")
for arm in result.get("selected_formal_arms", []):
    print(arm)
PY
  )
  if [[ ${#arms[@]} -eq 0 ]]; then
    echo "No selected formal arms for vibrational postprocessing." >&2
    exit 2
  fi
fi

result_args=(--baseline-result "original_A=${baseline}")
for arm in "${arms[@]}"; do
  result_args+=(--capacity-result "${arm}=${root}/formal/${arm}")
done
python scripts/qm9_hessian_vibrational_metrics.py \
  --manifest-json "${baseline}" \
  --dataset-dir "${dataset}" \
  "${result_args[@]}" \
  --output-dir "${output}"
