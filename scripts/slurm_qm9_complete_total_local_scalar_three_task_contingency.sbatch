#!/usr/bin/env bash
#SBATCH --job-name=ct_3task_contingency
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail

source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

activation=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/local_scalar_activation_audit_v3/formal_analysis_v1/summary.json
should_run=$(python - "${activation}" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
if result.get("test100_accessed") is not False:
    raise SystemExit("activation analysis does not certify frozen Test100")
if int(result.get("test100_evaluations_used", -1)) != 0:
    raise SystemExit("activation analysis Test100 count is not zero")
print("false" if bool(result.get("parent_cv_design_authorized")) else "true")
PY
)

if [[ "${should_run}" != true ]]; then
  echo "Activation audit passed; three-task contingency is not authorized."
  exit 0
fi

smoke_job=$(sbatch --parsable \
  --export=ALL,THREE_TASK_RUN_MODE=smoke \
  scripts/slurm_qm9_complete_total_local_scalar_three_task_pcgrad.sbatch)
selection_job=$(sbatch --parsable \
  --dependency="afterok:${smoke_job}" \
  scripts/slurm_qm9_complete_total_local_scalar_three_task_smoke_analysis.sbatch)
formal_submit_job=$(sbatch --parsable \
  --dependency="afterok:${selection_job}" \
  scripts/slurm_qm9_complete_total_local_scalar_three_task_formal_orchestrate.sbatch)
echo "smoke_job=${smoke_job} selection_job=${selection_job} formal_submit_job=${formal_submit_job}"
