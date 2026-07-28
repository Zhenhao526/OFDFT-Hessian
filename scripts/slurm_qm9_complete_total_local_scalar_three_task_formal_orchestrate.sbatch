#!/usr/bin/env bash
#SBATCH --job-name=ct_3task_formal_submit
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

root=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/local_scalar_three_task_pcgrad_v4
selection=${root}/smoke_analysis_v1/summary.json
arm=$(python - "${selection}" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
if result.get("test100_accessed") is not False:
    raise SystemExit("three-task smoke analysis does not certify frozen Test100")
if int(result.get("test100_evaluations_used", -1)) != 0:
    raise SystemExit("three-task smoke analysis Test100 count is not zero")
selected = result.get("selected_formal_arms", [])
if len(selected) > 1:
    raise SystemExit("three-task smoke selected more than one formal arm")
print(selected[0] if selected else "")
PY
)

if [[ -z "${arm}" ]]; then
  fallback_root=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/local_scalar_activation_audit_v3
  fallback_job=$(POST_V4_SOURCE_ROOT="${fallback_root}" \
    POST_V4_SOURCE_PROTOCOL=configs/audit/qm9_complete_total_hessian_local_scalar_activation_audit_v3.yaml \
    POST_V4_SOURCE_ANALYSIS="${fallback_root}/formal_analysis_v1/summary.json" \
    POST_V4_DIAGNOSTIC_ROOT="${root}/post_v4_diagnostics_no_formal_v1" \
    sbatch --parsable --export=ALL \
    scripts/slurm_qm9_complete_total_local_scalar_post_v4_diagnostics.sbatch)
  echo "No eligible three-task formal arm; fallback_diagnostic_job=${fallback_job}."
  exit 0
fi
case "${arm}" in
  K3_softplus_beta5_three_task) array_index=0 ;;
  K5_tanh_three_task) array_index=1 ;;
  *) echo "Unknown selected arm: ${arm}" >&2; exit 2 ;;
esac

formal_job=$(sbatch --parsable \
  --array="${array_index}" \
  --export=ALL,THREE_TASK_RUN_MODE=formal,THREE_TASK_ARM_ID="${arm}" \
  scripts/slurm_qm9_complete_total_local_scalar_three_task_pcgrad.sbatch)
analysis_job=$(sbatch --parsable \
  --dependency="afterok:${formal_job}" \
  --export=ALL,LOCAL_SCALAR_FORMAL_ROOT="${root}",LOCAL_SCALAR_FORMAL_PROTOCOL=configs/audit/qm9_complete_total_hessian_local_scalar_three_task_pcgrad_v4.yaml,LOCAL_SCALAR_FORMAL_SMOKE_SUMMARY="${selection}" \
  scripts/slurm_qm9_complete_total_local_scalar_activation_formal_analysis.sbatch)
vibrational_job=$(sbatch --parsable \
  --dependency="afterok:${analysis_job}" \
  --export=ALL,LOCAL_SCALAR_VIB_ROOT="${root}",LOCAL_SCALAR_VIB_SELECTION="${selection}" \
  scripts/slurm_qm9_complete_total_local_scalar_vibrational.sbatch)
diagnostic_job=$(sbatch --parsable \
  --dependency="afterok:${analysis_job}" \
  scripts/slurm_qm9_complete_total_local_scalar_post_v4_diagnostics.sbatch)
echo "formal_job=${formal_job} analysis_job=${analysis_job} vibrational_job=${vibrational_job} diagnostic_job=${diagnostic_job} arm=${arm}"
