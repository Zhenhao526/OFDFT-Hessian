#!/usr/bin/env bash
#SBATCH --job-name=ct_post_v4_diag
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail

source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

root=${POST_V4_SOURCE_ROOT:-/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/local_scalar_three_task_pcgrad_v4}
protocol=${POST_V4_SOURCE_PROTOCOL:-configs/audit/qm9_complete_total_hessian_local_scalar_three_task_pcgrad_v4.yaml}
analysis=${POST_V4_SOURCE_ANALYSIS:-${root}/formal_analysis_v1/summary.json}
diagnostic_root=${POST_V4_DIAGNOSTIC_ROOT:-${root}/post_v4_diagnostics_v1}
manifest=${diagnostic_root}/audit_manifest.json

should_run=$(python - "${analysis}" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
if result.get("test100_accessed") is not False:
    raise SystemExit("v4 analysis does not certify frozen Test100")
if int(result.get("test100_evaluations_used", -1)) != 0:
    raise SystemExit("v4 analysis Test100 count is nonzero")
print("false" if bool(result.get("parent_cv_design_authorized")) else "true")
PY
)
if [[ "${should_run}" != true ]]; then
  echo "v4 passed stable5; post-failure diagnostics are not authorized."
  exit 0
fi

mkdir -p "${diagnostic_root}"
freeze_args=(
  --protocol "${protocol}"
  --formal-analysis "${analysis}"
  --run-root "${root}/formal"
  --output "${manifest}"
)
if [[ -n "${POST_V4_SOURCE_ARM:-}" ]]; then
  freeze_args+=(--source-arm-id "${POST_V4_SOURCE_ARM}")
fi
python scripts/qm9_complete_total_freeze_local_scalar_post_v4_diagnostics.py "${freeze_args[@]}"

readarray -t source < <(python - "${manifest}" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
print(result["source_arm_id"])
print(result["source_checkpoint"])
PY
)
arm=${source[0]}
checkpoint=${source[1]}

jacobian_job=$(JACOBIAN_PROTOCOL="${protocol}" \
  JACOBIAN_ARM="${arm}" \
  JACOBIAN_CHECKPOINT="${checkpoint}" \
  JACOBIAN_OUTPUT="${diagnostic_root}/jacobian_range_30iter" \
  JACOBIAN_ITERATIONS=30 \
  sbatch --parsable --export=ALL \
  scripts/slurm_qm9_complete_total_local_scalar_jacobian_range_audit.sbatch)

for objective in joint hessian_only; do
  output=${diagnostic_root}/lbfgs_${objective}_100iter
  job=$(LBFGS_SOURCE_PROTOCOL="${protocol}" \
    LBFGS_SOURCE_ARM="${arm}" \
    LBFGS_SOURCE_CHECKPOINT="${checkpoint}" \
    LBFGS_AUDIT_MANIFEST="${manifest}" \
    LBFGS_OBJECTIVE="${objective}" \
    LBFGS_OUTPUT="${output}" \
    LBFGS_ITERATIONS=100 \
    sbatch --parsable --export=ALL \
    scripts/slurm_qm9_complete_total_local_scalar_lbfgs_ceiling.sbatch)
  if [[ "${objective}" == joint ]]; then
    joint_job=${job}
  else
    hessian_job=${job}
  fi
done

export POST_V4_DIAGNOSTIC_ROOT="${diagnostic_root}"
vibrational_job=$(sbatch --parsable \
  --dependency="afterok:${joint_job}:${hessian_job}" \
  scripts/slurm_qm9_complete_total_local_scalar_post_v4_vibrational.sbatch)
analysis_job=$(sbatch --parsable \
  --dependency="afterok:${jacobian_job}:${joint_job}:${hessian_job}:${vibrational_job}" \
  scripts/slurm_qm9_complete_total_local_scalar_post_v4_diagnostics_analysis.sbatch)
echo "arm=${arm} manifest=${manifest} jacobian_job=${jacobian_job} joint_lbfgs_job=${joint_job} hessian_lbfgs_job=${hessian_job} vibrational_job=${vibrational_job} analysis_job=${analysis_job}"
