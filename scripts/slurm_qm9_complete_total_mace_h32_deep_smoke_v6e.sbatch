#!/usr/bin/env bash
#SBATCH --job-name=ct_mace_h32smk
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export PYTHONPATH=/scratch/xzh/vendor/mace_torch_0_3_16
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

analysis=/scratch/xzh/models/complete_total_capacity/20260717/stage5_mace_h16_scaled_v6d/one_parent_lbfgs_v1/analysis_v1/summary.json
decision=$(python - "${analysis}" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
if summary["test100_accessed"] is not False:
    raise SystemExit("one-parent prerequisite does not prove Test100 freeze")
if summary["parent_cv_authorized"] is not False:
    raise SystemExit("one-parent prerequisite improperly authorizes parent CV")
if summary["single_parent_capacity_passed"] is True:
    print("skip")
    raise SystemExit
if summary["decision"] != "single_parent_span_not_supported_at_registered_budget":
    raise SystemExit("unexpected one-parent decision")
print("run")
PY
)
if [[ "${decision}" == "skip" ]]; then
  echo "h16 passes every one-parent gate; h32 fallback is not authorized"
  exit 0
fi
if [[ "${decision}" != "run" ]]; then
  echo "unexpected h32 prerequisite decision: ${decision}" >&2
  exit 1
fi
echo "h16 one-parent ceiling failed; running preregistered h32 capacity smoke"

root=/scratch/xzh/models/complete_total_capacity/20260717/stage5_mace_h32_deep_v6e
output_dir="${root}/smoke/E11e_mace_h32_l3_corr3_int3_cut20_scale1000"
mkdir -p "${output_dir}" /scratch/xzh/logs/complete_total_capacity

/usr/bin/time -v -o "${output_dir}/resource.time" \
  python scripts/qm9_complete_total_local_scalar_full_hessian_capacity.py \
    --protocol configs/audit/qm9_complete_total_hessian_mace_h32_deep_capacity_v6e.yaml \
    --arm-id E11e_mace_h32_l3_corr3_int3_cut20_scale1000 \
    --run-mode smoke \
    --output-dir "${output_dir}" \
    --device cuda:0
