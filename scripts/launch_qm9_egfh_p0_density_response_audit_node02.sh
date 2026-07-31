#!/usr/bin/env bash
set -euo pipefail

root=${QM9_NODE02_ROOT:-/home/shenwei01/xzh_node02_20260724}
repo=${QM9_NODE02_REPO:-${root}/work/structures25}
protocol=${QM9_EGFH_P0_PROTOCOL:-${repo}/configs/audit/qm9_egfh_p0_density_response_audit_v1.yaml}
output=${QM9_EGFH_P0_OUTPUT:-${root}/runs/qm9_egfh_p0_density_response_audit_v1/$(date +%Y%m%d_%H%M%S)}

source "${repo}/scripts/activate_qm9_node02_local.sh"
cd "${repo}"
mkdir -p "$(dirname "${output}")"

gpu=${QM9_EGFH_P0_GPU:-$(
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits |
    awk -F, '
      $2 + 0 < 100 && $3 + 0 < 5 && first == "" {
        gsub(/ /, "", $1)
        first = $1
      }
      END {print first}
    '
)}
if [[ -z "${gpu}" ]]; then
  echo "No idle GPU satisfies memory<100 MiB and utilization<5%." >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES="${gpu}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export MLDFT_SYMMETRIC_MATRIX_POWER_MODE=eigh_second_order_audit

exec /usr/bin/time -v -o "${output}.resource.time" \
  python scripts/qm9_egfh_p0_density_response_audit.py \
    --protocol "${protocol}" \
    --repo "${repo}" \
    --output-dir "${output}" \
    --device cuda:0
