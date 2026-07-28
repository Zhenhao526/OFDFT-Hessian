#!/usr/bin/env bash
set -euo pipefail

root=${QM9_NODE02_ROOT:-/home/shenwei01/xzh_node02_20260724}
repo=${QM9_NODE02_REPO:-${root}/work/structures25}
protocol=${QM9_ANALYTIC_HVP_PROTOCOL:-${repo}/configs/audit/qm9_graphformer_complete_total_analytic_relaxed_hvp_v2.yaml}
output=${QM9_ANALYTIC_HVP_OUTPUT:-${root}/runs/graphformer_analytic_hvp/0028399_$(date +%Y%m%d_%H%M%S)}

source "${repo}/scripts/activate_qm9_node02_local.sh"
cd "${repo}"
mkdir -p "${output}"

gpu=${QM9_ANALYTIC_HVP_GPU:-$(
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

exec /usr/bin/time -v -o "${output}/resource.time" \
  python scripts/qm9_graphformer_analytic_relaxed_hvp_audit.py \
    --protocol "${protocol}" \
    --output-dir "${output}" \
    --molecule 0028399 \
    --direction-index 0 \
    --device cuda:0
