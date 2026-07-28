#!/usr/bin/env bash
set -euo pipefail

cd /scratch/xzh/code/structures25

ROOT=/scratch/xzh/models/complete_total_capacity/20260717
CAPACITY_ARRAY=${THREE_BODY_SCAN_CAPACITY_ARRAY:-${ROOT}/one_parent_0028399_s0_lr0_h0_q0_job3202/hessian_arrays/step_0000200_0028399.npz}
BASELINE_TOTAL_ENERGY=${THREE_BODY_SCAN_BASELINE_TOTAL_ENERGY:--449.85565940420287}
OUTPUT_ROOT=${THREE_BODY_SCAN_OUTPUT_ROOT:-${ROOT}/geometry_residual_three_body_full_kernel_scan}
mkdir -p "${OUTPUT_ROOT}"

specifications=(
  "c4_l4_s0p50 4 4 0.50"
  "c5_l3_s0p50 5 3 0.50"
  "c5_l4_s0p35 5 4 0.35"
  "c5_l4_s0p50 5 4 0.50"
  "c5_l4_s0p75 5 4 0.75"
  "c6_l4_s0p50 6 4 0.50"
)

{
  printf 'job_id,name,center_count,angular_order,sigma_bohr\n'
  for specification in "${specifications[@]}"; do
    read -r name center_count angular_order sigma <<<"${specification}"
    output_dir="${OUTPUT_ROOT}/${name}"
    job_id=$(THREE_BODY_CAPACITY_ARRAY="${CAPACITY_ARRAY}" \
      THREE_BODY_OUTPUT_DIR="${output_dir}" \
      THREE_BODY_BASELINE_TOTAL_ENERGY="${BASELINE_TOTAL_ENERGY}" \
      THREE_BODY_CENTER_COUNT="${center_count}" \
      THREE_BODY_ANGULAR_ORDER="${angular_order}" \
      THREE_BODY_SIGMA="${sigma}" \
      sbatch --parsable --time=02:00:00 --mem=64G \
        --export=ALL,THREE_BODY_CAPACITY_ARRAY,THREE_BODY_OUTPUT_DIR,THREE_BODY_BASELINE_TOTAL_ENERGY,THREE_BODY_CENTER_COUNT,THREE_BODY_ANGULAR_ORDER,THREE_BODY_SIGMA \
        scripts/slurm_qm9_complete_total_geometry_three_body_capacity.sbatch)
    printf '%s,%s,%s,%s,%s\n' \
      "${job_id}" "${name}" "${center_count}" "${angular_order}" "${sigma}"
  done
} | tee "${OUTPUT_ROOT}/submitted_jobs.csv"
