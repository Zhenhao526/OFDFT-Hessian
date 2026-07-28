#!/usr/bin/env bash
set -euo pipefail

cd /scratch/xzh/code/structures25

ROOT=/scratch/xzh/models/complete_total_capacity/20260717
CAPACITY_ARRAY=${FOUR_BODY_SCAN_CAPACITY_ARRAY:-${ROOT}/geometry_residual_three_body_full_kernel_scan/c6_l4_s0p50/geometry_three_body_capacity.npz}
OUTPUT_ROOT=${FOUR_BODY_SCAN_OUTPUT_ROOT:-${ROOT}/geometry_residual_four_body_full_kernel_scan}
mkdir -p "${OUTPUT_ROOT}"

specifications=(
  "c3_l4_s0p75_b1p35 3 4 0.75 1.35"
  "c3_l4_s1p00_b1p35 3 4 1.00 1.35"
  "c4_l4_s0p75_b1p35 4 4 0.75 1.35"
  "c4_l5_s0p75_b1p35 4 5 0.75 1.35"
  "c4_l4_s0p75_b1p25 4 4 0.75 1.25"
  "c4_l4_s0p75_b1p45 4 4 0.75 1.45"
)

{
  printf 'job_id,name,center_count,torsion_order,sigma_bohr,bond_scale\n'
  for specification in "${specifications[@]}"; do
    read -r name center_count torsion_order sigma bond_scale <<<"${specification}"
    output_dir="${OUTPUT_ROOT}/${name}"
    job_id=$(FOUR_BODY_CAPACITY_ARRAY="${CAPACITY_ARRAY}" \
      FOUR_BODY_OUTPUT_DIR="${output_dir}" \
      FOUR_BODY_CENTER_COUNT="${center_count}" \
      FOUR_BODY_TORSION_ORDER="${torsion_order}" \
      FOUR_BODY_SIGMA="${sigma}" \
      FOUR_BODY_BOND_SCALE="${bond_scale}" \
      sbatch --parsable --export=ALL,FOUR_BODY_CAPACITY_ARRAY,FOUR_BODY_OUTPUT_DIR,FOUR_BODY_CENTER_COUNT,FOUR_BODY_TORSION_ORDER,FOUR_BODY_SIGMA,FOUR_BODY_BOND_SCALE \
        scripts/slurm_qm9_complete_total_geometry_four_body_capacity.sbatch)
    printf '%s,%s,%s,%s,%s,%s\n' \
      "${job_id}" "${name}" "${center_count}" "${torsion_order}" "${sigma}" "${bond_scale}"
  done
} | tee "${OUTPUT_ROOT}/submitted_jobs.csv"
