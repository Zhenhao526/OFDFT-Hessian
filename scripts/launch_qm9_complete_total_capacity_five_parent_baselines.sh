#!/usr/bin/env bash
set -euo pipefail

cd /scratch/xzh/code/structures25

ROOT=/scratch/xzh/models/complete_total_capacity/20260717
SOURCE_CHECKPOINT=${CAPACITY_FIVE_SOURCE_CHECKPOINT:-${ROOT}/one_parent_0028399_s50_lr1e-6_h1_q0_job3175/checkpoints/step_0000200.ckpt}
OUTPUT=${ROOT}/five_parent_baseline_jobs.csv

parents=(
  "0050129 45"
  "0031108 48"
  "0132419 48"
  "0031012 54"
)

{
  printf 'job_id,molecule_id,cartesian_direction_count,source_checkpoint\n'
  for parent in "${parents[@]}"; do
    read -r molecule direction_count <<<"${parent}"
    job_id=$(CAPACITY_CKPT="${SOURCE_CHECKPOINT}" \
      CAPACITY_MOLECULE="${molecule}" \
      CAPACITY_MAX_STEPS=0 \
      CAPACITY_LR=0 \
      CAPACITY_LAMBDA_E=0 \
      CAPACITY_LAMBDA_F=0 \
      CAPACITY_LAMBDA_RHO=0 \
      CAPACITY_LAMBDA_H=0 \
      CAPACITY_LAMBDA_Q=0 \
      CAPACITY_LAMBDA_SPEC=0 \
      CAPACITY_DIRECTION_LIMIT="${direction_count}" \
      CAPACITY_DIRECTIONS_PER_STEP=1 \
      CAPACITY_EVAL_INTERVAL=1 \
      CAPACITY_DENSITY_REFRESH_INTERVAL=100 \
      CAPACITY_GRADIENT_DIAGNOSTICS_INTERVAL=0 \
      sbatch --parsable \
        --export=ALL,CAPACITY_CKPT,CAPACITY_MOLECULE,CAPACITY_MAX_STEPS,CAPACITY_LR,CAPACITY_LAMBDA_E,CAPACITY_LAMBDA_F,CAPACITY_LAMBDA_RHO,CAPACITY_LAMBDA_H,CAPACITY_LAMBDA_Q,CAPACITY_LAMBDA_SPEC,CAPACITY_DIRECTION_LIMIT,CAPACITY_DIRECTIONS_PER_STEP,CAPACITY_EVAL_INTERVAL,CAPACITY_DENSITY_REFRESH_INTERVAL,CAPACITY_GRADIENT_DIAGNOSTICS_INTERVAL \
        scripts/slurm_qm9_complete_total_capacity_one_parent.sbatch)
    printf '%s,%s,%s,%s\n' \
      "${job_id}" "${molecule}" "${direction_count}" "${SOURCE_CHECKPOINT}"
  done
} | tee "${OUTPUT}"
