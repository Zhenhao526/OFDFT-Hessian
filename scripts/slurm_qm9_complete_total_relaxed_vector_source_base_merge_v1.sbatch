#!/usr/bin/env bash
#SBATCH --job-name=ct_v20_bmerge
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

root=/scratch/xzh/models/complete_total_capacity/20260717/stage6_relaxed_vector_train20_v1
run_dir=/scratch/xzh/models/train/runs/qm9_random1000_egf_forcew1p0_e10_20260714_202203
run="vector_source=${run_dir}=${run_dir}/checkpoints/epoch_009.ckpt"
mkdir -p "${root}/source_baseline_merged" /scratch/xzh/logs/complete_total_capacity
/usr/bin/time -v -o "${root}/source_baseline_merged/resource.time" \
  python scripts/qm9_complete_total_replay_baseline_merge.py \
    --protocol configs/audit/qm9_complete_total_relaxed_vector_stage2p5_targets_v1.yaml \
    --task-csv "${root}/selection/source_base_tasks.csv" \
    --baseline-root "${root}/source_baselines" \
    --output-dir "${root}/source_baseline_merged" \
    --run "${run}" \
    --require-complete
