#!/usr/bin/env bash
#SBATCH --job-name=qm9_s3rmerge
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/stage3_replay_merge_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/stage3_replay_merge_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

ROOT=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1
ASSETS=${ROOT}/assets_v3_uniform_A
BASELINE_ROOT=${ROOT}/replay_baseline_v3_uniform_A
OUTPUT=${ROOT}/replay_baseline_v3_uniform_A_merged
SOURCE_RUN=qm9_hvp_curvature_v1_A_w0ep00_f1em02_seed314159_s1200
RUN_DIR=/scratch/xzh/models/train/runs/${SOURCE_RUN}
RUN="stage3_base=${RUN_DIR}=${RUN_DIR}/checkpoints/last.ckpt"

python scripts/qm9_complete_total_replay_baseline_merge.py \
  --protocol configs/audit/qm9_complete_total_hessian_unseen_parent_v1.yaml \
  --task-csv "${ASSETS}/train800_replay_labels.csv" \
  --baseline-root "${BASELINE_ROOT}" \
  --output-dir "${OUTPUT}" \
  --run "${RUN}" \
  --no-require-complete
