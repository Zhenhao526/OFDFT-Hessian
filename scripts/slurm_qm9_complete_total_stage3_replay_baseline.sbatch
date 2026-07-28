#!/usr/bin/env bash
#SBATCH --job-name=qm9_s3_replay
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/xzh/logs/%x_%A_%a.out
#SBATCH --error=/scratch/xzh/logs/%x_%A_%a.err

set -euo pipefail

source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

ROOT=${STAGE3_ROOT:-/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1}
TASK_CSV=${STAGE3_TASK_CSV:-${ROOT}/assets_v2/train800_replay_labels.csv}
OUTPUT_DIR=${STAGE3_REPLAY_OUTPUT:-${ROOT}/replay_baseline_v1}
PROTOCOL=${STAGE3_PROTOCOL:-configs/audit/qm9_complete_total_hessian_unseen_parent_v1.yaml}
SHARD_COUNT=${STAGE3_SHARD_COUNT:-8}
RUN_DIR=${STAGE3_BASE_RUN_DIR:-/scratch/xzh/models/train/runs/qm9_hvp_curvature_v1_A_w0ep00_f1em02_seed314159_s1200}
CKPT=${STAGE3_BASE_CHECKPOINT:-${RUN_DIR}/checkpoints/last.ckpt}

mkdir -p /scratch/xzh/logs "${OUTPUT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}

if [[ "${STAGE3_FALLBACK_ALWAYS:-true}" == "true" ]]; then
  FALLBACK_ARG=(--fallback-always)
else
  FALLBACK_ARG=(--no-fallback-always)
fi
TASK_ARGS=()
if [[ -n "${STAGE3_TASK_INDEX:-}" ]]; then
  TASK_ARGS+=(--task-index "${STAGE3_TASK_INDEX}")
else
  SHARD_INDEX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
  TASK_ARGS+=(--shard-index "${SHARD_INDEX}" --shard-count "${SHARD_COUNT}")
fi

/usr/bin/time -v python scripts/qm9_complete_total_replay_baseline.py \
  --protocol "${PROTOCOL}" \
  --task-csv "${TASK_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --run "stage3_base=${RUN_DIR}=${CKPT}" \
  "${TASK_ARGS[@]}" \
  --device cuda:0 \
  --transform-device cpu \
  "${FALLBACK_ARG[@]}" \
  --integral-derivative-workers "${STAGE3_INTEGRAL_DERIVATIVE_WORKERS:-4}"
