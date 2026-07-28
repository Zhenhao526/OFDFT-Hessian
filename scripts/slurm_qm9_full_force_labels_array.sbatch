#!/usr/bin/env bash
#SBATCH --job-name=qm9_full_labels
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch/xzh/logs/qm9_full_labels/%A_%a.out
#SBATCH --error=/scratch/xzh/logs/qm9_full_labels/%A_%a.err

set -euo pipefail

: "${CHUNK_SIZE:=200}"
: "${TOTAL_MOLECULES:=133885}"
: "${NUM_PROCESSES:=20}"
: "${DFT_DATA:=/scratch/xzh/data}"
: "${DFT_MODELS:=/scratch/xzh/models}"
: "${CODE_DIR:=/scratch/xzh/code/structures25}"
: "${ENV_FILE:=/scratch/xzh/env.sh}"

source "${ENV_FILE}"

mkdir -p /scratch/xzh/logs/qm9_full_labels /scratch/xzh/tmp

START_IDX=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
if (( START_IDX >= TOTAL_MOLECULES )); then
  echo "Array task ${SLURM_ARRAY_TASK_ID}: start_idx=${START_IDX} >= TOTAL_MOLECULES=${TOTAL_MOLECULES}; exiting."
  exit 0
fi

N_MOLECULES="${CHUNK_SIZE}"
if (( START_IDX + N_MOLECULES > TOTAL_MOLECULES )); then
  N_MOLECULES=$((TOTAL_MOLECULES - START_IDX))
fi

export DFT_DATA DFT_MODELS
export RAW_DATA_DIR="${DFT_DATA}/QM9/raw"
export TMPDIR="/scratch/xzh/tmp/qm9_full_labels_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export NUM_PROCESSES
export NUM_THREADS_PER_PROCESS=1
export MAX_MEMORY_PER_PROCESS=5000

cd "${CODE_DIR}"

bash scripts/launch_qm9_full_force_labels_shard.sh "${START_IDX}" "${N_MOLECULES}" "${NUM_PROCESSES}"
