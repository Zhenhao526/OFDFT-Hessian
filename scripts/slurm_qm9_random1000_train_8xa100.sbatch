#!/usr/bin/env bash
#SBATCH --job-name=qm9_r1000_trn
#SBATCH --partition=compute
#SBATCH --nodelist=node05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --mem=900G
#SBATCH --gres=gpu:a100:8
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/xzh/logs/qm9_random1000_train/%j.out
#SBATCH --error=/scratch/xzh/logs/qm9_random1000_train/%j.err

set -euo pipefail

mkdir -p /scratch/xzh/logs/qm9_random1000_train /scratch/xzh/tmp

export CODE_DIR="${CODE_DIR:-/scratch/xzh/code/structures25}"
export ENV_FILE="${ENV_FILE:-/scratch/xzh/env.sh}"
export DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
export DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"
export DATASET_NAME="${DATASET_NAME:-QM9PBEForceRandom1000}"
export EXPECTED_LABELS="${EXPECTED_LABELS:-4000}"
export EPOCHS="${EPOCHS:-10}"

# The shared env file currently defaults to four visible GPUs on direct SSH.
# This job owns the full node, so make all eight A100s visible explicitly.
export TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export SPLIT_PROCESSES="${SPLIT_PROCESSES:-64}"
export TRANSFORM_NUM_PROCESSES="${TRANSFORM_NUM_PROCESSES:-64}"
export TRANSFORM_NUM_THREADS="${TRANSFORM_NUM_THREADS:-1}"
export STAT_BATCH_SIZE="${STAT_BATCH_SIZE:-16}"
export STAT_NUM_WORKERS="${STAT_NUM_WORKERS:-4}"
export EG_BATCH_SIZE="${EG_BATCH_SIZE:-8}"
export EGF_BATCH_SIZE="${EGF_BATCH_SIZE:-4}"
export TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-8}"
export ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"

cd "${CODE_DIR}"

bash scripts/launch_qm9_random1000_train_8xa100.sh
