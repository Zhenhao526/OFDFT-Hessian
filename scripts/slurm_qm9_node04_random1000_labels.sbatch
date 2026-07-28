#!/usr/bin/env bash
#SBATCH --job-name=qm9_r1000_lbl
#SBATCH --partition=compute
#SBATCH --nodelist=node04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=120
#SBATCH --mem=95G
#SBATCH --time=2-00:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/xzh/logs/qm9_random1000_labels/%j.out
#SBATCH --error=/scratch/xzh/logs/qm9_random1000_labels/%j.err

set -euo pipefail

mkdir -p /scratch/xzh/logs/qm9_random1000_labels /scratch/xzh/tmp

export CODE_DIR="${CODE_DIR:-/scratch/xzh/code/structures25}"
export ENV_FILE="${ENV_FILE:-/scratch/xzh/env.sh}"
export DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
export DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"
export DATASET_NAME="${DATASET_NAME:-QM9PBEForceRandom1000}"
export DATASET_FILENAME="${DATASET_FILENAME:-qm9_pbe_force_random1000}"
export N_MOLECULES="${N_MOLECULES:-1000}"
export RANDOM_SEED="${RANDOM_SEED:-20260709}"
export LABEL_NUM_PROCESSES="${LABEL_NUM_PROCESSES:-20}"
export LABEL_NUM_THREADS="${LABEL_NUM_THREADS:-1}"
export MAX_MEMORY_PER_PROCESS="${MAX_MEMORY_PER_PROCESS:-4000}"

cd "${CODE_DIR}"

bash scripts/launch_qm9_node04_random1000_labels.sh
