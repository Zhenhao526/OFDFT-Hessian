#!/usr/bin/env bash
#SBATCH --job-name=qm9_r1000_eval
#SBATCH --partition=compute
#SBATCH --nodelist=node05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/xzh/logs/qm9_random1000_eval/%j.out
#SBATCH --error=/scratch/xzh/logs/qm9_random1000_eval/%j.err

set -euo pipefail

mkdir -p /scratch/xzh/logs/qm9_random1000_eval

export CODE_DIR="${CODE_DIR:-/scratch/xzh/code/structures25}"
export ENV_FILE="${ENV_FILE:-/scratch/xzh/env.sh}"
export DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
export DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"
export DATASET_NAME="${DATASET_NAME:-QM9PBEForceRandom1000}"

export EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"
export ENERGY_FORCE_BATCH_SIZE="${ENERGY_FORCE_BATCH_SIZE:-8}"
export ENERGY_FORCE_WORKERS="${ENERGY_FORCE_WORKERS:-4}"
export HESSIAN_REF_MOLECULES="${HESSIAN_REF_MOLECULES:-10}"
export HESSIAN_REF_WORKERS="${HESSIAN_REF_WORKERS:-8}"
export HESSIAN_DISPLACEMENT="${HESSIAN_DISPLACEMENT:-1e-3}"
export HESSIAN_SCF_ITERATION="${HESSIAN_SCF_ITERATION:-1}"

cd "${CODE_DIR}"

bash scripts/launch_qm9_random1000_eval.sh
