#!/usr/bin/env bash
#SBATCH --job-name=qm9_r1000_hgpu
#SBATCH --partition=compute
#SBATCH --nodelist=node04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=900G
#SBATCH --gres=gpu:a100:8
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/xzh/logs/qm9_random1000_test100_hessian_gpu4pyscf/%j.out
#SBATCH --error=/scratch/xzh/logs/qm9_random1000_test100_hessian_gpu4pyscf/%j.err

set -euo pipefail

mkdir -p /scratch/xzh/logs/qm9_random1000_test100_hessian_gpu4pyscf

export CODE_DIR="${CODE_DIR:-/scratch/xzh/code/structures25}"
export ENV_FILE="${ENV_FILE:-/scratch/xzh/env.sh}"
export DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
export DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"
export DATASET_NAME="${DATASET_NAME:-QM9PBEForceRandom1000}"

export EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HESSIAN_REF_MOLECULES="${HESSIAN_REF_MOLECULES:-100}"
export GPU_REF_SHARDS="${GPU_REF_SHARDS:-8}"
export HESSIAN_MODEL_SHARDS="${HESSIAN_MODEL_SHARDS:-8}"
export HESSIAN_DISPLACEMENT="${HESSIAN_DISPLACEMENT:-1e-3}"
export HESSIAN_SCF_ITERATION="${HESSIAN_SCF_ITERATION:-1}"

cd "${CODE_DIR}"

bash scripts/launch_qm9_random1000_test100_hessian_eval_node04_gpu4pyscf.sh
