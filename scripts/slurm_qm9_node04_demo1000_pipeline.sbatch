#!/usr/bin/env bash
#SBATCH --job-name=qm9_demo1000
#SBATCH --partition=compute
#SBATCH --nodelist=node04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=120
#SBATCH --mem=95G
#SBATCH --time=2-00:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/xzh/logs/qm9_demo1000/%j.out
#SBATCH --error=/scratch/xzh/logs/qm9_demo1000/%j.err

set -euo pipefail

mkdir -p /scratch/xzh/logs/qm9_demo1000 /scratch/xzh/tmp

export CODE_DIR="${CODE_DIR:-/scratch/xzh/code/structures25}"
export ENV_FILE="${ENV_FILE:-/scratch/xzh/env.sh}"
export DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
export DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"

cd "${CODE_DIR}"

bash scripts/launch_qm9_node04_demo1000_pipeline.sh
