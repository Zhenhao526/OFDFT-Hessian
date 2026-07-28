#!/usr/bin/env bash
#SBATCH --job-name=qm9_hprotocol
#SBATCH --partition=compute
#SBATCH --nodelist=node04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:a100:8
#SBATCH --mem=900G
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/xzh/logs/qm9_random1000_hessian_protocol_scan/%j.out
#SBATCH --error=/scratch/xzh/logs/qm9_random1000_hessian_protocol_scan/%j.err

set -euo pipefail

mkdir -p /scratch/xzh/logs/qm9_random1000_hessian_protocol_scan
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

exec bash scripts/launch_qm9_random1000_hessian_protocol_scan_node04.sh
