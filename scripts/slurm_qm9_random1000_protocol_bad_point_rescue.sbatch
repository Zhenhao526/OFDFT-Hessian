#!/usr/bin/env bash
#SBATCH --job-name=qm9_hrescue
#SBATCH --partition=compute
#SBATCH --nodelist=node06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/xzh/logs/qm9_random1000_hessian_protocol_bad_point_rescue/%j.out
#SBATCH --error=/scratch/xzh/logs/qm9_random1000_hessian_protocol_bad_point_rescue/%j.err

set -euo pipefail
mkdir -p /scratch/xzh/logs/qm9_random1000_hessian_protocol_bad_point_rescue
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
exec bash scripts/launch_qm9_random1000_protocol_bad_point_rescue.sh
