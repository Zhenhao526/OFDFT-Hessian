#!/usr/bin/env bash
#SBATCH --job-name=qm9_hphysics
#SBATCH --partition=compute
#SBATCH --nodelist=node04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:a100:8
#SBATCH --mem=900G
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/xzh/logs/qm9_random1000_hessian_physics_audits/%j.out
#SBATCH --error=/scratch/xzh/logs/qm9_random1000_hessian_physics_audits/%j.err

set -euo pipefail
mkdir -p /scratch/xzh/logs/qm9_random1000_hessian_physics_audits
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
exec bash scripts/launch_qm9_random1000_hessian_physics_audits_node04.sh
