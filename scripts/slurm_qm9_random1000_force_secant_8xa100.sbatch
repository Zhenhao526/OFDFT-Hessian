#!/usr/bin/env bash
#SBATCH --job-name=qm9_fsec_w001
#SBATCH --partition=compute
#SBATCH --nodelist=node04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:a100:8
#SBATCH --mem=900G
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/xzh/logs/qm9_random1000_force_secant/%j.out
#SBATCH --error=/scratch/xzh/logs/qm9_random1000_force_secant/%j.err

set -euo pipefail

mkdir -p /scratch/xzh/logs/qm9_random1000_force_secant
cd /scratch/xzh/code/structures25

exec bash scripts/launch_qm9_random1000_force_secant_8xa100.sh
