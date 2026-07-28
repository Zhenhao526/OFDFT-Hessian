#!/usr/bin/env bash
#SBATCH --job-name=qm9_esec_eval
#SBATCH --partition=compute
#SBATCH --nodelist=node05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:a100:8
#SBATCH --mem=700G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/xzh/logs/qm9_random1000_energy_secant_eval/%j.out
#SBATCH --error=/scratch/xzh/logs/qm9_random1000_energy_secant_eval/%j.err

set -euo pipefail
mkdir -p /scratch/xzh/logs/qm9_random1000_energy_secant_eval
cd /scratch/xzh/code/structures25
exec bash scripts/launch_qm9_random1000_energy_secant_posttrain_eval.sh
