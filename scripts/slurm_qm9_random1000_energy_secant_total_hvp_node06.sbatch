#!/usr/bin/env bash
#SBATCH --job-name=qm9_esec_hvp
#SBATCH --partition=compute
#SBATCH --nodelist=node06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=700G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/xzh/logs/qm9_random1000_energy_secant_hvp/%j.out
#SBATCH --error=/scratch/xzh/logs/qm9_random1000_energy_secant_hvp/%j.err

set -euo pipefail
mkdir -p /scratch/xzh/logs/qm9_random1000_energy_secant_hvp
cd /scratch/xzh/code/structures25
exec bash scripts/launch_qm9_random1000_energy_secant_total_hvp_confirm.sh
