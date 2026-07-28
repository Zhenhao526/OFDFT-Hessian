#!/usr/bin/env bash
#SBATCH --job-name=qm9_test100f
#SBATCH --partition=compute
#SBATCH --nodelist=node05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:a100:8
#SBATCH --mem=900G
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/xzh/logs/qm9_random1000_frozen_test100/%j.out
#SBATCH --error=/scratch/xzh/logs/qm9_random1000_frozen_test100/%j.err

set -euo pipefail
mkdir -p /scratch/xzh/logs/qm9_random1000_frozen_test100
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
exec bash scripts/launch_qm9_random1000_frozen_test100.sh
