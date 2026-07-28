#!/usr/bin/env bash
#SBATCH --job-name=qm9_pairprep
#SBATCH --partition=compute
#SBATCH --nodelist=node02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=120
#SBATCH --mem=700G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/xzh/logs/qm9_random1000_paired_augmentation_prepare/%j.out
#SBATCH --error=/scratch/xzh/logs/qm9_random1000_paired_augmentation_prepare/%j.err

set -euo pipefail
mkdir -p /scratch/xzh/logs/qm9_random1000_paired_augmentation_prepare
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
exec bash scripts/launch_qm9_random1000_paired_augmentation_prepare.sh
