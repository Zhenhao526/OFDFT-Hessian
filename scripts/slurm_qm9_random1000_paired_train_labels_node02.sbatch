#!/usr/bin/env bash
#SBATCH --job-name=qm9_pairlabels
#SBATCH --partition=compute
#SBATCH --nodelist=node02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=120
#SBATCH --mem=700G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/xzh/logs/qm9_random1000_paired_train_labels/%j.out
#SBATCH --error=/scratch/xzh/logs/qm9_random1000_paired_train_labels/%j.err

set -euo pipefail
mkdir -p /scratch/xzh/logs/qm9_random1000_paired_train_labels
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export LABEL_NUM_PROCESSES="${LABEL_NUM_PROCESSES:-60}"
export LABEL_NUM_THREADS=1
export MAX_MEMORY_PER_PROCESS=6000
exec bash scripts/launch_qm9_random1000_paired_train_labels.sh
