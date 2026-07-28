#!/usr/bin/env bash
#SBATCH --job-name=qm9_pairtrain
#SBATCH --partition=compute
#SBATCH --nodelist=node06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:a100:8
#SBATCH --mem=800G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/xzh/logs/qm9_random1000_paired_candidate/%j.out
#SBATCH --error=/scratch/xzh/logs/qm9_random1000_paired_candidate/%j.err

set -euo pipefail

mkdir -p /scratch/xzh/logs/qm9_random1000_paired_candidate
cd /scratch/xzh/code/structures25

exec bash scripts/launch_qm9_random1000_paired_candidate_8xa100.sh
