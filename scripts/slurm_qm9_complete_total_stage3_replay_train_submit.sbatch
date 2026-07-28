#!/usr/bin/env bash
#SBATCH --job-name=ctH_s3submit
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/stage3_train_submit_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/stage3_train_submit_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

PROFILE=${STAGE3_TRAIN_PROFILE:-smoke}
SEED=${STAGE3_TRAIN_SEED:-20260720}
bash scripts/launch_qm9_complete_total_stage3_replay_train.sh "${PROFILE}" "${SEED}"
