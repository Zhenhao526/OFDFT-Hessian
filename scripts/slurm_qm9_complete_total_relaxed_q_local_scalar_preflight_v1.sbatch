#!/usr/bin/env bash
#SBATCH --job-name=ct_q59_pre
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

root=/scratch/xzh/models/complete_total_capacity/20260717/stage6_relaxed_q_train100_v1/stage2p5_local_scalar_v1
mkdir -p "${root}/preflight" /scratch/xzh/logs/complete_total_capacity
/usr/bin/time -v -o "${root}/preflight/resource.time" \
  python scripts/qm9_complete_total_relaxed_q_local_scalar.py preflight \
    --protocol configs/audit/qm9_complete_total_relaxed_q_stage2p5_local_scalar_v1.yaml \
    --output-dir "${root}/preflight"
