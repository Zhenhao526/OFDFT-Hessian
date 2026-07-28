#!/usr/bin/env bash
#SBATCH --job-name=ct_q59_audit
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

root=/scratch/xzh/models/complete_total_capacity/20260717/stage6_relaxed_q_train100_v1/stage2p5_local_scalar_v1
mkdir -p "${root}/feature_jet_audit" /scratch/xzh/logs/complete_total_capacity
/usr/bin/time -v -o "${root}/feature_jet_audit/resource.time" \
  python scripts/qm9_complete_total_relaxed_q_feature_jet_audit.py \
    --protocol configs/audit/qm9_complete_total_relaxed_q_stage2p5_local_scalar_v1.yaml \
    --jet-dir "${root}/feature_jets" \
    --output-dir "${root}/feature_jet_audit"
