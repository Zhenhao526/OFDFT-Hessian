#!/usr/bin/env bash
#SBATCH --job-name=ct_q_repl
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

output=/scratch/xzh/models/complete_total_capacity/20260717/stage6_relaxed_q_train100_v1/replacement_pool_v1
mkdir -p "$output" /scratch/xzh/logs/complete_total_capacity
/usr/bin/time -v -o "$output/resource.time" \
  python scripts/prepare_qm9_complete_total_relaxed_q_replacement_pool.py \
    --protocol configs/audit/qm9_complete_total_relaxed_q_replacement_pool_v1.yaml \
    --output-dir "$output"
