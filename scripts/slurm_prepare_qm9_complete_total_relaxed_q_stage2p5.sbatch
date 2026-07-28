#!/bin/bash
#SBATCH --job-name=ct_q_split
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/xzh/logs/ct_q_split_%j.out
#SBATCH --error=/scratch/xzh/logs/ct_q_split_%j.err

set -euo pipefail

cd /scratch/xzh/code/structures25
source /scratch/xzh/env.sh

PROTOCOL=${PROTOCOL:-configs/audit/qm9_complete_total_relaxed_q_stage2p5_split_v1.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/xzh/models/complete_total_capacity/20260717/stage6_relaxed_q_train100_v1/stage2p5_direction_split_v1_retry1}

mkdir -p /scratch/xzh/logs "$OUTPUT_DIR"
/usr/bin/time -v -o "$OUTPUT_DIR/resource.time" \
  python scripts/prepare_qm9_complete_total_relaxed_q_stage2p5.py \
    --protocol "$PROTOCOL" \
    --output-dir "$OUTPUT_DIR"
