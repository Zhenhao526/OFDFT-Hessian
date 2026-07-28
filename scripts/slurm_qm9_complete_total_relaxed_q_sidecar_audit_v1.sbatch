#!/usr/bin/env bash
#SBATCH --job-name=ct_relaxed_q
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

output=/scratch/xzh/models/complete_total_capacity/20260717/stage6_relaxed_q_train100_v1/eligibility_audit
mkdir -p "${output}" /scratch/xzh/logs/complete_total_capacity

/usr/bin/time -v -o "${output}/resource.time" \
  python scripts/qm9_complete_total_relaxed_q_sidecar_audit.py \
    --protocol configs/audit/qm9_complete_total_relaxed_q_train100_v1.yaml \
    --output-dir "${output}"
