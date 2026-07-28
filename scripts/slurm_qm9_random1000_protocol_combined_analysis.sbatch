#!/usr/bin/env bash
#SBATCH --job-name=qm9_hprotan
#SBATCH --partition=compute
#SBATCH --nodelist=node02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/xzh/logs/qm9_random1000_hessian_protocol_scan/%j_analysis.out
#SBATCH --error=/scratch/xzh/logs/qm9_random1000_hessian_protocol_scan/%j_analysis.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

protocol_root=/scratch/xzh/models/eval/qm9_random1000_hessian_protocol_scan/20260714_194632
anomaly_root="$(find /scratch/xzh/models/eval/qm9_random1000_anomaly_tol1e6_full_hessian \
  -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
baseline=/scratch/xzh/models/eval/qm9_random1000_test100_density_relaxed_hessian_node04/20260714_150546/summary.json

[[ -f "${anomaly_root}/combined_summary.json" ]] || {
  echo "ERROR: incomplete anomaly scan: ${anomaly_root}" >&2
  exit 1
}

python scripts/qm9_hessian_protocol_scan_analysis.py \
  --baseline-summary "${baseline}" \
  --scan-root "${protocol_root}" \
  --additional-scan-root "${anomaly_root}" \
  --selected-manifest "${protocol_root}/representative_manifest_8mol.json" \
  --output-dir "${protocol_root}/analysis_complete_with_anomaly1e6"
