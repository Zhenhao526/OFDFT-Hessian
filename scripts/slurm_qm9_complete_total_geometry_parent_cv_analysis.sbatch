#!/usr/bin/env bash
#SBATCH --job-name=ct_geo_cv_merge
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail

source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

root=/scratch/xzh/models/complete_total_capacity/20260717
stage3=${root}/stage3_unseen_parent_v1
base=${GEOMETRY_PARENT_CV_BASE:-${stage3}/geometry_parent_cv_v1}
analysis=${GEOMETRY_PARENT_CV_ANALYSIS:-${base}/analysis_v1}
protocol=configs/audit/qm9_complete_total_hessian_geometry_parent_cv_v1.yaml
baseline=${root}/stage2_direction_v1/selection_uniform_A_v1/selected_baseline_manifest.json
dataset=/scratch/xzh/data/QM9PBEForceRandom1000

variants=(
  --variant "G0_h128_fullh_no_replay=${base}/G0_h128_fullh_no_replay"
)
if [[ "${GEOMETRY_PARENT_CV_INCLUDE_G1:-0}" == "1" ]]; then
  variants+=(
    --variant "G1_h128_fullh_replay=${base}/G1_h128_fullh_replay"
  )
fi

mkdir -p "${analysis}"
python scripts/qm9_complete_total_geometry_parent_cv_analysis.py \
  --protocol "${protocol}" \
  --baseline-manifest "${baseline}" \
  "${variants[@]}" \
  --output-dir "${analysis}"

python scripts/qm9_hessian_vibrational_metrics.py \
  --manifest-json "${baseline}" \
  --dataset-dir "${dataset}" \
  --result-json "geometry_parent_cv=${analysis}/held_result_rows.json" \
  --baseline-result "original_A=${baseline}" \
  --output-dir "${analysis}/vibrational_metrics"

python scripts/qm9_complete_total_geometry_parent_cv_analysis.py \
  --protocol "${protocol}" \
  --baseline-manifest "${baseline}" \
  "${variants[@]}" \
  --vibrational-summary "${analysis}/vibrational_metrics/summary.json" \
  --output-dir "${analysis}"
