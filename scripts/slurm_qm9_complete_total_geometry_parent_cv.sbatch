#!/usr/bin/env bash
#SBATCH --job-name=ct_geo_parent_cv
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --array=0-4%5
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%A_%a.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%A_%a.err

set -euo pipefail

source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

root=/scratch/xzh/models/complete_total_capacity/20260717
stage3=${root}/stage3_unseen_parent_v1
protocol=configs/audit/qm9_complete_total_hessian_geometry_parent_cv_v1.yaml
baseline=${root}/stage2_direction_v1/selection_uniform_A_v1/selected_baseline_manifest.json
design=${root}/stage2_direction_v1/shared_descriptor_design_v2
inventory=${stage3}/train800_feature_inventory_floor1_v3/train800_feature_inventory_manifest.json
replay=${stage3}/replay_descriptor_cache_train800_union_floor1_v2_merged/replay_descriptor_cache_manifest.json
variant=${GEOMETRY_PARENT_CV_VARIANT:-G0_h128_fullh_no_replay}
fold=${SLURM_ARRAY_TASK_ID}
seed=$((20260722 + fold))
replay_seed=$((20261722 + fold))
output=${GEOMETRY_PARENT_CV_OUTPUT_BASE:-${stage3}/geometry_parent_cv_v1}/${variant}/fold_${fold}

mkdir -p "${output}" /scratch/xzh/logs/complete_total_capacity

replay_args=()
lambda_replay_energy=0
lambda_replay_force=0
replay_batch_size=0
curvature_ratio=inf
case "${variant}" in
  G0_h128_fullh_no_replay)
    ;;
  G1_h128_fullh_replay)
    replay_args=(--replay-cache-manifest "${replay}")
    lambda_replay_energy=1
    lambda_replay_force=1
    replay_batch_size=4
    curvature_ratio=1
    ;;
  *)
    echo "Unknown preregistered variant: ${variant}" >&2
    exit 2
    ;;
esac

/usr/bin/time -v -o "${output}/resource.time" \
python scripts/qm9_complete_total_geometry_mlp_capacity.py \
  --baseline-manifest "${baseline}" \
  --design-root "${design}" \
  --feature-inventory-manifest "${inventory}" \
  --zero-linear-initialization \
  --parent-cv-protocol "${protocol}" \
  --parent-cv-variant-id "${variant}" \
  --parent-cv-fold-index "${fold}" \
  --output-dir "${output}" \
  --device cuda:0 \
  --hidden-size 128 \
  --seed "${seed}" \
  --replay-seed "${replay_seed}" \
  --steps 30000 \
  --learning-rate 3e-5 \
  --gradient-clip 10 \
  --log-interval 100 \
  --energy-scale 0.1 \
  --force-scale 0.05 \
  --lambda-energy 1 \
  --lambda-force 1 \
  --lambda-hessian 10 \
  --lambda-spectrum 0 \
  --lambda-parameter 1e-8 \
  --lambda-replay-energy "${lambda_replay_energy}" \
  --lambda-replay-force "${lambda_replay_force}" \
  --hvp-warmup-steps 1000 \
  --pcgrad \
  --pcgrad-max-curvature-to-replay-norm-ratio "${curvature_ratio}" \
  --column-norm-relative-cutoff 0 \
  --feature-scale-mode floored_column_norm \
  --feature-scale-floor 1 \
  --absolute-hessian-scale 0.1 \
  --relative-loss-fraction 0.5 \
  --hvp-tail-fraction 0.2 \
  --hvp-tail-weight 0 \
  --replay-batch-size "${replay_batch_size}" \
  --replay-eval-size 64 \
  --replay-cpu-cache-size 256 \
  "${replay_args[@]}"
