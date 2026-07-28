#!/usr/bin/env bash
#SBATCH --job-name=gf_cth_verify
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/xzh/logs/graphformer_relaxed_hvp/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/graphformer_relaxed_hvp/%x_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

mode=${VERIFY_MODE:-derivative}
molecule=${MOLECULE_ID:-0028399}
direction_index=${DIRECTION_INDEX:-0}
implicit_diagonal_probes=${IMPLICIT_DIAGONAL_PROBES:-0}
implicit_max_iterations=${IMPLICIT_MAX_ITERATIONS:-2500}
root=/scratch/xzh/models/graphformer_relaxed_hvp/20260723/v1
output="${root}/verification/${mode}_${molecule}_d${direction_index}_p${implicit_diagonal_probes}_${SLURM_JOB_ID}"
mkdir -p "${output}" /scratch/xzh/logs/graphformer_relaxed_hvp

/usr/bin/time -v -o "${output}/resource.time" \
  python scripts/qm9_graphformer_complete_total_relaxed_hvp_verify.py \
    --protocol configs/audit/qm9_graphformer_complete_total_relaxed_hvp_v1.yaml \
    --direction-manifest "${root}/directions/stable5/manifest.json" \
    --output-dir "${output}" \
    --molecule "${molecule}" \
    --direction-index "${direction_index}" \
    --mode "${mode}" \
    --implicit-diagonal-probes "${implicit_diagonal_probes}" \
    --implicit-max-iterations "${implicit_max_iterations}" \
    --device cuda:0
