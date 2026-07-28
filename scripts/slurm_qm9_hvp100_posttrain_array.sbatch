#!/usr/bin/env bash
#SBATCH --job-name=hvp100_eval
#SBATCH --partition=compute
#SBATCH --nodelist=node01,node04,node05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-23%24
#SBATCH --output=/scratch/xzh/logs/hvp100/%A_%a_eval.out
#SBATCH --error=/scratch/xzh/logs/hvp100/%A_%a_eval.err

set -euo pipefail
cd /scratch/xzh/code/structures25
source /scratch/xzh/env.sh
export DFT_DATA=/scratch/xzh/data DFT_MODELS=/scratch/xzh/models
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

SEEDS=(676368232 20260716 314159)
NAMES=(force force_secant force_hvp_w1e5 force_hvp_w1e4 force_hvp_w1e3 force_secant_hvp_w1e5 force_secant_hvp_w1e4 force_secant_hvp_w1e3)
seed_index=$((SLURM_ARRAY_TASK_ID / 8))
variant_index=$((SLURM_ARRAY_TASK_ID % 8))
seed="${SEEDS[seed_index]}"
name="${NAMES[variant_index]}"
run_name="qm9_hvp100_${name}_seed${seed}_s600_20260716"
run_dir="${DFT_MODELS}/train/runs/${run_name}"
ckpt="${run_dir}/checkpoints/last.ckpt"
out="${DFT_MODELS}/hvp100/20260716/posttrain/${run_name}"
mkdir -p "${out}"

python scripts/qm9_force_eval.py \
  --run-dir "${run_dir}" \
  --ckpt "${ckpt}" \
  --output-json "${out}/validation_energy_force.json" \
  --worst-csv "${out}/validation_force_worst.csv" \
  --batch-size 4 \
  --num-workers 0 \
  --device cuda:0 \
  --split val \
  --ground-state-only

python scripts/qm9_hvp_sidecar_eval.py \
  --run-name "${run_name}" \
  --run-dir "${run_dir}" \
  --checkpoint "${ckpt}" \
  --split-file "${DFT_MODELS}/hvp100/20260716/selection/split.pkl" \
  --data-dir "${DFT_DATA}" \
  --sidecar-dir "${DFT_MODELS}/hvp100/20260716/hvp_sidecars/sidecars" \
  --output-dir "${out}/fixed_hvp" \
  --device cuda:0
