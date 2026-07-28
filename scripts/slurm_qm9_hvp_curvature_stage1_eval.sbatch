#!/usr/bin/env bash
#SBATCH --job-name=hvp_curve_eval
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-19%20
#SBATCH --output=/scratch/xzh/logs/hvp_curvature/stage1_%A_%a.out
#SBATCH --error=/scratch/xzh/logs/hvp_curvature/stage1_%A_%a.err

set -euo pipefail
cd /scratch/xzh/code/structures25
source /scratch/xzh/env.sh
export DFT_DATA=/scratch/xzh/data DFT_MODELS=/scratch/xzh/models
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

TASKS="${HVP_CURVATURE_TASKS:?Set HVP_CURVATURE_TASKS}"
line=$((SLURM_ARRAY_TASK_ID + 2))
IFS=$'\t' read -r task variant weight floor seed max_steps run_name screen_selection_status < <(
  awk -F'\t' -v line="${line}" 'NR==line' "${TASKS}"
)
run_name="${run_name%$'\r'}"
screen_selection_status="${screen_selection_status%$'\r'}"
RUN_DIR="${DFT_MODELS}/train/runs/${run_name}"
CKPT="${RUN_DIR}/checkpoints/last.ckpt"
if [[ ! -f "${CKPT}" ]]; then
  CKPT="$(find "${RUN_DIR}/checkpoints" -maxdepth 1 -name '*.ckpt' | sort | tail -n 1)"
fi
[[ -f "${CKPT}" ]] || { echo "No checkpoint for ${run_name}" >&2; exit 1; }

STAGE1_ROOT="${HVP_STAGE1_ROOT:-/scratch/xzh/models/hvp_curvature_v1/20260716/stage1}"
OUT="${STAGE1_ROOT}/${run_name}"
mkdir -p "${OUT}"
python scripts/qm9_force_eval.py \
  --run-dir "${RUN_DIR}" --ckpt "${CKPT}" \
  --output-json "${OUT}/validation_energy_force.json" \
  --worst-csv "${OUT}/validation_force_worst.csv" \
  --split val --ground-state-only --batch-size 4 --num-workers 0 \
  --device cuda:0 --top-k 20

python scripts/qm9_force_eval.py \
  --run-dir "${RUN_DIR}" --ckpt "${CKPT}" \
  --output-json "${OUT}/train700_forgetting.json" \
  --split train --parent-ids /scratch/xzh/models/hvp_curvature_v1/20260716/train700_parent_ids.txt \
  --base-geometry-only --ground-state-only --batch-size 4 --num-workers 0 \
  --device cuda:0 --top-k 20

python scripts/qm9_hvp_sidecar_eval.py \
  --run-name "${run_name}" --run-dir "${RUN_DIR}" --checkpoint "${CKPT}" \
  --split-file /scratch/xzh/data/QM9PBEForceRandom1000PairedAug/split.pkl \
  --data-dir /scratch/xzh/data \
  --sidecar-dir /scratch/xzh/models/hvp_branch_stability/20260716/formal_val20_v3/analysis_v3/stable_sidecars \
  --output-dir "${OUT}/fixed_hvp" --device cuda:0
