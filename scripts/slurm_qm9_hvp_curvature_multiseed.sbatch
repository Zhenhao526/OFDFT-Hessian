#!/usr/bin/env bash
#SBATCH --job-name=hvp_curve_3seed
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-14%15
#SBATCH --output=/scratch/xzh/logs/hvp_curvature/multiseed_%A_%a.out
#SBATCH --error=/scratch/xzh/logs/hvp_curvature/multiseed_%A_%a.err

set -euo pipefail
cd /scratch/xzh/code/structures25
source /scratch/xzh/env.sh
export DFT_DATA=/scratch/xzh/data DFT_MODELS=/scratch/xzh/models
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

TASKS="${HVP_CURVATURE_TASKS:?Set HVP_CURVATURE_TASKS}"
line=$((SLURM_ARRAY_TASK_ID + 2))
IFS=$'\t' read -r task variant weight floor seed max_steps run_name selection_status < <(
  awk -F'\t' -v line="${line}" 'NR==line' "${TASKS}"
)
run_name="${run_name%$'\r'}"
RUN_DIR="${DFT_MODELS}/train/runs/${run_name}"
LAST_CKPT="${RUN_DIR}/checkpoints/last.ckpt"
THROUGHPUT_CSV="${RUN_DIR}/throughput/throughput.csv"
FINAL_STEP=""
if [[ -f "${THROUGHPUT_CSV}" ]]; then
  FINAL_STEP="$(awk -F',' '
    NR == 1 { for (i = 1; i <= NF; i++) if ($i == "global_step") column = i; next }
    column && NF >= column { value = $column }
    END { gsub(/\r/, "", value); print value }
  ' "${THROUGHPUT_CSV}")"
fi
if [[ -f "${LAST_CKPT}" && "${FINAL_STEP}" == "${max_steps}" ]]; then
  echo "Reusing complete ${max_steps}-step checkpoint for ${run_name}"
  exit 0
fi
if [[ -d "${RUN_DIR}" ]]; then
  echo "Existing run is incomplete (final_step=${FINAL_STEP:-missing}); rerunning ${run_name} from the frozen base checkpoint"
fi
export HVP_REFERENCE_FLOOR="${floor}"
export MAX_STEPS="${max_steps}"
exec scripts/launch_qm9_hvp_stable_replay_train.sh \
  "${variant}" "${weight}" "${seed}" "${run_name}"
