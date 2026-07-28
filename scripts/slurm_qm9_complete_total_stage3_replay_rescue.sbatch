#!/usr/bin/env bash
#SBATCH --job-name=qm9_s3rescue
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/xzh/logs/qm9_s3rescue_%j.out
#SBATCH --error=/scratch/xzh/logs/qm9_s3rescue_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}

ROOT=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1
RUN_DIR=/scratch/xzh/models/train/runs/qm9_hvp_curvature_v1_A_w0ep00_f1em02_seed314159_s1200

python scripts/qm9_complete_total_replay_transient_rescue.py \
  --protocol configs/audit/qm9_complete_total_hessian_unseen_parent_v1.yaml \
  --task-csv "${ROOT}/assets_v2/train800_replay_labels.csv" \
  --output-dir "${ROOT}/replay_baseline_v3_uniform_A" \
  --run "stage3_base=${RUN_DIR}=${RUN_DIR}/checkpoints/last.ckpt" \
  --manifest "${ROOT}/replay_baseline_v3_uniform_A_merged/transient_rescue_manifest.json" \
  --device cuda:0 \
  --integral-derivative-workers 4 \
  --max-tasks 100
