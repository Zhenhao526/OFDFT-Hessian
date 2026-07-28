#!/usr/bin/env bash
#SBATCH --job-name=ct_mace_hgeo
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xzh/logs/complete_total_capacity/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/complete_total_capacity/%x_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
export PYTHONPATH=/scratch/xzh/vendor/mace_torch_0_3_16
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run=/scratch/xzh/models/complete_total_capacity/20260717/stage5_mace_force_secant_pcgrad_linesearch_h64_v6r/one_parent/0028399_s1000
checkpoint="${run}/step_0000100.ckpt"
printf '%s  %s\n' c9d0f6789089c7e369c6ca5792b5a05629ec04dfdd611bf017ee0aa57e9bb395 "${checkpoint}" | sha256sum --check --status

python scripts/qm9_complete_total_mace_checkpoint_hessian_geometry.py \
  --protocol configs/audit/qm9_complete_total_hessian_mace_force_secant_h64_pcgrad_linesearch_v6r.yaml \
  --arm-id E15_mace_h64_l3_corr3_int3_force_secant_fixed_scale_pcgrad_linesearch \
  --checkpoint "${checkpoint}" \
  --parent-id 0028399 \
  --output "${run}/step_0000100_hessian_geometry.json" \
  --device cuda:0
