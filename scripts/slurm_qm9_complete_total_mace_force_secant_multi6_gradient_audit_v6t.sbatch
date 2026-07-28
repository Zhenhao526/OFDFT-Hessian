#!/usr/bin/env bash
#SBATCH --job-name=ct_mace_m6_ga
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
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

checkpoint=/scratch/xzh/models/complete_total_capacity/20260717/stage5_mace_force_secant_h64_v6n/one_parent_pilot/0028399_s500/best.ckpt
printf '%s  %s\n' 982228588241a46321cb304fc4129b28a0c121cea2af401e9895f10f4cecc3d1 "${checkpoint}" | sha256sum --check --status
output=/scratch/xzh/models/complete_total_capacity/20260717/stage5_mace_force_secant_multi6_v6t/gradient_audit/0028399_groups8
mkdir -p "${output}" /scratch/xzh/logs/complete_total_capacity

/usr/bin/time -v -o "${output}/resource.time" \
  python scripts/qm9_complete_total_mace_force_secant_gradient_audit.py \
    --protocol configs/audit/qm9_complete_total_hessian_mace_force_secant_h64_pilot_v6n.yaml \
    --arm-id E12_mace_h64_l3_corr3_int3_force_secant \
    --checkpoint "${checkpoint}" \
    --parent-id 0028399 \
    --direction-groups '0,7,14,21,28,35;1,8,15,22,29,36;2,9,16,23,30,37;3,10,17,24,31,38;4,11,18,25,32,39;5,12,19,26,33,40;6,13,20,27,34,41;42,43,44,0,15,30' \
    --output-dir "${output}" \
    --device cuda:0
