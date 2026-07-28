#!/usr/bin/env bash
#SBATCH --job-name=ct_mace_pwr_pcg
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
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

checkpoint=/scratch/xzh/models/complete_total_capacity/20260717/stage5_mace_power_readout_smoke_v6g/smoke/E11b_mace_h16_power_rf2048_in1000/best.ckpt
expected_checkpoint_sha256=9c773b8d2955542a71c5834a9cc815bfd2ed11ce5b40f0316252be69a98ac8f7
printf '%s  %s\n' "${expected_checkpoint_sha256}" "${checkpoint}" | sha256sum --check --status

output=/scratch/xzh/models/complete_total_capacity/20260717/stage5_mace_power_readout_v6g/jacobian_range_preconditioned_v3/stable5_s3_p8_i100_anchor
mkdir -p "${output}" /scratch/xzh/logs/complete_total_capacity

/usr/bin/time -v -o "${output}/resource.time" \
  python scripts/qm9_complete_total_local_scalar_jacobian_range_audit.py \
    --protocol configs/audit/qm9_complete_total_hessian_mace_power_readout_smoke_v6g.yaml \
    --arm-id E11b_mace_h16_power_rf2048_in1000 \
    --checkpoint "${checkpoint}" \
    --anchor-energy-force \
    --exact-linear-parameter-map \
    --hutchinson-preconditioner-probes 8 \
    --preconditioner-seed 20260806 \
    --output-dir "${output}" \
    --iterations 100 \
    --relative-tolerance 1e-5 \
    --fd-relative-step 3e-6 \
    --nonlinear-alphas 0.01,0.03,0.1,0.3,1.0 \
    --device cuda:0
