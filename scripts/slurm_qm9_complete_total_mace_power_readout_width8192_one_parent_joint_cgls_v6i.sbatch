#!/usr/bin/env bash
#SBATCH --job-name=ct_mace_p8k_1j
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
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

checkpoint=/scratch/xzh/models/complete_total_capacity/20260717/stage5_mace_power_readout_width8192_smoke_v6h/smoke/E11c_mace_h16_power_rf8192_in1000/best.ckpt
expected_checkpoint_sha256=a10f8eafb88f68d12617174b7e716fd142a34827d3d8ffab3c673e653ffe6985
printf '%s  %s\n' "${expected_checkpoint_sha256}" "${checkpoint}" | sha256sum --check --status

output=/scratch/xzh/models/complete_total_capacity/20260717/stage5_mace_power_readout_joint_jet_v6i/one_parent_cgls_v1/0028399_s0_p4_i200_joint
mkdir -p "${output}" /scratch/xzh/logs/complete_total_capacity

/usr/bin/time -v -o "${output}/resource.time" \
  python scripts/qm9_complete_total_local_scalar_jacobian_range_audit.py \
    --protocol configs/audit/qm9_complete_total_hessian_mace_power_readout_joint_jet_v6i.yaml \
    --checkpoint-protocol configs/audit/qm9_complete_total_hessian_mace_power_readout_width8192_smoke_v6h.yaml \
    --arm-id E11c_mace_h16_power_rf8192_in1000 \
    --checkpoint "${checkpoint}" \
    --parent-id 0028399 \
    --include-energy-force-targets \
    --energy-scale 0.1 \
    --force-scale 0.05 \
    --exact-linear-parameter-map \
    --hutchinson-preconditioner-probes 4 \
    --preconditioner-seed 20260810 \
    --output-dir "${output}" \
    --iterations 200 \
    --relative-tolerance 1e-5 \
    --fd-relative-step 3e-6 \
    --nonlinear-alphas 0.01,0.03,0.1,0.3,1.0 \
    --device cuda:0
