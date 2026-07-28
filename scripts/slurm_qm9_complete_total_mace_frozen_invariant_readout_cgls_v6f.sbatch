#!/usr/bin/env bash
#SBATCH --job-name=ct_mace_frrf_cg
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
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

checkpoint=/scratch/xzh/models/complete_total_capacity/20260717/stage5_mace_frozen_invariant_readout_smoke_v6f/smoke/E11a_mace_h16_frozen_rf512_scale1000/best.ckpt
expected_checkpoint_sha256=f53a0602f04f3eadb505ca38988f6125e7a6c4e22b0d7cccabc16b28a7a575c2
printf '%s  %s\n' "${expected_checkpoint_sha256}" "${checkpoint}" | sha256sum --check --status

output=/scratch/xzh/models/complete_total_capacity/20260717/stage5_mace_frozen_invariant_readout_v6f/jacobian_range_v1/stable5_s5_i100_anchor
mkdir -p "${output}" /scratch/xzh/logs/complete_total_capacity

/usr/bin/time -v -o "${output}/resource.time" \
  python scripts/qm9_complete_total_local_scalar_jacobian_range_audit.py \
    --protocol configs/audit/qm9_complete_total_hessian_mace_frozen_invariant_readout_smoke_v6f.yaml \
    --arm-id E11a_mace_h16_frozen_rf512_scale1000 \
    --checkpoint "${checkpoint}" \
    --anchor-energy-force \
    --output-dir "${output}" \
    --iterations 100 \
    --relative-tolerance 1e-5 \
    --fd-relative-step 3e-6 \
    --nonlinear-alphas 0.01,0.03,0.1,0.3,1.0 \
    --device cuda:0
