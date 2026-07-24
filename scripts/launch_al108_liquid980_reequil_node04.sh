#!/usr/bin/env bash
set -euo pipefail

session=fe_liquid108_980_reequil
run_dir=/scratch/xzh/OFDFT-Hessian/runs/al/free_energy/mpn_samples/mpn_Al108_liquid_T0980_nvt_csvr500_tau10_reequil_from_T0980_node04
abacus=/scratch/xzh/OFDFT-Hessian/external/abacus-develop/build-gpu-gcc13-openmp/source/abacus_ml_gpu
mpirun=/scratch/xzh/conda/envs/abacus-gpu/bin/mpirun

if tmux has-session -t "=$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 1
fi
if find "$run_dir" -maxdepth 1 -type d -name 'OUT.*' -print -quit | grep -q .; then
    echo "ABACUS output already exists below: $run_dir" >&2
    exit 1
fi

tmux new-session -d -s "$session" \
    "cd \"$run_dir\" && exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0-31 \"$mpirun\" -np 32 --bind-to none \"$abacus\" > run.stdout 2>&1"

tmux list-sessions
