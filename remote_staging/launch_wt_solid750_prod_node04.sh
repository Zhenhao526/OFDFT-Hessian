#!/usr/bin/env bash
set -euo pipefail

root=/scratch/xzh/OFDFT-Hessian
run_dir=$root/runs/al/free_energy_wt/production/solid_T0750_vpa17p742_from_wt300_tau10/vpa_17p742
mpirun=/scratch/xzh/conda/envs/abacus-gpu/bin/mpirun
abacus=$root/external/abacus-develop/build-gpu-gcc13-openmp/source/abacus_ml_gpu
session=wt_sp750_prod

if tmux has-session -t "=$session" 2>/dev/null; then
    echo "session already exists: $session" >&2
    exit 1
fi
if [[ -d $run_dir/OUT.al108_solid_T0750_vpa_17p742 ]]; then
    echo "refusing to overwrite existing output in $run_dir" >&2
    exit 1
fi

tmux new-session -d -s "$session" \
    "cd '$run_dir' && env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    taskset -c 0-23 '$mpirun' -np 24 --bind-to none '$abacus' > run.stdout 2>&1"
echo "launched session=$session cpus=0-23 ranks=24 run_dir=$run_dir"
