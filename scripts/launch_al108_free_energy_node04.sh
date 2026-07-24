#!/usr/bin/env bash
set -euo pipefail

base=/scratch/xzh/OFDFT-Hessian/runs/al/free_energy/mpn_samples
abacus=/scratch/xzh/OFDFT-Hessian/external/abacus-develop/build-gpu-gcc13-openmp/source/abacus_ml_gpu
mpirun=/scratch/xzh/conda/envs/abacus-gpu/bin/mpirun

launch_job() {
    local session=$1
    local run_dir=$2
    local cpus=$3
    local ranks=$4

    if tmux has-session -t "$session" 2>/dev/null; then
        echo "tmux session already exists: $session" >&2
        return 1
    fi
    if find "$run_dir" -maxdepth 1 -type d -name 'OUT.*' -print -quit | grep -q .; then
        echo "ABACUS output already exists below: $run_dir" >&2
        return 1
    fi
    tmux new-session -d -s "$session" \
        "cd \"$run_dir\" && exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c \"$cpus\" \"$mpirun\" -np \"$ranks\" --bind-to none \"$abacus\" > run.stdout 2>&1"
}

launch_job \
    fe_solid108_850 \
    "$base/mpn_Al108_solid_T0850_nvt_csvr500_v2" \
    0-31 \
    32

launch_job \
    fe_liquid108_1600 \
    "$base/mpn_Al108_liquid_T1600_nvt_csvr1000" \
    38-69 \
    32

tmux list-sessions
