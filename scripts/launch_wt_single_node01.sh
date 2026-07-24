#!/usr/bin/env bash
set -euo pipefail

root=/scratch/xzh/OFDFT-Hessian
run=${1:?usage: launch_wt_single_node01.sh RUN_DIR MPI_RANKS CPU_RANGE}
ranks=${2:?usage: launch_wt_single_node01.sh RUN_DIR MPI_RANKS CPU_RANGE}
cpus=${3:?usage: launch_wt_single_node01.sh RUN_DIR MPI_RANKS CPU_RANGE}
binary=$root/external/abacus-develop/build-wt-ti-cpu2/source/abacus_wt_ti_cpu
mpirun=/scratch/xzh/conda/envs/abacus-gpu/bin/mpirun

if [[ ! -f $run/INPUT || ! -f $run/STRU ]]; then
    printf 'missing prepared WT input below %s\n' "$run" >&2
    exit 2
fi
if find "$run" -maxdepth 1 -type d -name 'OUT.*' | grep -q .; then
    printf '%s refusing_existing_output run=%s\n' "$(date -Iseconds)" "$run" >&2
    exit 2
fi
printf '%s starting_wt run=%s ranks=%s cpus=%s\n' \
    "$(date -Iseconds)" "$run" "$ranks" "$cpus" | tee "$run/launch_node01.log"
(
    cd "$run"
    exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        taskset -c "$cpus" "$mpirun" -np "$ranks" --bind-to none "$binary" \
        > run.stdout 2>&1
)
printf '%s finished_wt run=%s\n' "$(date -Iseconds)" "$run" \
    | tee -a "$run/launch_node01.log"
