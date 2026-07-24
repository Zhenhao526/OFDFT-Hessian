#!/usr/bin/env bash
set -euo pipefail

root=/scratch/xzh/OFDFT-Hessian
source_dir=$root/external/abacus-develop
build_dir=$source_dir/build-gpu-gcc13-openmp
base=$root/runs/al/free_energy/ti_windows
log=$base/build_ti_abacus_node04.log

mkdir -p "$base"
: > "$log"
while tmux has-session -t '=fe_nvt_volume_scan_108_v2' 2>/dev/null; do
    printf '%s waiting_for_volume_scan\n' "$(date -Iseconds)" >> "$log"
    sleep 30
done

printf '%s configure_started\n' "$(date -Iseconds)" >> "$log"
cmake -S "$source_dir" -B "$build_dir" >> "$log" 2>&1
printf '%s build_started\n' "$(date -Iseconds)" >> "$log"
cmake --build "$build_dir" --target abacus_ml_para -j 72 >> "$log" 2>&1
cp "$build_dir/source/abacus_ml_gpu" "$build_dir/source/abacus_mpn_ti_cpu"
printf '%s build_completed binary=%s\n' "$(date -Iseconds)" "$build_dir/source/abacus_mpn_ti_cpu" >> "$log"
