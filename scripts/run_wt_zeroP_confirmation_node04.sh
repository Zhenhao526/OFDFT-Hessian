#!/usr/bin/env bash
set -euo pipefail

root=/scratch/xzh/OFDFT-Hessian
base=${CONFIRM_ROOT:-$root/runs/al/free_energy_wt/zeroP_confirmation_T0900_tau5_1000_v2}
binary=$root/external/abacus-develop/build-wt-ti-cpu2/source/abacus_wt_ti_cpu
config=$root/config/abacus_wt_ti_node04_cpu12.json
mpirun=/scratch/xzh/conda/envs/abacus-gpu/bin/mpirun
log=${CONFIRM_LOG:-$root/runs/al/free_energy_wt/zeroP_confirmation_T0900_tau5_1000_v2.log}
solid_source=$root/runs/al/free_energy_wt/common_T0900_volume_scan_extension400_v2/solid/vpa_18p150
liquid_source=$root/runs/al/free_energy_wt/common_T0900_volume_scan_extension400_v2/liquid/vpa_18p750

mkdir -p "$(dirname "$log")"
: > "$log"
env PYTHONPATH="$root" python3 "$root/scripts/prepare_wt_zero_pressure_confirmation.py" \
    --out "$base" --solid-source "$solid_source" --liquid-source "$liquid_source" \
    --solid-volume 18.051036222240153 --liquid-volume 18.736615407000613 \
    --temperature 900 --steps 1000 --csvr-tau 5 --config "$config" >> "$log" 2>&1

points=("$base/solid" "$base/liquid")
cpu_ranges=(0-35 38-73)
pids=()
for slot in 0 1; do
    point=${points[$slot]}
    cpus=${cpu_ranges[$slot]}
    (
        cd "$point"
        exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            taskset -c "$cpus" "$mpirun" -np 36 --bind-to none "$binary" > run.stdout 2>&1
    ) &
    pids+=("$!")
    printf '%s started point=%s cpus=%s pid=%s\n' \
        "$(date -Iseconds)" "$point" "$cpus" "$!" >> "$log"
done

failed=0
for slot in 0 1; do
    if wait "${pids[$slot]}"; then
        printf '%s finished point=%s\n' "$(date -Iseconds)" "${points[$slot]}" >> "$log"
    else
        status=$?
        printf '%s failed point=%s status=%s\n' \
            "$(date -Iseconds)" "${points[$slot]}" "$status" >> "$log"
        failed=1
    fi
done
if [[ "$failed" -ne 0 ]]; then
    exit "$failed"
fi

cd "$root"
env PYTHONPATH="$root" python3 scripts/analyze_wt_zero_pressure_confirmation.py "$base" \
    --temperature-tolerance 25 --pressure-tolerance 2.5 >> "$log" 2>&1
printf '%s wt_zero_pressure_confirmation_complete\n' "$(date -Iseconds)" >> "$log"
