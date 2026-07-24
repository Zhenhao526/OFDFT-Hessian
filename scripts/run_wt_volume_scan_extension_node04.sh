#!/usr/bin/env bash
set -euo pipefail

root=/scratch/xzh/OFDFT-Hessian
parent=${PARENT_SCAN:-$root/runs/al/free_energy_wt/common_T0900_volume_scan_v1}
base=${EXTENSION_ROOT:-$root/runs/al/free_energy_wt/common_T0900_volume_scan_extension400_v2}
binary=$root/external/abacus-develop/build-wt-ti-cpu2/source/abacus_wt_ti_cpu
config=$root/config/abacus_wt_ti_node04_cpu12.json
mpirun=/scratch/xzh/conda/envs/abacus-gpu/bin/mpirun
log=${EXTENSION_LOG:-$root/runs/al/free_energy_wt/common_T0900_volume_scan_extension400_v2.log}

mkdir -p "$(dirname "$log")"
: > "$log"
env PYTHONPATH="$root" python3 "$root/scripts/prepare_al108_volume_scan_extension.py" \
    --parent "$parent" --out "$base" --config "$config" --temperature 900 --steps 400 \
    >> "$log" 2>&1

points=(
    "$base/solid/vpa_17p750"
    "$base/solid/vpa_17p950"
    "$base/solid/vpa_18p150"
    "$base/liquid/vpa_18p550"
    "$base/liquid/vpa_18p750"
    "$base/liquid/vpa_18p950"
)
pids=()
cpu_ranges=(0-11 12-23 24-35 38-49 50-61 62-73)
for slot in "${!points[@]}"; do
    point=${points[$slot]}
    cpus=${cpu_ranges[$slot]}
    (
        cd "$point"
        exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            taskset -c "$cpus" "$mpirun" -np 12 --bind-to none "$binary" > run.stdout 2>&1
    ) &
    pids+=("$!")
    printf '%s started point=%s cpus=%s pid=%s\n' \
        "$(date -Iseconds)" "$point" "$cpus" "$!" >> "$log"
done

failed=0
for slot in "${!pids[@]}"; do
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
env PYTHONPATH="$root" python3 scripts/prepare_al108_volume_scan.py analyze-md "$base/solid" >> "$log" 2>&1
env PYTHONPATH="$root" python3 scripts/prepare_al108_volume_scan.py analyze-md "$base/liquid" >> "$log" 2>&1
printf '%s wt_volume_scan_extension_complete\n' "$(date -Iseconds)" >> "$log"
