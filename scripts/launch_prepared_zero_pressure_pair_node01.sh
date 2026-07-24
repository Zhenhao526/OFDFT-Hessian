#!/usr/bin/env bash
set -euo pipefail

root=/scratch/xzh/OFDFT-Hessian
base=${1:?usage: launch_prepared_zero_pressure_pair_node01.sh RUN_ROOT}
binary=$root/external/abacus-develop/build-wt-ti-cpu2/source/abacus_wt_ti_cpu
mpirun=/scratch/xzh/conda/envs/abacus-gpu/bin/mpirun
log=$base/run_zero_pressure_pair_node01.log
points=("$base/solid" "$base/liquid")
cpu_ranges=(0-35 38-73)

for point in "${points[@]}"; do
    if [[ ! -f $point/INPUT || ! -f $point/STRU ]]; then
        printf 'missing prepared input below %s\n' "$point" >&2
        exit 2
    fi
    if find "$point" -maxdepth 1 -type d -name 'OUT.*' | grep -q .; then
        printf '%s refusing_existing_output point=%s\n' "$(date -Iseconds)" "$point"
        exit 2
    fi
done

: > "$log"
pids=()
for slot in 0 1; do
    point=${points[$slot]}
    cpus=${cpu_ranges[$slot]}
    (
        cd "$point"
        exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            taskset -c "$cpus" "$mpirun" -np 36 --bind-to none "$binary" \
            > run.stdout 2>&1
    ) &
    pids+=("$!")
    printf '%s started point=%s cpus=%s pid=%s\n' \
        "$(date -Iseconds)" "$point" "$cpus" "$!" | tee -a "$log"
done

failed=0
for slot in 0 1; do
    if wait "${pids[$slot]}"; then
        printf '%s finished point=%s\n' \
            "$(date -Iseconds)" "${points[$slot]}" | tee -a "$log"
    else
        status=$?
        printf '%s failed point=%s status=%s\n' \
            "$(date -Iseconds)" "${points[$slot]}" "$status" | tee -a "$log"
        failed=1
    fi
done
if ((failed != 0)); then
    exit "$failed"
fi

cd "$root"
env PYTHONPATH="$root" python3 scripts/analyze_wt_zero_pressure_confirmation.py "$base" \
    --temperature-tolerance 20 --pressure-tolerance 2.5 | tee -a "$log"
confirmation_status=$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$base/confirmation_summary.json")
if [[ $confirmation_status != all_confirmations_passed ]]; then
    printf '%s zero_pressure_pair_gate_failed status=%s\n' \
        "$(date -Iseconds)" "$confirmation_status" | tee -a "$log"
    exit 2
fi
printf '%s zero_pressure_pair_complete\n' "$(date -Iseconds)" | tee -a "$log"
