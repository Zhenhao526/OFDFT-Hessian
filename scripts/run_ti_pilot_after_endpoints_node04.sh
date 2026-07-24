#!/usr/bin/env bash
set -euo pipefail

root=/scratch/xzh/OFDFT-Hessian
endpoints=${ENDPOINT_ROOT:-$root/runs/al/free_energy_wt/ti_windows/common_T0900_zeroP_endpoints_v1}
base=${PILOT_ROOT:-$root/runs/al/free_energy_wt/ti_windows/common_T0900_zeroP_lambda9_steps300_v1}
binary=$root/external/abacus-develop/build-wt-ti-cpu2/source/abacus_wt_ti_cpu
pair_dat=$root/runs/al/free_energy_wt/reference_fit/wt_pair_s750_l980_final_v1.dat
config=$root/config/abacus_wt_ti_node04_cpu12.json
log=${PILOT_LOG:-$root/runs/al/free_energy_wt/ti_windows/common_T0900_zeroP_lambda9_steps300_v1.log}
mpirun=/scratch/xzh/conda/envs/abacus-gpu/bin/mpirun

mkdir -p "$(dirname "$log")"
: > "$log"
while [[ ! -f "$endpoints/solid/endpoint_validation.json" \
      || ! -f "$endpoints/liquid/endpoint_validation.json" ]]; do
    printf '%s waiting_for_endpoint_validation\n' "$(date -Iseconds)" >> "$log"
    sleep 30
done

env PYTHONPATH="$root" python3 "$root/scripts/prepare_ti_pilot_from_endpoints.py" \
    --endpoint-root "$endpoints" --out "$base" --pair-model "$pair_dat" \
    --config "$config" --steps 300 \
    >> "$log" 2>&1

labels=(lambda_0p000 lambda_0p125 lambda_0p250 lambda_0p375 lambda_0p500 lambda_0p625 lambda_0p750 lambda_0p875 lambda_1p000)
lambdas=(0.0 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1.0)
points=()
values=()
for phase in solid liquid; do
    for index in "${!labels[@]}"; do
        points+=("$base/$phase/${labels[$index]}")
        values+=("${lambdas[$index]}")
    done
done

run_wave() {
    local first=$1
    local count=$2
    local pids=()
    local cpu_ranges=(0-11 12-23 24-35 38-49 50-61 62-73)
    for ((slot=0; slot<count; slot++)); do
        index=$((first + slot))
        point=${points[$index]}
        lambda=${values[$index]}
        cpus=${cpu_ranges[$slot]}
        (
            cd "$point"
            exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
                MPN_TI_LAMBDA="$lambda" MPN_TI_PAIR_MODEL="$pair_dat" \
                taskset -c "$cpus" "$mpirun" -np 12 --bind-to none "$binary" > run.stdout 2>&1
        ) &
        pids+=("$!")
        printf '%s started window=%s lambda=%s cpus=%s pid=%s\n' \
            "$(date -Iseconds)" "$point" "$lambda" "$cpus" "$!" >> "$log"
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
}

run_wave 0 6
run_wave 6 6
run_wave 12 6

env PYTHONPATH="$root" python3 "$root/scripts/prepare_al108_ti_windows.py" analyze "$base/solid" >> "$log" 2>&1
env PYTHONPATH="$root" python3 "$root/scripts/prepare_al108_ti_windows.py" analyze "$base/liquid" >> "$log" 2>&1
printf '%s ti_pilot_completed\n' "$(date -Iseconds)" >> "$log"
