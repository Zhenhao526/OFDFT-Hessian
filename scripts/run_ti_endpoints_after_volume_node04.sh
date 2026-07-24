#!/usr/bin/env bash
set -euo pipefail

root=/scratch/xzh/OFDFT-Hessian
scan=${SCAN_ROOT:-$root/runs/al/free_energy_wt/common_T0900_volume_scan_extension400_v2}
base=${ENDPOINT_ROOT:-$root/runs/al/free_energy_wt/ti_windows/common_T0900_zeroP_endpoints_v1}
binary=$root/external/abacus-develop/build-wt-ti-cpu2/source/abacus_wt_ti_cpu
pair_json=$root/runs/al/free_energy_wt/reference_fit/wt_pair_s750_l980_final_v1.json
pair_dat=$root/runs/al/free_energy_wt/reference_fit/wt_pair_s750_l980_final_v1.dat
config=$root/config/abacus_wt_ti_node04_cpu12.json
log=${ENDPOINT_LOG:-$root/runs/al/free_energy_wt/ti_windows/common_T0900_zeroP_endpoints_v1.log}
mpirun=/scratch/xzh/conda/envs/abacus-gpu/bin/mpirun

mkdir -p "$(dirname "$log")"
: > "$log"
while [[ ! -f "$scan/solid/nvt_volume_scan_result.json" \
      || ! -f "$scan/liquid/nvt_volume_scan_result.json" \
      || ! -x "$binary" ]]; do
    printf '%s waiting_for_volume_results_or_ti_binary\n' "$(date -Iseconds)" >> "$log"
    sleep 30
done

env PYTHONPATH="$root" python3 "$root/scripts/prepare_ti_endpoints_from_volume_scan.py" \
    --scan-root "$scan" --out "$base" --pair-model "$pair_dat" --config "$config" \
    --temperature 900 --steps 10 \
    >> "$log" 2>&1

points=(
    "$base/solid/baseline/lambda_1p000"
    "$base/solid/ti/lambda_0p000"
    "$base/solid/ti/lambda_1p000"
    "$base/liquid/baseline/lambda_1p000"
    "$base/liquid/ti/lambda_0p000"
    "$base/liquid/ti/lambda_1p000"
)
lambdas=(baseline 0.0 1.0 baseline 0.0 1.0)
pids=()
cpu_ranges=(0-11 12-23 24-35 38-49 50-61 62-73)
for slot in "${!points[@]}"; do
    point=${points[$slot]}
    lambda=${lambdas[$slot]}
    cpus=${cpu_ranges[$slot]}
    (
        cd "$point"
        if [[ "$lambda" == baseline ]]; then
            exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
                taskset -c "$cpus" "$mpirun" -np 12 --bind-to none "$binary" > run.stdout 2>&1
        else
            exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
                MPN_TI_LAMBDA="$lambda" MPN_TI_PAIR_MODEL="$pair_dat" \
                taskset -c "$cpus" "$mpirun" -np 12 --bind-to none "$binary" > run.stdout 2>&1
        fi
    ) &
    pids+=("$!")
    printf '%s started endpoint=%s mode=%s cpus=%s pid=%s\n' \
        "$(date -Iseconds)" "$point" "$lambda" "$cpus" "$!" >> "$log"
done
for pid in "${pids[@]}"; do
    wait "$pid"
done

env PYTHONPATH="$root" python3 "$root/scripts/validate_al108_ti_endpoints.py" "$base/solid" \
    --pair-model "$pair_json" --expected-steps 10 >> "$log" 2>&1
env PYTHONPATH="$root" python3 "$root/scripts/validate_al108_ti_endpoints.py" "$base/liquid" \
    --pair-model "$pair_json" --expected-steps 10 >> "$log" 2>&1
printf '%s endpoint_validation_completed\n' "$(date -Iseconds)" >> "$log"
