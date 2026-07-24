#!/usr/bin/env bash
set -euo pipefail

root=/scratch/xzh/OFDFT-Hessian
confirmation=${CONFIRM_ROOT:-$root/runs/al/free_energy_wt/zeroP_confirmation_T0900_tau5_1000_v2}
base=${ENDPOINT_ROOT:-$root/runs/al/free_energy_wt/ti_windows/common_T0900_zeroP_endpoints_v1}
binary=$root/external/abacus-develop/build-wt-ti-cpu2/source/abacus_wt_ti_cpu
pair_json=$root/runs/al/free_energy_wt/reference_fit/wt_pair_s750_l980_final_v1.json
pair_dat=$root/runs/al/free_energy_wt/reference_fit/wt_pair_s750_l980_final_v1.dat
config=$root/config/abacus_wt_ti_node04_cpu12.json
log=${ENDPOINT_LOG:-$root/runs/al/free_energy_wt/ti_windows/common_T0900_zeroP_endpoints_v1.log}
mpirun=/scratch/xzh/conda/envs/abacus-gpu/bin/mpirun

mkdir -p "$(dirname "$log")"
: > "$log"
while [[ ! -f "$confirmation/confirmation_summary.json" ]]; do
    printf '%s waiting_for_zero_pressure_confirmation\n' "$(date -Iseconds)" >> "$log"
    sleep 30
done

env PYTHONPATH="$root" python3 "$root/scripts/prepare_ti_endpoints_from_confirmations.py" \
    --confirmation-root "$confirmation" --out "$base" --pair-model "$pair_dat" \
    --config "$config" --steps 10 >> "$log" 2>&1

points=(
    "$base/solid/baseline/lambda_1p000"
    "$base/solid/ti/lambda_0p000"
    "$base/solid/ti/lambda_1p000"
    "$base/liquid/baseline/lambda_1p000"
    "$base/liquid/ti/lambda_0p000"
    "$base/liquid/ti/lambda_1p000"
)
lambdas=(baseline 0.0 1.0 baseline 0.0 1.0)
cpu_ranges=(0-11 12-23 24-35 38-49 50-61 62-73)
pids=()
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

cd "$root"
env PYTHONPATH="$root" python3 scripts/validate_al108_ti_endpoints.py "$base/solid" \
    --pair-model "$pair_json" --expected-steps 10 >> "$log" 2>&1
env PYTHONPATH="$root" python3 scripts/validate_al108_ti_endpoints.py "$base/liquid" \
    --pair-model "$pair_json" --expected-steps 10 >> "$log" 2>&1
grep -q '"status": "endpoint_validation_passed"' "$base/solid/endpoint_validation.json"
grep -q '"status": "endpoint_validation_passed"' "$base/liquid/endpoint_validation.json"
printf '%s endpoint_validation_completed\n' "$(date -Iseconds)" >> "$log"
