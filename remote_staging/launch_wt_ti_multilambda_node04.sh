#!/usr/bin/env bash
set -euo pipefail

root=/scratch/xzh/OFDFT-Hessian
base=$root/runs/al/free_energy_wt/ti_windows/multilambda_s750_l980_steps300_final_v1
binary=$root/external/abacus-develop/build-wt-ti-cpu2/source/abacus_wt_ti_cpu
pair_dat=$root/runs/al/free_energy_wt/reference_fit/wt_pair_s750_l980_final_v1.dat
mpirun=/scratch/xzh/conda/envs/abacus-gpu/bin/mpirun
log=$base/run_multilambda_node04.log

points=(
    "$base/solid/lambda_0p000"
    "$base/solid/lambda_0p250"
    "$base/solid/lambda_0p500"
    "$base/solid/lambda_0p750"
    "$base/solid/lambda_1p000"
    "$base/liquid/lambda_0p000"
    "$base/liquid/lambda_0p250"
    "$base/liquid/lambda_0p500"
    "$base/liquid/lambda_0p750"
    "$base/liquid/lambda_1p000"
)
lambdas=(0.0 0.25 0.5 0.75 1.0 0.0 0.25 0.5 0.75 1.0)

for point in "${points[@]}"; do
    if find "$point" -maxdepth 1 -type d -name 'OUT.*' | grep -q .; then
        printf '%s refusing_existing_output point=%s\n' "$(date -Iseconds)" "$point"
        exit 2
    fi
done

: > "$log"
failed=0

run_wave() {
    local wave=$1
    local first=$2
    local last=$3
    local slot=0
    local index point lambda lo hi status
    local -a pids=()
    local -a wave_points=()

    printf '%s wave_started wave=%s first=%s last=%s\n' \
        "$(date -Iseconds)" "$wave" "$first" "$last" | tee -a "$log"
    for ((index=first; index<=last; index+=1)); do
        point=${points[$index]}
        lambda=${lambdas[$index]}
        lo=$((slot * 12))
        hi=$((lo + 11))
        (
            cd "$point"
            exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
                MPN_TI_LAMBDA="$lambda" MPN_TI_PAIR_MODEL="$pair_dat" \
                taskset -c "$lo-$hi" "$mpirun" -np 12 --bind-to none "$binary" \
                > run.stdout 2>&1
        ) &
        pids+=("$!")
        wave_points+=("$point")
        printf '%s started wave=%s point=%s lambda=%s cpus=%s-%s pid=%s\n' \
            "$(date -Iseconds)" "$wave" "$point" "$lambda" "$lo" "$hi" "$!" \
            | tee -a "$log"
        slot=$((slot + 1))
    done

    for index in "${!pids[@]}"; do
        if wait "${pids[$index]}"; then
            printf '%s finished wave=%s point=%s\n' \
                "$(date -Iseconds)" "$wave" "${wave_points[$index]}" | tee -a "$log"
        else
            status=$?
            printf '%s failed wave=%s point=%s status=%s\n' \
                "$(date -Iseconds)" "$wave" "${wave_points[$index]}" "$status" \
                | tee -a "$log"
            failed=1
        fi
    done
    printf '%s wave_finished wave=%s failed=%s\n' \
        "$(date -Iseconds)" "$wave" "$failed" | tee -a "$log"
}

run_wave 1 0 5
run_wave 2 6 9

cd "$root"
env PYTHONPATH="$root" python3 scripts/prepare_al108_ti_windows.py analyze "$base/solid" \
    >> "$log" 2>&1 || failed=1
env PYTHONPATH="$root" python3 scripts/prepare_al108_ti_windows.py analyze "$base/liquid" \
    >> "$log" 2>&1 || failed=1
printf '%s multilambda_pipeline_complete failed=%s\n' \
    "$(date -Iseconds)" "$failed" | tee -a "$log"
exit "$failed"
