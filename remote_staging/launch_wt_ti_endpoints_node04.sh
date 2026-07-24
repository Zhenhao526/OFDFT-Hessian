#!/usr/bin/env bash
set -euo pipefail

root=/scratch/xzh/OFDFT-Hessian
base=$root/runs/al/free_energy_wt/ti_windows/endpoints_s750_l980_final_v1
binary=$root/external/abacus-develop/build-wt-ti-cpu2/source/abacus_wt_ti_cpu
pair_dat=$root/runs/al/free_energy_wt/reference_fit/wt_pair_s750_l980_final_v1.dat
mpirun=/scratch/xzh/conda/envs/abacus-gpu/bin/mpirun
log=$base/run_endpoints_node04.log

points=(
    "$base/solid/lambda_0p000"
    "$base/solid/lambda_1p000"
    "$base/liquid/lambda_0p000"
    "$base/liquid/lambda_1p000"
)
lambdas=(0.0 1.0 0.0 1.0)

: > "$log"
pids=()
for index in "${!points[@]}"; do
    point=${points[$index]}
    lambda=${lambdas[$index]}
    lo=$((index * 12))
    hi=$((lo + 11))
    if find "$point" -maxdepth 1 -type d -name 'OUT.*' | grep -q .; then
        printf '%s refusing_existing_output point=%s\n' "$(date -Iseconds)" "$point" | tee -a "$log"
        exit 2
    fi
    (
        cd "$point"
        exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            MPN_TI_LAMBDA="$lambda" MPN_TI_PAIR_MODEL="$pair_dat" \
            taskset -c "$lo-$hi" "$mpirun" -np 12 --bind-to none "$binary" > run.stdout 2>&1
    ) &
    pids+=("$!")
    printf '%s started point=%s lambda=%s cpus=%s-%s pid=%s\n' \
        "$(date -Iseconds)" "$point" "$lambda" "$lo" "$hi" "$!" | tee -a "$log"
done

failed=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        printf '%s finished point=%s\n' "$(date -Iseconds)" "${points[$index]}" | tee -a "$log"
    else
        status=$?
        printf '%s failed point=%s status=%s\n' \
            "$(date -Iseconds)" "${points[$index]}" "$status" | tee -a "$log"
        failed=1
    fi
done

cd "$root"
env PYTHONPATH="$root" python3 scripts/prepare_al108_ti_windows.py analyze "$base/solid" >> "$log" 2>&1 || failed=1
env PYTHONPATH="$root" python3 scripts/prepare_al108_ti_windows.py analyze "$base/liquid" >> "$log" 2>&1 || failed=1
printf '%s endpoint_pipeline_complete failed=%s\n' "$(date -Iseconds)" "$failed" | tee -a "$log"
exit "$failed"
