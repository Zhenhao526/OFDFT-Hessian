#!/usr/bin/env bash
set -u

root=/scratch/xzh/OFDFT-Hessian
base=$root/runs/al/free_energy/pressure_validation
liquid=$base/liquid_Al108_T0980_fd
solid=$base/solid_Al108_T0850_fd
abacus=$root/external/abacus-develop/build-gpu-gcc13-openmp/source/abacus_ml_gpu
mpirun=/scratch/xzh/conda/envs/abacus-gpu/bin/mpirun
log=$base/pressure_fd_node04.log

exec >>"$log" 2>&1

jobs=(
    "$liquid/minus:0-11"
    "$liquid/center:12-23"
    "$liquid/plus:24-35"
    "$solid/minus:36-47"
    "$solid/center:48-59"
    "$solid/plus:60-71"
)

for spec in "${jobs[@]}"; do
    point=${spec%%:*}
    if find "$point" -maxdepth 1 -type d -name 'OUT.*' -print -quit | grep -q .; then
        echo "refusing existing ABACUS output: $point" >&2
        exit 1
    fi
done

pids=()
for spec in "${jobs[@]}"; do
    point=${spec%%:*}
    cpus=${spec##*:}
    (
        cd "$point" || exit 1
        exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            taskset -c "$cpus" "$mpirun" -np 12 --bind-to none "$abacus" \
            > run.stdout 2>&1
    ) &
    pids+=("$!")
    printf '%s started point=%s cpus=%s pid=%s\n' \
        "$(date --iso-8601=seconds)" "$point" "$cpus" "$!"
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
printf '%s static_jobs_exit=%s\n' "$(date --iso-8601=seconds)" "$status"
if [[ $status -ne 0 ]]; then
    exit "$status"
fi

cd "$root" || exit 1
PYTHONPATH=. /usr/bin/python3 scripts/check_mpn_pressure_finite_difference.py \
    analyze "$liquid" > "$liquid/analysis.stdout"
liquid_status=$?
PYTHONPATH=. /usr/bin/python3 scripts/check_mpn_pressure_finite_difference.py \
    analyze "$solid" > "$solid/analysis.stdout"
solid_status=$?

cat "$liquid/analysis.stdout"
cat "$solid/analysis.stdout"
printf '%s liquid_analysis_exit=%s solid_analysis_exit=%s\n' \
    "$(date --iso-8601=seconds)" "$liquid_status" "$solid_status"
exit $((liquid_status != 0 || solid_status != 0))
