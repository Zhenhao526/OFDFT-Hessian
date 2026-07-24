#!/usr/bin/env bash
set -euo pipefail

root=/scratch/xzh/OFDFT-Hessian
base=$root/runs/al/free_energy/volume_scan/nvt_T0850_v2
solid=$base/solid
liquid=$base/liquid
log=$base/nvt_volume_scan_node04.log
mpirun=/scratch/xzh/conda/envs/abacus-gpu/bin/mpirun
abacus=$root/external/abacus-develop/build-gpu-gcc13-openmp/source/abacus_ml_gpu

mkdir -p "$base"
: > "$log"
points=(
    "$solid/vpa_17p650"
    "$solid/vpa_17p800"
    "$solid/vpa_17p950"
    "$liquid/vpa_18p450"
    "$liquid/vpa_18p650"
    "$liquid/vpa_18p850"
)
pids=()
for slot in "${!points[@]}"; do
    point=${points[$slot]}
    lo=$((slot * 12))
    hi=$((lo + 11))
    (
        cd "$point"
        exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            taskset -c "$lo-$hi" "$mpirun" -np 12 --bind-to none "$abacus" > run.stdout 2>&1
    ) &
    pids+=("$!")
    printf '%s started point=%s cpus=%s-%s pid=%s\n' "$(date -Iseconds)" "$point" "$lo" "$hi" "$!" >> "$log"
done
for pid in "${pids[@]}"; do
    wait "$pid"
done

env PYTHONPATH="$root" python3 "$root/scripts/prepare_al108_volume_scan.py" analyze-md "$solid" >> "$log" 2>&1
env PYTHONPATH="$root" python3 "$root/scripts/prepare_al108_volume_scan.py" analyze-md "$liquid" >> "$log" 2>&1
printf '%s completed\n' "$(date -Iseconds)" >> "$log"
