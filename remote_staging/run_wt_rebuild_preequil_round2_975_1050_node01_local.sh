#!/usr/bin/env bash
set -euo pipefail

workspace=${WORKSPACE:-/home/shenwei01/WT_Al_melting_workspace_20260724}
repo=${REPO:-$workspace/repository}
runtime=${RUNTIME:-/home/shenwei01/wt_melting_runtime_20260724}
root=${RUN_ROOT:-$workspace/runs/step4_rebuild}
config=$repo/config/abacus_wt_node01_local_cpu18.json
binary=$runtime/build-abacus-wt-cpu/source/abacus_pw_para
mpirun=$runtime/conda_prefix/bin/mpirun
log=$workspace/audit/step4_rebuild_preequil_round2.log
done_file=$workspace/audit/step4_rebuild_preequil_round2.done
failed_file=$workspace/audit/step4_rebuild_preequil_round2.failed

mkdir -p "$root" "$workspace/audit"
rm -f "$done_file" "$failed_file"
printf '%s preequil_round2_started\n' "$(date -Iseconds)" > "$log"

mark_failed() {
    local rc=$?
    trap - ERR
    if [[ ! -e $done_file ]]; then
        printf '%s preequil_round2_failed exit_code=%s\n' \
            "$(date -Iseconds)" "$rc" | tee -a "$log" "$failed_file"
    fi
    exit "$rc"
}
trap mark_failed ERR

test -x "$binary"
test -x "$mpirun"

prepare_temperature() {
    local temperature=$1
    local solid_volume=$2
    local liquid_volume=$3
    local seed=$4
    local parent=$root/T${temperature}_preequil500_from_T1100
    local out=$root/T${temperature}_preequil500_round2_preservevel
    local solid_source
    local liquid_source

    solid_source=$(find "$parent/solid" -path '*/MD_dump' -type f | head -1)
    liquid_source=$(find "$parent/liquid" -path '*/MD_dump' -type f | head -1)
    test -s "$solid_source"
    test -s "$liquid_source"
    if [[ ! -e $out/confirmation_manifest.json ]]; then
        env PYTHONPATH="$repo" python3 \
            "$repo/scripts/prepare_wt_zero_pressure_confirmation.py" \
            --out "$out" \
            --solid-source "$solid_source" \
            --liquid-source "$liquid_source" \
            --solid-volume "$solid_volume" \
            --liquid-volume "$liquid_volume" \
            --temperature "$temperature" \
            --steps 500 \
            --csvr-tau 2 \
            --seed "$seed" \
            --config "$config" >> "$log" 2>&1
    fi
}

prepare_temperature 0975 18.08480608406171 18.891575304254044 2026077975
prepare_temperature 1050 18.250457879770096 19.043 2026078050

points=(
    "$root/T0975_preequil500_round2_preservevel/solid"
    "$root/T0975_preequil500_round2_preservevel/liquid"
    "$root/T1050_preequil500_round2_preservevel/solid"
    "$root/T1050_preequil500_round2_preservevel/liquid"
)
cpu_ranges=(0-17 19-36 38-55 57-74)

for point in "${points[@]}"; do
    test -s "$point/INPUT"
    test -s "$point/STRU"
    test -s "$point/KPT"
    if find "$point" -maxdepth 1 -type d -name 'OUT.*' | grep -q .; then
        printf '%s refusing_existing_output point=%s\n' \
            "$(date -Iseconds)" "$point" | tee -a "$log"
        exit 2
    fi
done

pids=()
for index in "${!points[@]}"; do
    point=${points[$index]}
    cpus=${cpu_ranges[$index]}
    (
        cd "$point"
        exec env \
            PATH="$runtime/conda_prefix/bin:$PATH" \
            LD_LIBRARY_PATH="$runtime/conda_prefix/lib:${LD_LIBRARY_PATH:-}" \
            OMP_NUM_THREADS=1 \
            OPENBLAS_NUM_THREADS=1 \
            MKL_NUM_THREADS=1 \
            taskset -c "$cpus" "$mpirun" -np 18 --bind-to none "$binary" \
            > run.stdout 2>&1
    ) &
    pids+=("$!")
    printf '%s started point=%s cpus=%s pid=%s\n' \
        "$(date -Iseconds)" "$point" "$cpus" "$!" | tee -a "$log"
done

run_failed=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        printf '%s finished point=%s\n' \
            "$(date -Iseconds)" "${points[$index]}" | tee -a "$log"
    else
        rc=$?
        printf '%s failed point=%s exit_code=%s\n' \
            "$(date -Iseconds)" "${points[$index]}" "$rc" | tee -a "$log"
        run_failed=1
    fi
done
if ((run_failed != 0)); then
    exit 2
fi

gate_failed=0
for temperature in 0975 1050; do
    out=$root/T${temperature}_preequil500_round2_preservevel
    env PYTHONPATH="$repo" python3 \
        "$repo/scripts/analyze_wt_zero_pressure_confirmation.py" \
        "$out" --temperature-tolerance 20 --pressure-tolerance 2.5 \
        >> "$log" 2>&1
    status=$(python3 - "$out/confirmation_summary.json" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])
PY
)
    if [[ $status != all_confirmations_passed ]]; then
        gate_failed=1
    fi
done

(
    cd "$root"
    find T0975_preequil500_round2_preservevel \
        T1050_preequil500_round2_preservevel -type f -print0 \
        | sort -z | xargs -0 sha256sum
) > "$root/PREEQUIL_ROUND2_SHA256SUMS"

if ((gate_failed != 0)); then
    printf '%s preequil_round2_gate_failed\n' "$(date -Iseconds)" \
        | tee -a "$log" "$failed_file"
    exit 2
fi

date -Iseconds > "$done_file"
printf '%s preequil_round2_verified\n' "$(date -Iseconds)" | tee -a "$log"
