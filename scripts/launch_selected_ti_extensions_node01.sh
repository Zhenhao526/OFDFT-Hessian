#!/usr/bin/env bash
set -euo pipefail

root=/scratch/xzh/OFDFT-Hessian
base=${1:?usage: launch_selected_ti_extensions_node01.sh RUN_ROOT}
binary=$root/external/abacus-develop/build-wt-ti-cpu2/source/abacus_wt_ti_cpu
mpirun=/scratch/xzh/conda/envs/abacus-gpu/bin/mpirun
ranks_per_window=12
slots=6
log=$base/run_selected_extensions_node01.log

mapfile -t records < <(
    python3 -c '
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
for phase_dir in sorted(path for path in root.iterdir() if path.is_dir()):
    manifest_path = phase_dir / "manifest.json"
    if not manifest_path.exists():
        continue
    manifest = json.loads(manifest_path.read_text())
    model = pathlib.Path(manifest["pair_model"]).resolve()
    for window in manifest["windows"]:
        point = phase_dir / window["label"]
        lambda_value = window["lambda"]
        print(f"{point}\t{lambda_value}\t{model}")
' "$base"
)

if ((${#records[@]} == 0)); then
    printf 'no selected windows below %s\n' "$base" >&2
    exit 2
fi
for record in "${records[@]}"; do
    IFS=$'\t' read -r point _ _ <<< "$record"
    if find "$point" -maxdepth 1 -type d -name 'OUT.*' | grep -q .; then
        printf '%s refusing_existing_output point=%s\n' "$(date -Iseconds)" "$point"
        exit 2
    fi
done

: > "$log"
failed=0
wave=0
for ((first=0; first<${#records[@]}; first+=slots)); do
    wave=$((wave + 1))
    last=$((first + slots - 1))
    if ((last >= ${#records[@]})); then
        last=$((${#records[@]} - 1))
    fi
    printf '%s wave_started wave=%d first=%d last=%d\n' \
        "$(date -Iseconds)" "$wave" "$first" "$last" | tee -a "$log"
    pids=()
    points=()
    slot=0
    for ((index=first; index<=last; index+=1)); do
        IFS=$'\t' read -r point lambda model <<< "${records[$index]}"
        lo=$((slot * ranks_per_window))
        hi=$((lo + ranks_per_window - 1))
        (
            cd "$point"
            exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
                MPN_TI_LAMBDA="$lambda" MPN_TI_PAIR_MODEL="$model" \
                taskset -c "$lo-$hi" "$mpirun" -np "$ranks_per_window" \
                --bind-to none "$binary" > run.stdout 2>&1
        ) &
        pids+=("$!")
        points+=("$point")
        printf '%s started wave=%d point=%s lambda=%s model=%s cpus=%d-%d pid=%s\n' \
            "$(date -Iseconds)" "$wave" "$point" "$lambda" "$model" "$lo" "$hi" "$!" \
            | tee -a "$log"
        slot=$((slot + 1))
    done
    for index in "${!pids[@]}"; do
        if wait "${pids[$index]}"; then
            printf '%s finished wave=%d point=%s\n' \
                "$(date -Iseconds)" "$wave" "${points[$index]}" | tee -a "$log"
        else
            status=$?
            printf '%s failed wave=%d point=%s status=%d\n' \
                "$(date -Iseconds)" "$wave" "${points[$index]}" "$status" | tee -a "$log"
            failed=1
        fi
    done
    printf '%s wave_finished wave=%d failed=%d\n' \
        "$(date -Iseconds)" "$wave" "$failed" | tee -a "$log"
done

printf '%s selected_ti_extensions_complete failed=%d\n' \
    "$(date -Iseconds)" "$failed" | tee -a "$log"
exit "$failed"
