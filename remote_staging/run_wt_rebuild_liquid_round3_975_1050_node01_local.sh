#!/usr/bin/env bash
set -euo pipefail

workspace=${WORKSPACE:-/home/shenwei01/WT_Al_melting_workspace_20260724}
repo=${REPO:-$workspace/repository}
runtime=${RUNTIME:-/home/shenwei01/wt_melting_runtime_20260724}
root=${RUN_ROOT:-$workspace/runs/step4_rebuild}
config=$repo/config/abacus_wt_node01_local_cpu18.json
binary=$runtime/build-abacus-wt-cpu/source/abacus_pw_para
mpirun=$runtime/conda_prefix/bin/mpirun
log=$workspace/audit/step4_rebuild_liquid_round3.log
done_file=$workspace/audit/step4_rebuild_liquid_round3.done
failed_file=$workspace/audit/step4_rebuild_liquid_round3.failed

mkdir -p "$root" "$workspace/audit"
rm -f "$done_file" "$failed_file"
printf '%s liquid_round3_started\n' "$(date -Iseconds)" > "$log"

mark_failed() {
    local rc=$?
    trap - ERR
    if [[ ! -e $done_file ]]; then
        printf '%s liquid_round3_failed exit_code=%s\n' \
            "$(date -Iseconds)" "$rc" | tee -a "$log" "$failed_file"
    fi
    exit "$rc"
}
trap mark_failed ERR

prepare_liquid() {
    local temperature=$1
    local volume=$2
    local seed=$3
    local parent=$root/T${temperature}_preequil500_round2_preservevel/liquid
    local out=$root/T${temperature}_liquid_preequil500_round3_preservevel
    local source
    source=$(find "$parent" -path '*/MD_dump' -type f | head -1)
    test -s "$source"
    if [[ ! -e $out/phase_continuation_manifest.json ]]; then
        env PYTHONPATH="$repo" python3 \
            "$repo/scripts/prepare_wt_phase_continuation.py" \
            --source "$source" --out "$out" --phase liquid \
            --volume "$volume" --temperature "$temperature" \
            --steps 500 --csvr-tau 2 --seed "$seed" --config "$config" \
            >> "$log" 2>&1
    fi
}

prepare_liquid 0975 18.891575304254044 2026078975
prepare_liquid 1050 19.043 2026079050

points=(
    "$root/T0975_liquid_preequil500_round3_preservevel"
    "$root/T1050_liquid_preequil500_round3_preservevel"
)
cpu_ranges=(0-35 38-73)

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
            taskset -c "$cpus" "$mpirun" -np 36 --bind-to none "$binary" \
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
for point in "${points[@]}"; do
    env PYTHONPATH="$repo" python3 "$repo/scripts/analyze_phase_run.py" \
        "$point" --expected liquid --thermalized-initial \
        --out "$point/phase_analysis.json" >> "$log" 2>&1
    if env PYTHONPATH="$repo" python3 - "$point" <<'PY'
import json
import math
import sys
from pathlib import Path

from scripts.analyze_two_phase_run import parse_md_log, series_stats

run = Path(sys.argv[1]).resolve()
manifest = json.loads(
    (run / "phase_continuation_manifest.json").read_text(encoding="utf-8")
)
phase = json.loads((run / "phase_analysis.json").read_text(encoding="utf-8"))
logs = sorted(run.glob("OUT.*/running_md.log"))
rows, max_step = parse_md_log(logs[-1])
late = rows[len(rows) // 2 :]
temperature = series_stats([row["temperature_K"] for row in late])
pressure = series_stats([row["pressure_kbar"] for row in late])
nearest = phase.get("trajectory", {}).get("nearest_neighbor_A", 0.0)
checks = {
    "reached_requested_step": max_step >= int(manifest["steps"]),
    "phase_verified": phase.get("status") == "liquid_verified",
    "temperature_mean_within_tolerance": abs(
        float(temperature.get("mean", math.inf))
        - float(manifest["temperature_K"])
    )
    <= 20.0,
    "pressure_mean_within_tolerance": abs(
        float(pressure.get("mean", math.inf))
    )
    <= 2.5,
    "nearest_neighbor_gt_2_A": float(nearest) > 2.0,
}
result = {
    **manifest,
    "max_step": max_step,
    "temperature_last_half_K": temperature,
    "pressure_last_half_kbar": pressure,
    "nearest_neighbor_A": nearest,
    "phase_status": phase.get("status"),
    "checks": checks,
    "status": "confirmation_passed" if all(checks.values()) else "confirmation_failed",
}
(run / "phase_confirmation.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
if result["status"] != "confirmation_passed":
    raise SystemExit(2)
PY
    then
        :
    else
        gate_failed=1
    fi
done

(
    cd "$root"
    find T0975_liquid_preequil500_round3_preservevel \
        T1050_liquid_preequil500_round3_preservevel -type f -print0 \
        | sort -z | xargs -0 sha256sum
) > "$root/LIQUID_PREEQUIL_ROUND3_SHA256SUMS"

if ((gate_failed != 0)); then
    printf '%s liquid_round3_gate_failed\n' "$(date -Iseconds)" \
        | tee -a "$log" "$failed_file"
    exit 2
fi

date -Iseconds > "$done_file"
printf '%s liquid_round3_verified\n' "$(date -Iseconds)" | tee -a "$log"
