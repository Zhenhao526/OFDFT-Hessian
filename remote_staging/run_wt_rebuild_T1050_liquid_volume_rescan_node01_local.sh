#!/usr/bin/env bash
set -euo pipefail

workspace=${WORKSPACE:-/home/shenwei01/WT_Al_melting_workspace_20260724}
repo=${REPO:-$workspace/repository}
runtime=${RUNTIME:-/home/shenwei01/wt_melting_runtime_20260724}
root=${RUN_ROOT:-$workspace/runs/step4_rebuild}
scan=$root/T1050_liquid_volume_rescan_from_round3
config=$repo/config/abacus_wt_node01_local_cpu18.json
binary=$runtime/build-abacus-wt-cpu/source/abacus_pw_para
mpirun=$runtime/conda_prefix/bin/mpirun
source_root=$root/T1050_liquid_preequil500_round3_preservevel
log=$workspace/audit/step4_T1050_liquid_volume_rescan.log
done_file=$workspace/audit/step4_T1050_liquid_volume_rescan.done
failed_file=$workspace/audit/step4_T1050_liquid_volume_rescan.failed

mkdir -p "$scan" "$workspace/audit"
rm -f "$done_file" "$failed_file"
printf '%s T1050_liquid_volume_rescan_started\n' "$(date -Iseconds)" > "$log"

mark_failed() {
    local rc=$?
    trap - ERR
    if [[ ! -e $done_file ]]; then
        printf '%s T1050_liquid_volume_rescan_failed exit_code=%s\n' \
            "$(date -Iseconds)" "$rc" | tee -a "$log" "$failed_file"
    fi
    exit "$rc"
}
trap mark_failed ERR

source=$(find "$source_root" -path '*/MD_dump' -type f | head -1)
test -s "$source"

prepare_point() {
    local label=$1
    local volume=$2
    local seed=$3
    local out=$scan/$label
    if [[ ! -e $out/phase_continuation_manifest.json ]]; then
        env PYTHONPATH="$repo" python3 \
            "$repo/scripts/prepare_wt_phase_continuation.py" \
            --source "$source" --out "$out" --phase liquid \
            --volume "$volume" --temperature 1050 \
            --steps 500 --csvr-tau 2 --seed "$seed" --config "$config" \
            >> "$log" 2>&1
    fi
}

prepare_point vpa_19p120 19.120 2026080120
prepare_point vpa_19p200 19.200 2026080200

points=("$scan/vpa_19p120" "$scan/vpa_19p200")
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

for point in "${points[@]}"; do
    env PYTHONPATH="$repo" python3 \
        "$repo/scripts/analyze_wt_phase_confirmation.py" "$point" \
        --expected liquid --temperature 1050 --pressure 0 --steps 500 \
        --temperature-tolerance 20 --pressure-tolerance 10 \
        --out "$point/phase_confirmation.json" >> "$log" 2>&1
done

env PYTHONPATH="$repo" python3 - "$source_root" "$scan" <<'PY'
import json
import sys
from pathlib import Path

source_root = Path(sys.argv[1]).resolve()
scan = Path(sys.argv[2]).resolve()
base = json.loads(
    (source_root / "phase_confirmation.json").read_text(encoding="utf-8")
)
rows = [
    {
        "label": "vpa_19p043",
        "volume_per_atom_A3": 19.043,
        "source": str(source_root),
        "confirmation": base,
    }
]
for label, volume in (("vpa_19p120", 19.120), ("vpa_19p200", 19.200)):
    report = json.loads(
        (scan / label / "phase_confirmation.json").read_text(encoding="utf-8")
    )
    rows.append(
        {
            "label": label,
            "volume_per_atom_A3": volume,
            "source": str(scan / label),
            "confirmation": report,
        }
    )

valid = []
for row in rows:
    report = row["confirmation"]
    physical_checks = {
        key: value
        for key, value in report["checks"].items()
        if key != "pressure_mean_within_tolerance"
    }
    if all(physical_checks.values()):
        valid.append(
            {
                **row,
                "pressure_mean_kbar": report["pressure_last_half_kbar"]["mean"],
                "temperature_mean_K": report["temperature_last_half_K"]["mean"],
                "nearest_neighbor_A": report["nearest_neighbor_A"],
                "phase_status": report["phase_status"],
            }
        )
valid.sort(key=lambda row: row["volume_per_atom_A3"])
bracket = None
for left, right in zip(valid, valid[1:]):
    if left["pressure_mean_kbar"] * right["pressure_mean_kbar"] <= 0.0:
        bracket = (left, right)
        break
if bracket is None:
    raise SystemExit("volume rescan did not bracket zero pressure")
left, right = bracket
v0, v1 = left["volume_per_atom_A3"], right["volume_per_atom_A3"]
p0, p1 = left["pressure_mean_kbar"], right["pressure_mean_kbar"]
fit = v0 - p0 * (v1 - v0) / (p1 - p0)
monotonic = all(
    left["pressure_mean_kbar"] > right["pressure_mean_kbar"]
    for left, right in zip(valid, valid[1:])
)
checks = {
    "at_least_two_valid_points": len(valid) >= 2,
    "zero_pressure_bracket_found": bracket is not None,
    "pressure_decreases_with_volume": monotonic,
    "fit_inside_bracket": v0 <= fit <= v1,
}
result = {
    "schema": "wt-T1050-liquid-volume-rescan-v1",
    "target_kedf": "wt",
    "target_temperature_K": 1050.0,
    "rows": rows,
    "valid_rows": valid,
    "zero_pressure_bracket_A3_per_atom": [v0, v1],
    "linear_zero_pressure_volume_A3_per_atom": fit,
    "checks": checks,
    "status": "zero_pressure_volume_verified" if all(checks.values()) else "volume_gate_failed",
}
(scan / "volume_scan_summary.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
if result["status"] != "zero_pressure_volume_verified":
    raise SystemExit("volume scan checks did not all pass")
PY

(
    cd "$scan"
    find . -type f -print0 | sort -z | xargs -0 sha256sum
) > "$scan/SHA256SUMS"
date -Iseconds > "$done_file"
printf '%s T1050_liquid_volume_rescan_verified\n' "$(date -Iseconds)" \
    | tee -a "$log"
