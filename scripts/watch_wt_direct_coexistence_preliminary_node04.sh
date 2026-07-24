#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/scratch/xzh/OFDFT-Hessian}
root=${RUN_ROOT:-$repo/runs/al/method_paper_wt/direct_coexist_prelim_T1004_v1_node04}
config=${WT_CONFIG:-$repo/config/abacus_wt_ti_node04_cpu12.json}
nominal_temperature=${NOMINAL_TEMPERATURE:-1003.840218}
low_temperature=${LOW_TEMPERATURE:-953.840218}
high_temperature=${HIGH_TEMPERATURE:-1053.840218}
log=$root/pipeline.log
done_file=$root/pipeline.done
failed_file=$root/pipeline.failed

mkdir -p "$root"

mark_failed() {
    local rc=$?
    trap - ERR
    if [[ ! -e $done_file ]]; then
        printf '%s failed exit_code=%s\n' "$(date -Iseconds)" "$rc" \
            | tee -a "$log" "$failed_file"
    fi
    exit "$rc"
}
trap mark_failed ERR

if [[ -e $done_file ]]; then
    printf '%s already_complete file=%s\n' "$(date -Iseconds)" "$done_file"
    exit 0
fi
rm -f "$failed_file"

plan=$root/validation_plan.json
if [[ ! -e $plan ]]; then
    printf 'missing preliminary validation plan: %s\n' "$plan" >&2
    exit 2
fi
split=$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["geometry"]["split_fraction"])' \
    "$plan")
printf '%s starting nominal=%s low=%s high=%s split=%s\n' \
    "$(date -Iseconds)" "$nominal_temperature" "$low_temperature" \
    "$high_temperature" "$split" | tee "$log"

wait_for_existing_run() {
    local run=$1
    local expected_steps=$2
    local launch_log=$run/launch_node01.log
    while ! grep -q 'finished_wt' "$launch_log" 2>/dev/null; do
        local observed=0
        if [[ -s $run/run.stdout ]]; then
            observed=$(awk '/STEP OF MOLECULAR DYNAMICS:/ {step=$NF} END {print step+0}' \
                "$run/run.stdout")
        fi
        printf '%s waiting run=%s observed_step=%s expected=%s\n' \
            "$(date -Iseconds)" "$run" "$observed" "$expected_steps" \
            | tee -a "$log"
        if (( observed >= expected_steps )); then
            sleep 15
            break
        fi
        if ! pgrep -f "$run" >/dev/null 2>&1; then
            printf 'existing run stopped before completion: %s (step %s/%s)\n' \
                "$run" "$observed" "$expected_steps" >&2
            return 2
        fi
        sleep 60
    done
}

run_and_analyze() {
    local run=$1
    local expected_steps=$2
    local ranks=$3
    local cpus=$4
    local require_two_phase=$5
    local analysis=$run/two_phase_analysis.json

    if find "$run" -maxdepth 1 -type d -name 'OUT.*' | grep -q .; then
        wait_for_existing_run "$run" "$expected_steps"
    else
        bash "$repo/scripts/launch_wt_single_node01.sh" "$run" "$ranks" "$cpus" \
            >> "$log" 2>&1
    fi

    env PYTHONPATH="$repo" OMP_NUM_THREADS=1 python3 \
        "$repo/scripts/analyze_two_phase_run.py" "$run" --split "$split" \
        --out "$analysis" >> "$log" 2>&1
    python3 - "$analysis" "$expected_steps" "$require_two_phase" <<'PY' \
        >> "$log"
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
expected = int(sys.argv[2])
require_two_phase = sys.argv[3] == "yes"
max_step = int(report.get("md", {}).get("max_step", -1))
nearest = float(report.get("trajectory", {}).get("nearest_neighbor_A", 0.0))
if max_step < expected:
    raise SystemExit(f"trajectory incomplete: {max_step} < {expected}")
if nearest <= 2.0:
    raise SystemExit(f"nearest-neighbor gate failed: {nearest}")
if require_two_phase and report.get("status") != "two_phase_verified":
    raise SystemExit(f"two-phase gate failed: {report.get('status')}")
print(json.dumps({
    "status": report.get("status"),
    "max_step": max_step,
    "nearest_neighbor_A": nearest,
    "interface_migration": report.get("interface_migration_indicators", {}),
}, sort_keys=True))
PY
    printf '%s stage_verified run=%s analysis=%s\n' \
        "$(date -Iseconds)" "$run" "$analysis" | tee -a "$log"
}

hot=$root/template_hot_T1600_fixsolid
run_and_analyze "$hot" 500 16 0-15 yes

cool=$root/template_cool_Tm_fixsolid
if [[ ! -e $cool/metadata.json ]]; then
    env PYTHONPATH="$repo" python3 -m mpn_melting.cli make-restart \
        --element Al --out "$cool" --abacus-config "$config" \
        --source "$hot" --region-source "$hot" --fix-region solid_seed \
        --temperature "$nominal_temperature" --steps 500 --ensemble nvt \
        --thermostat csvr --csvr-tau 10 --dumpfreq 5 --restartfreq 100 \
        --seed 2026073002 >> "$log" 2>&1
fi
run_and_analyze "$cool" 500 16 0-15 yes

tiled=$root/tiled_Al1728_Tm_fixsolid
if [[ ! -e $tiled/metadata.json ]]; then
    env PYTHONPATH="$repo" python3 -m mpn_melting.cli make-tiled-coexist \
        --element Al --out "$tiled" --abacus-config "$config" \
        --template "$cool" --region-source "$cool" --repeat 3 3 3 \
        --split-axis 2 --split "$split" --fix-region solid_seed \
        --temperature "$nominal_temperature" --steps 200 --ensemble nvt \
        --thermostat csvr --csvr-tau 10 --dumpfreq 5 --restartfreq 100 \
        --seed 2026073003 >> "$log" 2>&1
fi
run_and_analyze "$tiled" 200 64 0-71 yes

released=$root/released_Al1728_Tm_thermalize
if [[ ! -e $released/metadata.json ]]; then
    env PYTHONPATH="$repo" python3 -m mpn_melting.cli make-restart \
        --element Al --out "$released" --abacus-config "$config" \
        --source "$tiled" --region-source "$tiled" --discard-velocities \
        --temperature "$nominal_temperature" --steps 300 --ensemble nvt \
        --thermostat csvr --csvr-tau 5 --dumpfreq 5 --restartfreq 100 \
        --seed 2026073004 >> "$log" 2>&1
fi
run_and_analyze "$released" 300 64 0-71 yes

prepare_branch() {
    local label=$1
    local temperature=$2
    local seed=$3
    local run=$root/production_$label
    if [[ ! -e $run/metadata.json ]]; then
        env PYTHONPATH="$repo" python3 -m mpn_melting.cli make-restart \
            --element Al --out "$run" --abacus-config "$config" \
            --source "$released" --region-source "$released" \
            --discard-velocities --temperature "$temperature" --steps 500 \
            --ensemble nvt --thermostat csvr --csvr-tau 20 --dumpfreq 5 \
            --restartfreq 100 --seed "$seed" >> "$log" 2>&1
    fi
    printf '%s\n' "$run"
}

low_run=$(prepare_branch low "$low_temperature" 2026073011)
nominal_run=$(prepare_branch nominal "$nominal_temperature" 2026073012)
high_run=$(prepare_branch high "$high_temperature" 2026073013)
run_and_analyze "$low_run" 500 64 0-71 no
run_and_analyze "$nominal_run" 500 64 0-71 yes
run_and_analyze "$high_run" 500 64 0-71 no

summary=$root/preliminary_validation_summary.json
python3 - "$low_run/two_phase_analysis.json" \
    "$nominal_run/two_phase_analysis.json" \
    "$high_run/two_phase_analysis.json" "$summary" \
    "$low_temperature" "$nominal_temperature" "$high_temperature" <<'PY'
import json
import sys
from pathlib import Path

paths = [Path(value) for value in sys.argv[1:4]]
out = Path(sys.argv[4])
temperatures = [float(value) for value in sys.argv[5:8]]
labels = ["low", "nominal", "high"]
runs = {}
for label, path, temperature in zip(labels, paths, temperatures):
    report = json.load(open(path, encoding="utf-8"))
    migration = report.get("interface_migration_indicators", {})
    runs[label] = {
        "analysis": str(path.resolve()),
        "target_temperature_k": temperature,
        "phase_status": report.get("status"),
        "max_step": report.get("md", {}).get("max_step"),
        "temperature_last_100_steps_k": report.get("md", {}).get(
            "temperature_last_100_steps_K", {}
        ),
        "nearest_neighbor_A": report.get("trajectory", {}).get(
            "nearest_neighbor_A"
        ),
        "integrated_ordered_fraction_change": migration.get(
            "integrated_ordered_fraction_change"
        ),
    }

low_change = runs["low"]["integrated_ordered_fraction_change"]
nominal_change = runs["nominal"]["integrated_ordered_fraction_change"]
high_change = runs["high"]["integrated_ordered_fraction_change"]
quality = all(
    row["max_step"] is not None
    and int(row["max_step"]) >= 500
    and row["nearest_neighbor_A"] is not None
    and float(row["nearest_neighbor_A"]) > 2.0
    and row["integrated_ordered_fraction_change"] is not None
    for row in runs.values()
)
directions = (
    low_change is not None
    and float(low_change) >= 0.02
    and high_change is not None
    and float(high_change) <= -0.02
)
nominal_two_phase = runs["nominal"]["phase_status"] == "two_phase_verified"
nominal_stable = (
    nominal_change is not None and abs(float(nominal_change)) <= 0.05
)
if quality and directions and nominal_two_phase and nominal_stable:
    status = "preliminary_center_validated"
elif quality and directions:
    status = "preliminary_bracket_verified_needs_refinement"
else:
    status = "validation_failed"
result = {
    "schema": "wt-direct-coexistence-preliminary-v1",
    "status": status,
    "preliminary_center_temperature_k": temperatures[1],
    "temperature_bracket_k": [temperatures[0], temperatures[2]],
    "checks": {
        "all_run_quality_verified": quality,
        "low_solid_growth_and_high_liquid_growth": directions,
        "nominal_two_phase_verified": nominal_two_phase,
        "nominal_interface_nearly_stable": nominal_stable,
    },
    "runs": runs,
}
out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
PY

date -Iseconds > "$done_file"
printf '%s preliminary_validation_complete summary=%s\n' \
    "$(date -Iseconds)" "$summary" | tee -a "$log"
