#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/scratch/xzh/OFDFT-Hessian}
enthalpy_root=$repo/runs/al/free_energy_wt/fusion_enthalpy_sampling/T0900_T0975_T1050_adaptive_T1100_pairv2_steps3000_v2_node01
melting=${MELTING:-$enthalpy_root/gibbs_helmholtz_melting.json}
pipeline_done=${PIPELINE_DONE:-$repo/runs/al/free_energy_wt/melting_pipeline_pairv2_v1_node01/pipeline.done}
root=${VALIDATION_ROOT:-$repo/runs/al/free_energy_wt/direct_coexistence_validation/pairv2_v1_node01}
config=$repo/config/abacus_wt_ti_node04_cpu12.json
log=$root/validation_pipeline.log
done_file=$root/validation.done
failed_file=$root/validation.failed

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
printf '%s waiting_for_verified_melting file=%s\n' \
    "$(date -Iseconds)" "$melting" > "$log"
while [[ ! -e $pipeline_done || ! -e $melting ]]; do
    if [[ -e ${pipeline_done%.done}.failed ]]; then
        printf '%s refusing_validation reason=free_energy_pipeline_failed\n' \
            "$(date -Iseconds)" | tee -a "$log"
        exit 2
    fi
    sleep 60
done

read -r low_temperature nominal_temperature high_temperature < <(
    python3 - "$melting" <<'PY'
import json
import math
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "verified" or not report.get("checks") or not all(report["checks"].values()):
    raise SystemExit("melting result is not verified")
temperature = float(report["melting_temperature_k"])
half_width = report.get("statistical_interval", {}).get("half_width_k")
offset = max(20.0, float(half_width) if half_width is not None else 20.0)
offset = min(offset, 50.0)
print(f"{temperature - offset:.6f} {temperature:.6f} {temperature + offset:.6f}")
PY
)
printf '%s validation_temperatures low=%s nominal=%s high=%s\n' \
    "$(date -Iseconds)" "$low_temperature" "$nominal_temperature" \
    "$high_temperature" | tee -a "$log"

if [[ ! -e $root/validation_plan.json ]]; then
    env PYTHONPATH="$repo" python3 "$repo/scripts/prepare_wt_coexistence_validation.py" \
        --melting "$melting" --out "$root" --config "$config" \
        --hot-temperature 1600 --steps 500 --csvr-tau 10 --seed 2026072501 \
        --mpi-ranks 16 >> "$log" 2>&1
fi
split=$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["geometry"]["split_fraction"])' \
    "$root/validation_plan.json")

run_and_analyze() {
    local run=$1
    local expected_steps=$2
    local ranks=$3
    local cpus=$4
    local require_two_phase=$5
    local analysis=$run/two_phase_analysis.json
    if [[ ! -e $analysis ]]; then
        if ! find "$run" -maxdepth 1 -type d -name 'OUT.*' | grep -q .; then
            bash "$repo/scripts/launch_wt_single_node01.sh" "$run" "$ranks" "$cpus" \
                >> "$log" 2>&1
        fi
        env PYTHONPATH="$repo" python3 "$repo/scripts/analyze_two_phase_run.py" \
            "$run" --split "$split" --out "$analysis" >> "$log" 2>&1
    fi
    python3 - "$analysis" "$expected_steps" "$require_two_phase" <<'PY' >> "$log"
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
expected = int(sys.argv[2])
require_two_phase = sys.argv[3] == "yes"
if int(report.get("md", {}).get("max_step", -1)) < expected:
    raise SystemExit(f"trajectory is incomplete: {report.get('md', {}).get('max_step')} < {expected}")
trajectory = report.get("trajectory", {})
nearest = float(trajectory.get("nearest_neighbor_A", 0.0))
minimum_nearest = float(trajectory.get("minimum_nearest_neighbor_A", nearest))
if nearest <= 2.0:
    raise SystemExit("trajectory failed the nearest-neighbor gate")
if minimum_nearest <= 2.0:
    raise SystemExit("trajectory failed the all-frames nearest-neighbor gate")
if require_two_phase and report.get("status") != "two_phase_verified":
    raise SystemExit(f"two-phase gate failed: {report.get('status')}")
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
        --seed 2026072502 >> "$log" 2>&1
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
        --seed 2026072503 >> "$log" 2>&1
fi
run_and_analyze "$tiled" 200 64 0-71 yes

released=$root/released_Al1728_Tm_thermalize
if [[ ! -e $released/metadata.json ]]; then
    env PYTHONPATH="$repo" python3 -m mpn_melting.cli make-restart \
        --element Al --out "$released" --abacus-config "$config" \
        --source "$tiled" --region-source "$tiled" --discard-velocities \
        --temperature "$nominal_temperature" --steps 300 --ensemble nvt \
        --thermostat csvr --csvr-tau 5 --dumpfreq 5 --restartfreq 100 \
        --seed 2026072504 >> "$log" 2>&1
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
            --source "$released" --region-source "$released" --discard-velocities \
            --temperature "$temperature" --steps 500 --ensemble nvt \
            --thermostat csvr --csvr-tau 20 --dumpfreq 5 --restartfreq 100 \
            --seed "$seed" >> "$log" 2>&1
    fi
    printf '%s\n' "$run"
}

low_run=$(prepare_branch low "$low_temperature" 2026072511)
nominal_run=$(prepare_branch nominal "$nominal_temperature" 2026072512)
high_run=$(prepare_branch high "$high_temperature" 2026072513)
run_and_analyze "$low_run" 500 64 0-71 no
run_and_analyze "$nominal_run" 500 64 0-71 yes
run_and_analyze "$high_run" 500 64 0-71 no

summary=$root/validation_summary.json
env PYTHONPATH="$repo" python3 "$repo/scripts/summarize_wt_coexistence_validation.py" \
    --melting "$melting" --low "$low_run/two_phase_analysis.json" \
    --nominal "$nominal_run/two_phase_analysis.json" \
    --high "$high_run/two_phase_analysis.json" \
    --low-temperature "$low_temperature" \
    --nominal-temperature "$nominal_temperature" \
    --high-temperature "$high_temperature" --requested-steps 500 \
    --out "$summary" >> "$log" 2>&1
python3 - "$summary" <<'PY' >> "$log"
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "verified" or not report.get("checks") or not all(report["checks"].values()):
    raise SystemExit("independent coexistence validation failed")
PY
printf '%s independent_validation_verified summary=%s\n' \
    "$(date -Iseconds)" "$summary" | tee -a "$log"
date -Iseconds > "$done_file"
