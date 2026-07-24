#!/usr/bin/env bash
set -euo pipefail

repo=/scratch/xzh/OFDFT-Hessian
run_root=${1:?usage: watch_wt_enthalpy_analysis.sh RUN_ROOT [ANALYSIS_ROOT]}
analysis_root=${2:-$run_root/enthalpy_analysis}
launcher_log=$run_root/run_zero_pressure_pair_node01.log
watch_log=$analysis_root/watch_enthalpy_analysis.log

mkdir -p "$analysis_root"
printf '%s waiting_for_zero_pressure_pair launcher_log=%s\n' \
    "$(date -Iseconds)" "$launcher_log" > "$watch_log"
while ! grep -q 'zero_pressure_pair_complete' "$launcher_log" 2>/dev/null; do
    if grep -Eq 'failed point=|zero_pressure_pair_gate_failed' "$launcher_log" 2>/dev/null; then
        printf '%s upstream_failed\n' "$(date -Iseconds)" | tee -a "$watch_log"
        exit 2
    fi
    sleep 60
done

cd "$repo"
manifest=$analysis_root/manifest.json
env PYTHONPATH="$repo" python3 scripts/build_wt_fusion_enthalpy_manifest.py \
    "$run_root" --out "$manifest" >> "$watch_log" 2>&1
reports=()
for suffix in d25 d50 d75; do
    case $suffix in
        d25) discard=0.25 ;;
        d50) discard=0.50 ;;
        d75) discard=0.75 ;;
    esac
    report=$analysis_root/enthalpy_${suffix}.json
    env PYTHONPATH="$repo" python3 scripts/analyze_wt_fusion_enthalpy_series.py \
        "$manifest" --out "$report" --discard-fraction "$discard" \
        --temperature-tolerance 20 --pressure-tolerance 2.5 \
        --maximum-half-drift 5 >> "$watch_log" 2>&1
    reports+=("$report")
done
env PYTHONPATH="$repo" python3 scripts/summarize_wt_enthalpy_convergence.py \
    "${reports[@]}" --out "$analysis_root/discard_convergence_summary.json" \
    >> "$watch_log" 2>&1
printf '%s enthalpy_analysis_complete\n' "$(date -Iseconds)" | tee -a "$watch_log"
touch "$analysis_root/enthalpy_analysis.done"
