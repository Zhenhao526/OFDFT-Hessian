#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/scratch/xzh/OFDFT-Hessian}
pipeline_root=${PIPELINE_ROOT:-$repo/runs/al/free_energy_wt/melting_pipeline_pairv2_recovery_T1100_v2_node01}
analysis_root=${ANALYSIS_ROOT:-$repo/runs/al/free_energy_wt/fusion_enthalpy_sampling/T0900_T0975_T1050_adaptive_T1100_pairv2_steps3000_v2_node01}
combined=${COMBINED:-$repo/runs/al/free_energy_wt/melting_free_energy_T0900_pairv2_v2.json}

main_done=$pipeline_root/pipeline.done
main_failed=$pipeline_root/pipeline.failed
recovery_done=$analysis_root/enthalpy_recovery.done
recovery_failed=$analysis_root/enthalpy_recovery.failed
log=$analysis_root/enthalpy_recovery.log

mkdir -p "$analysis_root/critical_extensions"

mark_failed() {
    local rc=$?
    trap - ERR
    if [[ ! -e $recovery_done ]]; then
        printf '%s recovery_failed exit_code=%s\n' "$(date -Iseconds)" "$rc" \
            | tee -a "$log" "$recovery_failed"
    fi
    exit "$rc"
}
trap mark_failed ERR

if [[ -e $main_done ]]; then
    printf '%s main_pipeline_already_complete\n' "$(date -Iseconds)" > "$log"
    exit 0
fi
rm -f "$recovery_failed"
printf '%s waiting_for_main_pipeline\n' "$(date -Iseconds)" > "$log"

while [[ ! -e $main_done && ! -e $main_failed ]]; do
    sleep 60
done
if [[ -e $main_done ]]; then
    printf '%s main_pipeline_completed_without_recovery\n' "$(date -Iseconds)" \
        | tee -a "$log"
    exit 0
fi

convergence=$analysis_root/discard_convergence_summary_extension_round1.json
if [[ ! -e $convergence ]]; then
    printf '%s refusing_recovery reason=missing_round1_convergence\n' \
        "$(date -Iseconds)" | tee -a "$log"
    exit 2
fi

declare -A roots
roots[900]=$repo/runs/al/free_energy_wt/fusion_enthalpy_sampling/T0900_continuation_steps3000_v2_node01
roots[975]=$analysis_root/critical_extensions/T0975_steps3000_round1
roots[1050]=$analysis_root/critical_extensions/T1050_steps3000_round1
roots[1100]=$analysis_root/critical_extensions/T1100_steps3000_round1
selected_series=$analysis_root/enthalpy_extension_d50.json

run_confirmation() {
    local root=$1
    if [[ -e $root/confirmation_summary.json ]]; then
        python3 - "$root/confirmation_summary.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "all_confirmations_passed":
    raise SystemExit("existing extension confirmation did not pass")
if not report.get("phase_results") or any(
    row.get("status") != "confirmation_passed"
    or not row.get("checks")
    or not all(row["checks"].values())
    for row in report["phase_results"]
):
    raise SystemExit("existing extension phase gates did not all pass")
PY
        printf '%s confirmation_already_verified root=%s\n' \
            "$(date -Iseconds)" "$root" | tee -a "$log"
        return
    fi
    printf '%s launching_confirmation root=%s\n' "$(date -Iseconds)" "$root" \
        | tee -a "$log"
    bash "$repo/scripts/launch_prepared_zero_pressure_pair_node01.sh" "$root" \
        >> "$log" 2>&1
    printf '%s confirmation_verified root=%s\n' "$(date -Iseconds)" "$root" \
        | tee -a "$log"
}

convergence_verified() {
    python3 - "$1" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "verified" or report.get("critical_temperatures"):
    raise SystemExit(1)
if not report.get("checks") or not all(report["checks"].values()):
    raise SystemExit(1)
PY
}

select_extendable_temperatures() {
    python3 - "$1" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
allowed_reasons = {"report_gate", "discard_sensitivity", "block_standard_error"}
allowed_failed_checks = {
    "half_drift_within_tolerance",
    "solid_liquid_temperature_means_match",
}
critical = []
for point in report.get("points", []):
    if point.get("status") == "verified":
        continue
    temperature = int(round(float(point["temperature_k"])))
    reasons = set(point.get("extension_reasons", []))
    if not reasons or not reasons <= allowed_reasons:
        raise SystemExit(
            f"non-statistical extension reason at {temperature} K: {sorted(reasons)}"
        )
    for row in point.get("reports", []):
        failed = set(row.get("failed_checks", []))
        if not failed <= allowed_failed_checks:
            raise SystemExit(
                f"physical gate failed at {temperature} K: {sorted(failed)}"
            )
    critical.append(temperature)
if not critical:
    raise SystemExit("no statistically extendable temperatures")
for temperature in sorted(critical):
    print(temperature)
PY
}

for round in 2 3; do
    if convergence_verified "$convergence"; then
        break
    fi
    critical_output=$(select_extendable_temperatures "$convergence")
    mapfile -t critical_temperatures <<< "$critical_output"
    printf '%s preparing_round=%s temperatures=%s\n' \
        "$(date -Iseconds)" "$round" "${critical_temperatures[*]}" | tee -a "$log"

    for temperature in "${critical_temperatures[@]}"; do
        source_root=${roots[$temperature]:-}
        if [[ -z $source_root || ! -e $source_root/confirmation_summary.json ]]; then
            printf '%s refusing_extension reason=missing_verified_source temperature=%s source=%s\n' \
                "$(date -Iseconds)" "$temperature" "$source_root" | tee -a "$log"
            exit 2
        fi
        extension_run=$(printf '%s/critical_extensions/T%04d_steps3000_round%d' \
            "$analysis_root" "$temperature" "$round")
        if [[ ! -e $extension_run/confirmation_manifest.json ]]; then
            env PYTHONPATH="$repo" python3 "$repo/scripts/prepare_wt_enthalpy_extension.py" \
                --source "$source_root" --out "$extension_run" --steps 3000 \
                --csvr-tau 5 --seed "$((2026073601 + round * 1000 + temperature))" \
                --config "$repo/config/abacus_wt_ti_node04_cpu12.json" >> "$log" 2>&1
        fi
        run_confirmation "$extension_run"
        roots[$temperature]=$extension_run
    done

    manifest=$analysis_root/manifest_extension_round${round}.json
    env PYTHONPATH="$repo" python3 "$repo/scripts/build_wt_fusion_enthalpy_manifest.py" \
        "${roots[900]}" "${roots[975]}" "${roots[1050]}" "${roots[1100]}" \
        --out "$manifest" >> "$log" 2>&1

    reports=()
    for suffix in d25 d50 d75; do
        case $suffix in
            d25) discard=0.25 ;;
            d50) discard=0.50 ;;
            d75) discard=0.75 ;;
        esac
        report=$analysis_root/enthalpy_extension_round${round}_${suffix}.json
        env PYTHONPATH="$repo" python3 "$repo/scripts/analyze_wt_fusion_enthalpy_series.py" \
            "$manifest" --out "$report" --discard-fraction "$discard" \
            --temperature-tolerance 20 --pressure-tolerance 2.5 \
            --maximum-half-drift 5 >> "$log" 2>&1
        reports+=("$report")
    done
    convergence=$analysis_root/discard_convergence_summary_extension_round${round}.json
    env PYTHONPATH="$repo" python3 "$repo/scripts/summarize_wt_enthalpy_convergence.py" \
        "${reports[@]}" --out "$convergence" >> "$log" 2>&1
    selected_series=$analysis_root/enthalpy_extension_round${round}_d50.json
    printf '%s round_analyzed round=%s convergence=%s\n' \
        "$(date -Iseconds)" "$round" "$convergence" | tee -a "$log"
done

if ! convergence_verified "$convergence"; then
    printf '%s refusing_melting_solve reason=enthalpy_still_unverified convergence=%s\n' \
        "$(date -Iseconds)" "$convergence" | tee -a "$log"
    exit 2
fi

melting=$analysis_root/gibbs_helmholtz_melting.json
env PYTHONPATH="$repo" python3 "$repo/scripts/solve_wt_melting_gibbs_helmholtz.py" \
    --combination "$combined" --enthalpy-series "$selected_series" \
    --enthalpy-convergence "$convergence" --out "$melting" >> "$log" 2>&1
python3 - "$melting" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "verified":
    raise SystemExit("Gibbs-Helmholtz result is not verified")
if not report.get("checks") or not all(report["checks"].values()):
    raise SystemExit("Gibbs-Helmholtz checks did not all pass")
PY

printf '%s gibbs_helmholtz_verified output=%s\n' \
    "$(date -Iseconds)" "$melting" | tee -a "$log"
date -Iseconds > "$main_done"
rm -f "$main_failed"
date -Iseconds > "$recovery_done"
