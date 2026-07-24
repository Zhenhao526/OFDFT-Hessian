#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/scratch/xzh/OFDFT-Hessian}
liquid_root=${LIQUID_ROOT:-$repo/runs/al/free_energy_wt/ti_windows/phase_specific_T0900_zeroP_liquid_pairv2_lambda5_steps1000_v3_node01}
solid_root=${SOLID_ROOT:-$repo/runs/al/free_energy_wt/ti_windows/phase_specific_T0900_zeroP_lambda5_steps1000_v1_node01/solid}
combined=${COMBINED:-$repo/runs/al/free_energy_wt/melting_free_energy_T0900_pairv2_v2.json}
pipeline_root=${PIPELINE_ROOT:-$repo/runs/al/free_energy_wt/melting_pipeline_pairv2_v1_node01}
extension_root=$repo/runs/al/free_energy_wt/ti_windows/phase_specific_T0900_zeroP_liquid_pairv2_critical_extension_round1_steps1000_node01
scan_root=$repo/runs/al/free_energy_wt/multitemp_volume_scan
scan_975=${SCAN_975:-$scan_root/T0975_prepared_v1_node01}
scan_1050=${SCAN_1050:-$scan_root/T1050_prepared_v1_node01}
scan_1100=${SCAN_1100:-$scan_root/T1100_prepared_v1_node01}
enthalpy_base=$repo/runs/al/free_energy_wt/fusion_enthalpy_sampling
enthalpy_900=$enthalpy_base/T0900_continuation_steps3000_v2_node01
enthalpy_975=$enthalpy_base/T0975_zeroP_steps3000_pairv2_v1_node01
enthalpy_1050=$enthalpy_base/T1050_zeroP_steps3000_pairv2_v1_node01
enthalpy_1100=$enthalpy_base/T1100_zeroP_steps3000_pairv2_v1_node01
analysis_root=$enthalpy_base/T0900_T0975_T1050_adaptive_T1100_pairv2_steps3000_v2_node01
log=$pipeline_root/pipeline.log
done_file=$pipeline_root/pipeline.done
failed_file=$pipeline_root/pipeline.failed

mkdir -p "$pipeline_root"

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
printf '%s waiting_for_anchor liquid_root=%s\n' "$(date -Iseconds)" "$liquid_root" > "$log"

while [[ ! -e $liquid_root/post_ti_gate.done \
    && ! -e $liquid_root/post_ti_gate.failed ]]; do
    sleep 60
done

combine_anchor() {
    local liquid_analysis=$1
    local liquid_summary=$2
    env PYTHONPATH="$repo" python3 "$repo/scripts/combine_wt_melting_free_energy.py" \
        --analytic "$repo/runs/al/free_energy_wt/analytic_reference_T0900_p100_sigma1p36_v2.json" \
        --solid-anchor "$repo/runs/al/free_energy_wt/absolute_anchor/solid_einstein_pair_T0900_v1/lambda9_steps20000_v1_node01/ti_analysis.json" \
        --liquid-anchor "$repo/runs/al/free_energy_wt/absolute_anchor/liquid_suf_pair_T0900_p100_sigma1p36_lambda9_steps20000_v2_node01/ti_analysis_strict_v2.json" \
        --anchor-audit "$repo/runs/al/free_energy_wt/reference_anchor_audit_T0900_p100sigma1p36_pairv2_v2.json" \
        --solid-wt "$solid_root/ti_prod_d50.json" \
        --liquid-wt "$liquid_analysis" \
        --solid-wt-convergence "$solid_root/discard_convergence_summary_v2.json" \
        --liquid-wt-convergence "$liquid_summary" \
        --zero-pressure-summary "$repo/runs/al/free_energy_wt/zeroP_confirmation_T0900_combined_v17p940_v2_node01/confirmation_summary.json" \
        --out "$combined" >> "$log" 2>&1
}

anchor_is_verified() {
    [[ -e $combined ]] || return 1
    python3 - "$combined" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "anchor_temperature_free_energy_verified":
    raise SystemExit(1)
checks = report.get("checks", {})
if not checks or not all(checks.values()):
    raise SystemExit(1)
PY
}

liquid_summary=$liquid_root/discard_convergence_summary.json
if anchor_is_verified; then
    printf '%s reusing_verified_anchor file=%s\n' \
        "$(date -Iseconds)" "$combined" | tee -a "$log"
elif [[ -e $liquid_root/post_ti_gate.failed ]]; then
    if [[ ! -e $liquid_summary ]]; then
        printf '%s refusing_downstream reason=liquid_ti_summary_missing\n' \
            "$(date -Iseconds)" | tee -a "$log"
        exit 2
    fi
    summary_status=$(python3 -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
        "$liquid_summary")
    if [[ $summary_status == verified ]]; then
        printf '%s recovering_verified_liquid_summary_after_watcher_failure\n' \
            "$(date -Iseconds)" | tee -a "$log"
        combine_anchor "$liquid_root/ti_prod_d50.json" "$liquid_summary"
    else
        mapfile -t critical_labels < <(
            python3 - "$liquid_summary" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("lambda_refinement_required"):
    raise SystemExit("lambda refinement is required; a continuation cannot repair it")
critical = report.get("critical_windows", [])
if not critical:
    raise SystemExit("TI convergence failed without identifiable critical windows")
allowed = {
    "liquid_diffusion_slope_gate",
    "temperature_gate",
    "window_block_standard_error",
    "window_half_drift",
    "window_discard_spread",
}
for window in critical:
    reasons = set(window.get("reasons", []))
    if not reasons or not reasons <= allowed:
        raise SystemExit(
            f"non-statistical TI failure for {window.get('label')}: {sorted(reasons)}"
        )
    print(window["label"])
PY
        )
        if ((${#critical_labels[@]} == 0)); then
            printf '%s refusing_downstream reason=no_extendable_liquid_windows\n' \
                "$(date -Iseconds)" | tee -a "$log"
            exit 2
        fi
        printf '%s preparing_liquid_extensions labels=%s\n' \
            "$(date -Iseconds)" "${critical_labels[*]}" | tee -a "$log"
        if [[ ! -e $extension_root/liquid/manifest.json ]]; then
            env PYTHONPATH="$repo" python3 "$repo/scripts/prepare_ti_window_extensions.py" \
                --phase-root "liquid=$liquid_root" --out "$extension_root" \
                --phases liquid --labels "${critical_labels[@]}" --steps 1000 \
                --csvr-tau 10 --seed 2026072401 \
                --config "$repo/config/abacus_wt_ti_node04_cpu12.json" >> "$log" 2>&1
        fi

        complete_extensions=$(python3 - "$extension_root/liquid" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
complete = 0
for window in manifest["windows"]:
    logs = sorted((root / window["label"]).glob("OUT.*/running_md.log"))
    if logs and re.search(r"MPN_TI_COMPONENTS step=999 ", logs[-1].read_text(errors="replace")):
        complete += 1
print(f"{complete}/{len(manifest['windows'])}")
PY
        )
        if [[ $complete_extensions != "${#critical_labels[@]}/${#critical_labels[@]}" ]]; then
            if find "$extension_root" -mindepth 3 -maxdepth 3 -type d -name 'OUT.*' | grep -q .; then
                printf '%s refusing_liquid_extension reason=partial_existing_output progress=%s\n' \
                    "$(date -Iseconds)" "$complete_extensions" | tee -a "$log"
                exit 2
            fi
            bash "$repo/scripts/launch_selected_ti_extensions_node01.sh" "$extension_root" \
                >> "$log" 2>&1
        fi

        overrides=()
        for label in "${critical_labels[@]}"; do
            overrides+=(--window-override "$label=$extension_root/liquid/$label")
        done
        extension_reports=()
        for suffix in d25 d50 d75; do
            case $suffix in
                d25) discard=0.25 ;;
                d50) discard=0.50 ;;
                d75) discard=0.75 ;;
            esac
            report=$extension_root/liquid_ti_prod_${suffix}.json
            env PYTHONPATH="$repo" python3 "$repo/scripts/analyze_wt_pair_ti_production.py" \
                "$liquid_root" --discard-fraction "$discard" \
                --max-quadrature-difference 1 "${overrides[@]}" --out "$report" \
                >> "$log" 2>&1
            extension_reports+=("$report")
        done
        extension_summary=$extension_root/discard_convergence_summary.json
        env PYTHONPATH="$repo" python3 "$repo/scripts/summarize_wt_ti_convergence.py" \
            "${extension_reports[@]}" --out "$extension_summary" >> "$log" 2>&1
        python3 - "$extension_summary" <<'PY' >> "$log"
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "verified":
    raise SystemExit("liquid TI remains unverified after one targeted extension")
if report.get("critical_windows") or report.get("lambda_refinement_required"):
    raise SystemExit("liquid TI extension still has critical windows or lambda refinement")
PY
        combine_anchor "$extension_root/liquid_ti_prod_d50.json" "$extension_summary"
        date -Iseconds > "$extension_root/post_ti_gate_extension.done"
        printf '%s liquid_extensions_verified root=%s\n' \
            "$(date -Iseconds)" "$extension_root" | tee -a "$log"
    fi
elif [[ ! -e $combined ]]; then
    combine_anchor "$liquid_root/ti_prod_d50.json" "$liquid_summary"
fi

env PYTHONPATH="$repo" python3 "$repo/scripts/select_wt_enthalpy_temperature_grid.py" \
    "$combined" --out "$pipeline_root/anchor_grid_decision.json" >> "$log" 2>&1
need_1100=$(python3 - "$pipeline_root/anchor_grid_decision.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
print("yes" if 1100.0 in report["enthalpy_temperature_grid_k"] else "no")
PY
)
printf '%s anchor_and_temperature_grid_verified\n' "$(date -Iseconds)" | tee -a "$log"

run_volume_scan() {
    local root=$1
    if [[ -e $root/solid/nvt_volume_scan_result.json \
        && -e $root/liquid/nvt_volume_scan_result.json ]]; then
        python3 - "$root" <<'PY' >> "$log"
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
for phase in ("solid", "liquid"):
    report = json.loads((root / phase / "nvt_volume_scan_result.json").read_text())
    if report.get("status") != "zero_pressure_volume_verified":
        raise SystemExit(f"existing {phase} volume scan is not verified")
    if not report.get("checks") or not all(report["checks"].values()):
        raise SystemExit(f"existing {phase} volume scan checks failed")
PY
        printf '%s volume_scan_already_verified root=%s\n' \
            "$(date -Iseconds)" "$root" | tee -a "$log"
        return
    fi
    if find "$root" -mindepth 3 -maxdepth 3 -type d -name 'OUT.*' | grep -q .; then
        printf '%s refusing_volume_scan reason=partial_existing_output root=%s\n' \
            "$(date -Iseconds)" "$root" | tee -a "$log"
        return 2
    fi
    printf '%s launching_volume_scan root=%s\n' "$(date -Iseconds)" "$root" \
        | tee -a "$log"
    bash "$repo/scripts/launch_wt_volume_scan_pair_node01.sh" "$root" >> "$log" 2>&1
    printf '%s volume_scan_verified root=%s\n' "$(date -Iseconds)" "$root" \
        | tee -a "$log"
}

prepare_confirmation() {
    local scan=$1
    local root=$2
    local temperature=$3
    local seed=$4
    if [[ -e $root/confirmation_manifest.json ]]; then
        printf '%s confirmation_already_prepared root=%s\n' "$(date -Iseconds)" "$root" \
            | tee -a "$log"
        return
    fi
    env PYTHONPATH="$repo" python3 "$repo/scripts/prepare_wt_zero_pressure_confirmation.py" \
        --out "$root" --volume-scan-root "$scan" --temperature "$temperature" \
        --steps 3000 --csvr-tau 5 --seed "$seed" \
        --config "$repo/config/abacus_wt_ti_node04_cpu12.json" >> "$log" 2>&1
    printf '%s confirmation_prepared root=%s\n' "$(date -Iseconds)" "$root" \
        | tee -a "$log"
}

run_confirmation() {
    local root=$1
    if [[ -e $root/confirmation_summary.json ]]; then
        python3 - "$root/confirmation_summary.json" <<'PY' >> "$log"
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "all_confirmations_passed":
    raise SystemExit("existing confirmation did not pass")
PY
        printf '%s confirmation_already_verified root=%s\n' "$(date -Iseconds)" "$root" \
            | tee -a "$log"
        return
    fi
    if find "$root" -mindepth 2 -maxdepth 2 -type d -name 'OUT.*' | grep -q .; then
        printf '%s refusing_confirmation reason=partial_existing_output root=%s\n' \
            "$(date -Iseconds)" "$root" | tee -a "$log"
        return 2
    fi
    printf '%s launching_confirmation root=%s\n' "$(date -Iseconds)" "$root" \
        | tee -a "$log"
    bash "$repo/scripts/launch_prepared_zero_pressure_pair_node01.sh" "$root" \
        >> "$log" 2>&1
    printf '%s confirmation_verified root=%s\n' "$(date -Iseconds)" "$root" \
        | tee -a "$log"
}

run_volume_scan "$scan_975"
run_volume_scan "$scan_1050"
if [[ $need_1100 == yes ]]; then
    run_volume_scan "$scan_1100"
fi
prepare_confirmation "$scan_975" "$enthalpy_975" 975 2026072301
prepare_confirmation "$scan_1050" "$enthalpy_1050" 1050 2026072303
if [[ $need_1100 == yes ]]; then
    prepare_confirmation "$scan_1100" "$enthalpy_1100" 1100 2026072305
fi

run_confirmation "$enthalpy_900"
run_confirmation "$enthalpy_975"
run_confirmation "$enthalpy_1050"
if [[ $need_1100 == yes ]]; then
    run_confirmation "$enthalpy_1100"
fi

mkdir -p "$analysis_root"
manifest=$analysis_root/manifest.json
confirmation_roots=("$enthalpy_900" "$enthalpy_975" "$enthalpy_1050")
sampled_temperatures=(900 975 1050)
if [[ $need_1100 == yes ]]; then
    confirmation_roots+=("$enthalpy_1100")
    sampled_temperatures+=(1100)
fi
env PYTHONPATH="$repo" python3 "$repo/scripts/build_wt_fusion_enthalpy_manifest.py" \
    "${confirmation_roots[@]}" --out "$manifest" \
    >> "$log" 2>&1

reports=()
for suffix in d25 d50 d75; do
    case $suffix in
        d25) discard=0.25 ;;
        d50) discard=0.50 ;;
        d75) discard=0.75 ;;
    esac
    report=$analysis_root/enthalpy_${suffix}.json
    env PYTHONPATH="$repo" python3 "$repo/scripts/analyze_wt_fusion_enthalpy_series.py" \
        "$manifest" --out "$report" --discard-fraction "$discard" \
        --temperature-tolerance 20 --pressure-tolerance 2.5 \
        --maximum-half-drift 5 >> "$log" 2>&1
    reports+=("$report")
done

convergence=$analysis_root/discard_convergence_summary.json
env PYTHONPATH="$repo" python3 "$repo/scripts/summarize_wt_enthalpy_convergence.py" \
    "${reports[@]}" --out "$convergence" >> "$log" 2>&1

selection=$analysis_root/extension_selection.json
env PYTHONPATH="$repo" python3 "$repo/scripts/select_wt_enthalpy_extensions.py" \
    "$convergence" --out "$selection" >> "$log" 2>&1
selection_status=$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$selection")
selected_enthalpy_series=$analysis_root/enthalpy_d50.json
if [[ $selection_status == statistical_extension_required ]]; then
    mapfile -t critical_temperatures < <(
        python3 -c '
import json
import sys
for value in json.load(open(sys.argv[1]))["critical_temperatures_k"]:
    print(int(round(float(value))))
' "$selection"
    )
    if ((${#critical_temperatures[@]} == 0)); then
        printf '%s refusing_enthalpy_extension reason=no_critical_temperatures\n' \
            "$(date -Iseconds)" | tee -a "$log"
        exit 2
    fi
    printf '%s preparing_enthalpy_extensions temperatures=%s\n' \
        "$(date -Iseconds)" "${critical_temperatures[*]}" | tee -a "$log"
    declare -A extension_roots=()
    for temperature in "${critical_temperatures[@]}"; do
        source_root=
        for index in "${!sampled_temperatures[@]}"; do
            if [[ ${sampled_temperatures[$index]} -eq $temperature ]]; then
                source_root=${confirmation_roots[$index]}
                break
            fi
        done
        if [[ -z $source_root ]]; then
            printf '%s refusing_enthalpy_extension reason=unknown_temperature temperature=%s\n' \
                "$(date -Iseconds)" "$temperature" | tee -a "$log"
            exit 2
        fi
        label=$(printf 'T%04d_steps3000_round1' "$temperature")
        extension_run=$analysis_root/critical_extensions/$label
        extension_roots[$temperature]=$extension_run
        if [[ ! -e $extension_run/confirmation_manifest.json ]]; then
            env PYTHONPATH="$repo" python3 "$repo/scripts/prepare_wt_enthalpy_extension.py" \
                --source "$source_root" --out "$extension_run" --steps 3000 \
                --csvr-tau 5 --seed "$((2026072601 + temperature))" \
                --config "$repo/config/abacus_wt_ti_node04_cpu12.json" >> "$log" 2>&1
        fi
        run_confirmation "$extension_run"
    done

    extended_roots=()
    for index in "${!sampled_temperatures[@]}"; do
        temperature=${sampled_temperatures[$index]}
        if [[ -n ${extension_roots[$temperature]+x} ]]; then
            extended_roots+=("${extension_roots[$temperature]}")
        else
            extended_roots+=("${confirmation_roots[$index]}")
        fi
    done
    extended_manifest=$analysis_root/manifest_extension_round1.json
    env PYTHONPATH="$repo" python3 "$repo/scripts/build_wt_fusion_enthalpy_manifest.py" \
        "${extended_roots[@]}" --out "$extended_manifest" >> "$log" 2>&1
    extended_reports=()
    for suffix in d25 d50 d75; do
        case $suffix in
            d25) discard=0.25 ;;
            d50) discard=0.50 ;;
            d75) discard=0.75 ;;
        esac
        report=$analysis_root/enthalpy_extension_${suffix}.json
        env PYTHONPATH="$repo" python3 "$repo/scripts/analyze_wt_fusion_enthalpy_series.py" \
            "$extended_manifest" --out "$report" --discard-fraction "$discard" \
            --temperature-tolerance 20 --pressure-tolerance 2.5 \
            --maximum-half-drift 5 >> "$log" 2>&1
        extended_reports+=("$report")
    done
    convergence=$analysis_root/discard_convergence_summary_extension_round1.json
    env PYTHONPATH="$repo" python3 "$repo/scripts/summarize_wt_enthalpy_convergence.py" \
        "${extended_reports[@]}" --out "$convergence" >> "$log" 2>&1
    selected_enthalpy_series=$analysis_root/enthalpy_extension_d50.json
    printf '%s enthalpy_extensions_analyzed convergence=%s\n' \
        "$(date -Iseconds)" "$convergence" | tee -a "$log"
fi

python3 - "$convergence" <<'PY' >> "$log"
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "verified" or report.get("critical_temperatures"):
    raise SystemExit("fusion-enthalpy convergence remains unverified")
if not report.get("checks") or not all(report["checks"].values()):
    raise SystemExit("fusion-enthalpy convergence checks failed")
PY

melting=$analysis_root/gibbs_helmholtz_melting.json
env PYTHONPATH="$repo" python3 "$repo/scripts/solve_wt_melting_gibbs_helmholtz.py" \
    --combination "$combined" --enthalpy-series "$selected_enthalpy_series" \
    --enthalpy-convergence "$convergence" --out "$melting" >> "$log" 2>&1
python3 - "$melting" <<'PY' >> "$log"
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "verified":
    raise SystemExit("Gibbs-Helmholtz melting result is not verified")
if not report.get("checks") or not all(report["checks"].values()):
    raise SystemExit("Gibbs-Helmholtz checks failed")
PY

printf '%s gibbs_helmholtz_verified output=%s independent_validation_required=true\n' \
    "$(date -Iseconds)" "$melting" | tee -a "$log"
date -Iseconds > "$done_file"
