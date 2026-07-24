#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/scratch/xzh/OFDFT-Hessian}
liquid_root=${LIQUID_ROOT:-$repo/runs/al/free_energy_wt/ti_windows/phase_specific_T0900_zeroP_liquid_pairv2_lambda5_steps1000_v3_node01}
solid_root=${SOLID_ROOT:-$repo/runs/al/free_energy_wt/ti_windows/phase_specific_T0900_zeroP_lambda5_steps1000_v1_node01/solid}
round1=${ROUND1_ROOT:-$repo/runs/al/free_energy_wt/ti_windows/phase_specific_T0900_zeroP_liquid_pairv2_critical_extension_round1_steps1000_node01}
round2=${ROUND2_ROOT:-$repo/runs/al/free_energy_wt/ti_windows/phase_specific_T0900_zeroP_liquid_pairv2_critical_extension_round2_lambda0500_steps1000_node01}
combined=${COMBINED:-$repo/runs/al/free_energy_wt/melting_free_energy_T0900_pairv2_v2.json}
pipeline_root=${PIPELINE_ROOT:-$repo/runs/al/free_energy_wt/melting_pipeline_pairv2_v1_node01}
log=$round2/round2_recovery.log
done_file=$round2/round2_recovery.done
failed_file=$round2/round2_recovery.failed

mkdir -p "$round2"

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

python3 - "$round1/discard_convergence_summary.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "extension_or_refinement_required":
    raise SystemExit("round-one summary is not awaiting a statistical extension")
if report.get("lambda_refinement_required"):
    raise SystemExit("lambda refinement cannot be repaired by a continuation")
critical = report.get("critical_windows", [])
expected = [{"label": "lambda_0p500", "lambda": 0.5, "reasons": ["window_discard_spread"]}]
if critical != expected:
    raise SystemExit(f"unexpected round-one critical windows: {critical!r}")
PY

if [[ ! -e $round2/liquid/manifest.json ]]; then
    env PYTHONPATH="$repo" python3 "$repo/scripts/prepare_ti_window_extensions.py" \
        --phase-root "liquid=$round1/liquid" --out "$round2" \
        --phases liquid --labels lambda_0p500 --steps 1000 \
        --csvr-tau 10 --seed 2026072501 \
        --config "$repo/config/abacus_wt_ti_node04_cpu12.json" >> "$log" 2>&1
fi

window=$round2/liquid/lambda_0p500
run_log=$(find "$window" -path '*/running_md.log' -type f | head -1 || true)
if [[ -z $run_log ]]; then
    if find "$round2" -mindepth 3 -maxdepth 3 -type d -name 'OUT.*' | grep -q .; then
        printf '%s refusing_round2 reason=partial_output_without_log\n' \
            "$(date -Iseconds)" | tee -a "$log"
        exit 2
    fi
    printf '%s launching_round2 label=lambda_0p500\n' "$(date -Iseconds)" | tee -a "$log"
    bash "$repo/scripts/launch_selected_ti_extensions_node01.sh" "$round2" \
        >> "$log" 2>&1
    run_log=$(find "$window" -path '*/running_md.log' -type f | head -1 || true)
fi

while [[ -z $run_log ]] || ! grep -q 'MPN_TI_COMPONENTS step=999 ' "$run_log"; do
    if ! pgrep -af abacus_wt_ti_cpu | grep -Fq "$window" \
        && ! tmux list-sessions -F '#S' 2>/dev/null | grep -qx ti_lpv3_round2_500; then
        printf '%s refusing_round2 reason=process_gone_before_completion\n' \
            "$(date -Iseconds)" | tee -a "$log"
        exit 2
    fi
    step=$(grep -o 'MPN_TI_COMPONENTS step=[0-9]*' "$run_log" 2>/dev/null \
        | tail -1 | sed 's/.*=//' || true)
    printf '%s progress label=lambda_0p500 step=%s\n' \
        "$(date -Iseconds)" "${step:-none}" >> "$log"
    sleep 60
    run_log=$(find "$window" -path '*/running_md.log' -type f | head -1 || true)
done

overrides=(
    --window-override "lambda_0p250=$round1/liquid/lambda_0p250"
    --window-override "lambda_0p500=$round2/liquid/lambda_0p500"
    --window-override "lambda_0p750=$round1/liquid/lambda_0p750"
)
reports=()
for suffix in d25 d50 d75; do
    case $suffix in
        d25) discard=0.25 ;;
        d50) discard=0.50 ;;
        d75) discard=0.75 ;;
    esac
    report=$round2/liquid_ti_prod_${suffix}.json
    env PYTHONPATH="$repo" python3 "$repo/scripts/analyze_wt_pair_ti_production.py" \
        "$liquid_root" --discard-fraction "$discard" \
        --max-quadrature-difference 1 "${overrides[@]}" --out "$report" \
        >> "$log" 2>&1
    reports+=("$report")
done

summary=$round2/discard_convergence_summary.json
env PYTHONPATH="$repo" python3 "$repo/scripts/summarize_wt_ti_convergence.py" \
    "${reports[@]}" --out "$summary" >> "$log" 2>&1

python3 - "$summary" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "verified":
    raise SystemExit("liquid TI remains unverified after round two")
if report.get("critical_windows") or report.get("lambda_refinement_required"):
    raise SystemExit("round-two TI still has critical windows or lambda refinement")
PY

env PYTHONPATH="$repo" python3 "$repo/scripts/combine_wt_melting_free_energy.py" \
    --analytic "$repo/runs/al/free_energy_wt/analytic_reference_T0900_p100_sigma1p36_v2.json" \
    --solid-anchor "$repo/runs/al/free_energy_wt/absolute_anchor/solid_einstein_pair_T0900_v1/lambda9_steps20000_v1_node01/ti_analysis.json" \
    --liquid-anchor "$repo/runs/al/free_energy_wt/absolute_anchor/liquid_suf_pair_T0900_p100_sigma1p36_lambda9_steps20000_v2_node01/ti_analysis_strict_v2.json" \
    --anchor-audit "$repo/runs/al/free_energy_wt/reference_anchor_audit_T0900_p100sigma1p36_pairv2_v2.json" \
    --solid-wt "$solid_root/ti_prod_d50.json" \
    --liquid-wt "$round2/liquid_ti_prod_d50.json" \
    --solid-wt-convergence "$solid_root/discard_convergence_summary_v2.json" \
    --liquid-wt-convergence "$summary" \
    --zero-pressure-summary "$repo/runs/al/free_energy_wt/zeroP_confirmation_T0900_combined_v17p940_v2_node01/confirmation_summary.json" \
    --out "$combined" >> "$log" 2>&1

python3 - "$combined" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "anchor_temperature_free_energy_verified":
    raise SystemExit("combined anchor is not verified")
if not report.get("checks") or not all(report["checks"].values()):
    raise SystemExit("combined anchor checks failed")
PY

date -Iseconds > "$done_file"
printf '%s round2_anchor_verified output=%s\n' \
    "$(date -Iseconds)" "$combined" | tee -a "$log"

if ! tmux list-sessions -F '#S' 2>/dev/null | grep -qx wt_melt_pipeline; then
    rm -f "$pipeline_root/pipeline.failed"
    tmux new-session -d -s wt_melt_pipeline \
        "cd $repo && exec bash scripts/watch_free_energy_then_multitemp_enthalpy_node01.sh"
    printf '%s restarted_downstream session=wt_melt_pipeline\n' \
        "$(date -Iseconds)" | tee -a "$log"
fi
