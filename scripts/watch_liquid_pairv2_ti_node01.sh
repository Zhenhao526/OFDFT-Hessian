#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/scratch/xzh/OFDFT-Hessian}
liquid_root=${LIQUID_ROOT:-$repo/runs/al/free_energy_wt/ti_windows/phase_specific_T0900_zeroP_liquid_pairv2_lambda5_steps1000_v3_node01}
solid_root=${SOLID_ROOT:-$repo/runs/al/free_energy_wt/ti_windows/phase_specific_T0900_zeroP_lambda5_steps1000_v1_node01/solid}
log=$liquid_root/post_ti_gate.log
done_file=$liquid_root/post_ti_gate.done
failed_file=$liquid_root/post_ti_gate.failed

mark_failed() {
    local rc=$?
    trap - ERR
    if [[ ! -e $done_file ]]; then
        printf '%s failed reason=post_ti_gate_command_error exit_code=%s\n' \
            "$(date -Iseconds)" "$rc" | tee -a "$log" "$failed_file"
    fi
    exit "$rc"
}
trap mark_failed ERR

if [[ -e $done_file ]]; then
    printf '%s already_complete file=%s\n' "$(date -Iseconds)" "$done_file"
    exit 0
fi
rm -f "$failed_file"
printf '%s waiting_for_liquid_ti root=%s\n' "$(date -Iseconds)" "$liquid_root" > "$log"

expected=$(python3 - "$liquid_root/manifest.json" <<'PY'
import json
import sys
print(len(json.load(open(sys.argv[1], encoding="utf-8"))["windows"]))
PY
)
while true; do
    complete=0
    for window in "$liquid_root"/lambda_*; do
        run_log=$(find "$window" -path '*/running_md.log' -type f | head -1 || true)
        if [[ -n $run_log ]] && grep -q 'MPN_TI_COMPONENTS step=999 ' "$run_log"; then
            complete=$((complete + 1))
        fi
    done
    printf '%s progress complete=%s expected=%s\n' \
        "$(date -Iseconds)" "$complete" "$expected" >> "$log"
    if [[ $complete -eq $expected ]]; then
        break
    fi
    if ! tmux list-sessions -F '#S' 2>/dev/null \
        | grep -Eq '^ti_lpv3_(000|250|500|750|1000)$'; then
        printf '%s failed reason=compute_sessions_gone_before_completion\n' \
            "$(date -Iseconds)" | tee -a "$log" "$failed_file"
        exit 2
    fi
    sleep 60
done

cd "$repo"
for percent in 25 50 75; do
    fraction=$(printf '0.%02d' "$percent")
    output=$liquid_root/ti_prod_d${percent}.json
    printf '%s analysis_started discard=%s output=%s\n' \
        "$(date -Iseconds)" "$fraction" "$output" | tee -a "$log"
    env PYTHONPATH="$repo" python3 scripts/analyze_wt_pair_ti_production.py \
        "$liquid_root" --discard-fraction "$fraction" \
        --max-quadrature-difference 1 --out "$output" >> "$log" 2>&1
done

liquid_summary=$liquid_root/discard_convergence_summary.json
env PYTHONPATH="$repo" python3 scripts/summarize_wt_ti_convergence.py \
    "$liquid_root/ti_prod_d25.json" \
    "$liquid_root/ti_prod_d50.json" \
    "$liquid_root/ti_prod_d75.json" \
    --out "$liquid_summary" >> "$log" 2>&1

if ! python3 - "$liquid_summary" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "verified":
    raise SystemExit(2)
if report.get("critical_windows") or report.get("lambda_refinement_required"):
    raise SystemExit(2)
PY
then
    printf '%s failed reason=liquid_ti_convergence_gate\n' \
        "$(date -Iseconds)" | tee -a "$log" "$failed_file"
    exit 2
fi

combined=$repo/runs/al/free_energy_wt/melting_free_energy_T0900_pairv2_v2.json
env PYTHONPATH="$repo" python3 scripts/combine_wt_melting_free_energy.py \
    --analytic "$repo/runs/al/free_energy_wt/analytic_reference_T0900_p100_sigma1p36_v2.json" \
    --solid-anchor "$repo/runs/al/free_energy_wt/absolute_anchor/solid_einstein_pair_T0900_v1/lambda9_steps20000_v1_node01/ti_analysis.json" \
    --liquid-anchor "$repo/runs/al/free_energy_wt/absolute_anchor/liquid_suf_pair_T0900_p100_sigma1p36_lambda9_steps20000_v2_node01/ti_analysis_strict_v2.json" \
    --anchor-audit "$repo/runs/al/free_energy_wt/reference_anchor_audit_T0900_p100sigma1p36_pairv2_v2.json" \
    --solid-wt "$solid_root/ti_prod_d50.json" \
    --liquid-wt "$liquid_root/ti_prod_d50.json" \
    --solid-wt-convergence "$solid_root/discard_convergence_summary_v2.json" \
    --liquid-wt-convergence "$liquid_summary" \
    --zero-pressure-summary "$repo/runs/al/free_energy_wt/zeroP_confirmation_T0900_combined_v17p940_v2_node01/confirmation_summary.json" \
    --out "$combined" >> "$log" 2>&1

python3 - "$combined" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "anchor_temperature_free_energy_verified":
    raise SystemExit(2)
if not all(report.get("checks", {}).values()):
    raise SystemExit(2)
PY

printf '%s verified_free_energy output=%s\n' "$(date -Iseconds)" "$combined" \
    | tee -a "$log"
date -Iseconds > "$done_file"
