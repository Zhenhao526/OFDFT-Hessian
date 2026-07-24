#!/usr/bin/env bash
set -euo pipefail

repo=/scratch/xzh/OFDFT-Hessian
ti_root=${TI_ROOT:-$repo/runs/al/free_energy_wt/ti_windows/phase_specific_T0900_zeroP_lambda5_steps1000_v1_node01}
enthalpy_root=${ENTHALPY_ROOT:-$repo/runs/al/free_energy_wt/fusion_enthalpy_sampling/T0900_continuation_steps3000_v2_node01}
decision_log=$enthalpy_root/ti_to_enthalpy_gate.log
launcher_log=$ti_root/run_phase_specific_ti_node01.log

printf '%s waiting_for_verified_ti root=%s\n' "$(date -Iseconds)" "$ti_root" \
    > "$decision_log"
while [[ ! -e $ti_root/post_ti_analysis.done ]]; do
    if grep -q 'phase_specific_ti_complete failed=' "$launcher_log" 2>/dev/null \
        && ! grep -q 'phase_specific_ti_complete failed=0' "$launcher_log"; then
        printf '%s refusing_enthalpy reason=ti_launcher_failed\n' "$(date -Iseconds)" \
            | tee -a "$decision_log"
        exit 2
    fi
    sleep 60
done

solid_summary=$ti_root/solid/discard_convergence_summary.json
liquid_summary=$ti_root/liquid/discard_convergence_summary.json
if ! env PYTHONPATH="$repo" python3 - "$solid_summary" "$liquid_summary" <<'PY'
import json
import sys

expected = ("solid", "liquid")
for path, phase in zip(sys.argv[1:], expected):
    report = json.load(open(path, encoding="utf-8"))
    if report.get("status") != "verified" or report.get("phase") != phase:
        raise SystemExit(2)
    if report.get("lambda_refinement_required"):
        raise SystemExit(2)
    if report.get("critical_windows"):
        raise SystemExit(2)
PY
then
    printf '%s refusing_enthalpy reason=ti_convergence_gate_failed\n' \
        "$(date -Iseconds)" | tee -a "$decision_log"
    exit 2
fi

combined=$repo/runs/al/free_energy_wt/melting_free_energy_T0900_v1.json
if ! env PYTHONPATH="$repo" python3 scripts/combine_wt_melting_free_energy.py \
    --analytic "$repo/runs/al/free_energy_wt/analytic_reference_T0900_v1.json" \
    --solid-anchor "$repo/runs/al/free_energy_wt/absolute_anchor/solid_einstein_pair_T0900_v1/lambda9_steps20000_v1_node01/ti_analysis.json" \
    --liquid-anchor "$repo/runs/al/free_energy_wt/absolute_anchor/liquid_suf_pair_T0900_p50_sigma1p54_lambda9_steps20000_v1_node01/ti_analysis.json" \
    --anchor-audit "$repo/runs/al/free_energy_wt/reference_anchor_audit_T0900_v1.json" \
    --solid-wt "$ti_root/solid/ti_prod_d50.json" \
    --liquid-wt "$ti_root/liquid/ti_prod_d50.json" \
    --solid-wt-convergence "$solid_summary" \
    --liquid-wt-convergence "$liquid_summary" \
    --zero-pressure-summary "$repo/runs/al/free_energy_wt/zeroP_confirmation_T0900_combined_v17p940_v2_node01/confirmation_summary.json" \
    --out "$combined" >> "$decision_log" 2>&1
then
    printf '%s refusing_enthalpy reason=free_energy_combination_failed\n' \
        "$(date -Iseconds)" | tee -a "$decision_log"
    exit 2
fi
printf '%s free_energy_anchor_verified output=%s\n' "$(date -Iseconds)" "$combined" \
    | tee -a "$decision_log"

if [[ -e $enthalpy_root/run_zero_pressure_pair_node01.log ]] \
    || find "$enthalpy_root" -mindepth 2 -maxdepth 2 -type d -name 'OUT.*' | grep -q .; then
    printf '%s refusing_enthalpy reason=existing_launcher_or_output\n' \
        "$(date -Iseconds)" | tee -a "$decision_log"
    exit 2
fi

printf '%s launching_enthalpy root=%s\n' "$(date -Iseconds)" "$enthalpy_root" \
    | tee -a "$decision_log"
cd "$repo"
exec bash scripts/launch_prepared_zero_pressure_pair_node01.sh "$enthalpy_root"
