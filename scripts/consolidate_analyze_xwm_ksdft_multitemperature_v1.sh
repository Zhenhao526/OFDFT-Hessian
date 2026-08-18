#!/usr/bin/env bash
set -euo pipefail

WS=/home/shenwei01/WT_Mg_melting_workspace_20260804
ROOT=runs/mg_three_method_20260804/xwm/ksdft_multitemperature_oneway_diagnostic_v1
ANALYZER=repository/scripts/analyze_xwm_ksdft_multitemperature_oneway.py
XWM_ANCHOR=runs/mg_three_method_20260804/xwm/free_energy_T0900/formal_anchor_with_user_gate_waivers_20260815_v1/preliminary_anchor_and_tm.json
ENTHALPY=runs/mg_three_method_20260804/xwm/multitemp_enthalpy_v1/preliminary_enthalpy_with_pressure_waiver_v1/enthalpy_window_convergence_summary.json
WAIVER=runs/mg_three_method_20260804/xwm/user_gate_waiver_all_remaining_20260815_v1/user_gate_waiver.json
OUTPUT=$ROOT/oneway_diagnostic_analysis.json

cd "$WS"
for node in node02 node06; do
  upper=${node^^}
  rsync -a "$node:$WS/$ROOT/${upper}_ASSIGNED_OUTPUT_VALIDATION.txt" "$ROOT/"
  rsync -a "$node:$WS/$ROOT/${upper}_ASSIGNED_OUTPUT_VALIDATION.txt.sha256" "$ROOT/"
  sha256sum -c "$ROOT/${upper}_ASSIGNED_OUTPUT_VALIDATION.txt.sha256"
  manifest="$ROOT/manifests/production_${node}.txt"
  while IFS= read -r relative; do
    [[ -n "$relative" ]] || continue
    rsync -a "$node:$WS/$ROOT/$relative/" "$ROOT/$relative/"
  done < "$manifest"
done

PYTHONPATH=repository python3 "$ANALYZER" \
  --root "$ROOT" \
  --xwm-anchor "$XWM_ANCHOR" \
  --enthalpy-summary "$ENTHALPY" \
  --waiver "$WAIVER" \
  --output "$OUTPUT" \
  --bootstrap-samples 20000 \
  --seed 20260815 | tee "$ROOT/analysis.stdout"

{
  find "$ROOT/jobs" -name OUTPUT_SHA256SUMS -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum \
    "$ROOT/NODE02_ASSIGNED_OUTPUT_VALIDATION.txt" \
    "$ROOT/NODE06_ASSIGNED_OUTPUT_VALIDATION.txt" \
    "$ROOT/PREPARATION_MANIFEST.json" \
    "$ROOT/PREPARATION_SHA256SUMS" \
    "$OUTPUT" \
    "$ROOT/analysis.stdout"
} > "$ROOT/FINAL_ANALYSIS_SHA256SUMS"
sha256sum "$ROOT/FINAL_ANALYSIS_SHA256SUMS"
