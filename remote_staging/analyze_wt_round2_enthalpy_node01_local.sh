#!/usr/bin/env bash
set -euo pipefail

workspace=${WORKSPACE:-/home/shenwei01/WT_Al_melting_workspace_20260724}
repo=${REPO:-$workspace/repository}
inputs=${INPUTS:-$workspace/staged_inputs/free_energy_wt}
analysis=${ANALYSIS:-$workspace/analysis/step4_round2_enthalpy}
stage_ready=$workspace/audit/staged_inputs.ready
log=$workspace/audit/step4_round2_enthalpy.log
done_file=$workspace/audit/step4_round2_enthalpy.done
verified_file=$workspace/audit/step4_round2_enthalpy.verified
failed_file=$workspace/audit/step4_round2_enthalpy.failed

mkdir -p "$analysis" "$workspace/audit"
rm -f "$done_file" "$verified_file" "$failed_file"
printf '%s step4_watcher_started\n' "$(date -Iseconds)" >> "$log"

mark_failed() {
    local rc=$?
    trap - ERR
    if [[ ! -e $done_file ]]; then
        printf '%s step4_failed exit_code=%s\n' "$(date -Iseconds)" "$rc" \
            | tee -a "$log" "$failed_file"
    fi
    exit "$rc"
}
trap mark_failed ERR

while [[ ! -e $stage_ready ]]; do
    printf '%s waiting_for_staged_inputs\n' "$(date -Iseconds)" >> "$log"
    sleep 300
done

(
    cd "$workspace"
    sha256sum -c "$workspace/audit/staged_inputs.sha256"
) >> "$log" 2>&1
printf '%s staged_input_checksums_verified\n' "$(date -Iseconds)" >> "$log"

series=T0900_T0975_T1050_adaptive_T1100_pairv2_steps3000_v2_node01
root900=$inputs/fusion_enthalpy_sampling/T0900_continuation_steps3000_v2_node01
root975=$inputs/fusion_enthalpy_sampling/$series/critical_extensions/T0975_steps3000_round2
root1050=$inputs/fusion_enthalpy_sampling/$series/critical_extensions/T1050_steps3000_round2
combination=$inputs/melting_free_energy_T0900_pairv2_v2.json

for root in "$root900" "$root975" "$root1050"; do
    test -s "$root/confirmation_manifest.json"
    test -s "$root/confirmation_summary.json"
    for phase in solid liquid; do
        test "$(find "$root/$phase" -path '*/running_md.log' -type f | wc -l)" -ge 1
        test "$(find "$root/$phase" -path '*/MD_dump' -type f | wc -l)" -ge 1
    done
done
test -s "$combination"
printf '%s required_round2_inputs_verified\n' "$(date -Iseconds)" >> "$log"

manifest=$analysis/manifest_T0900_T0975_T1050_round2_local.json
env PYTHONPATH="$repo" python3 "$repo/scripts/build_wt_fusion_enthalpy_manifest.py" \
    "$root900" "$root975" "$root1050" --out "$manifest" >> "$log" 2>&1

reports=()
for suffix in d25 d50 d75; do
    case $suffix in
        d25) discard=0.25 ;;
        d50) discard=0.50 ;;
        d75) discard=0.75 ;;
    esac
    report=$analysis/enthalpy_T0900_T0975_T1050_round2_${suffix}.json
    env PYTHONPATH="$repo" python3 "$repo/scripts/analyze_wt_fusion_enthalpy_series.py" \
        "$manifest" --out "$report" --discard-fraction "$discard" \
        --temperature-tolerance 20 --phase-temperature-difference-tolerance 20 \
        --pressure-tolerance 2.5 --maximum-half-drift 5 >> "$log" 2>&1
    reports+=("$report")
done

convergence=$analysis/discard_convergence_T0900_T0975_T1050_round2.json
env PYTHONPATH="$repo" python3 "$repo/scripts/summarize_wt_enthalpy_convergence.py" \
    "${reports[@]}" --maximum-discard-spread 2 \
    --maximum-block-standard-error 3 --out "$convergence" >> "$log" 2>&1

python3 - "$convergence" "$analysis/input_gate_summary.json" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
report = json.loads(source.read_text(encoding="utf-8"))
summary = {
    "schema": "wt-step4-round2-input-gate-v1",
    "status": report["status"],
    "checks": report["checks"],
    "critical_temperatures": report["critical_temperatures"],
    "points": report["points"],
    "provenance": {"convergence": str(source)},
}
output.write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

status=$(python3 - "$convergence" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])
PY
)

if [[ $status != verified ]]; then
    printf '%s enthalpy_gate_not_verified convergence=%s\n' \
        "$(date -Iseconds)" "$convergence" | tee -a "$log"
    date -Iseconds > "$done_file"
    exit 0
fi

melting=$analysis/gibbs_helmholtz_T0900_T0975_T1050_round2.json
env PYTHONPATH="$repo" python3 "$repo/scripts/solve_wt_melting_gibbs_helmholtz.py" \
    --combination "$combination" \
    --enthalpy-series "$analysis/enthalpy_T0900_T0975_T1050_round2_d50.json" \
    --enthalpy-convergence "$convergence" --out "$melting" >> "$log" 2>&1

python3 - "$melting" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "verified":
    raise SystemExit("Gibbs-Helmholtz result did not pass every gate")
if not report.get("checks") or not all(report["checks"].values()):
    raise SystemExit("Gibbs-Helmholtz checks did not all pass")
PY

(
    cd "$analysis"
    find . -type f -print0 | sort -z | xargs -0 sha256sum
) > "$analysis/SHA256SUMS"
date -Iseconds > "$done_file"
date -Iseconds > "$verified_file"
printf '%s step4_gibbs_helmholtz_verified output=%s\n' \
    "$(date -Iseconds)" "$melting" | tee -a "$log"
