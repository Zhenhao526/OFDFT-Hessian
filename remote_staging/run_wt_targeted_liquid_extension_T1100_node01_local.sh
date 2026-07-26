#!/usr/bin/env bash
set -euo pipefail

workspace=${WORKSPACE:-/home/shenwei01/WT_Al_melting_workspace_20260724}
repo=${REPO:-$workspace/repository}
runtime=${RUNTIME:-/home/shenwei01/wt_melting_runtime_20260724}
root=${RUN_ROOT:-$workspace/runs/step4_rebuild}
prod=$root/enthalpy_production
extensions=$prod/critical_extensions
source_analysis=$prod/analysis_T1100_recovered_extension_round1
source_root=$extensions/T1100_steps6000_recovered_round1
out=$extensions/T1100_liquid_steps3000_recovered_round2
analysis=$prod/analysis_T1100_targeted_liquid_round2
config=$repo/config/abacus_wt_node01_local_cpu18.json
binary=$runtime/build-abacus-wt-cpu/source/abacus_pw_para
mpirun=$runtime/conda_prefix/bin/mpirun
log=$workspace/audit/step4_enthalpy_targeted_T1100_liquid_round2.log
done_file=$workspace/audit/step4_enthalpy_targeted_T1100_liquid_round2.done
failed_file=$workspace/audit/step4_enthalpy_targeted_T1100_liquid_round2.failed
extension_file=$workspace/audit/step4_enthalpy_targeted_T1100_liquid_round2.needs_extension
source_sha_file=$workspace/audit/step4_enthalpy_targeted_T1100_liquid_round2.source.sha256
extension_steps=3000

mkdir -p "$extensions" "$analysis" "$workspace/audit"
rm -f "$done_file" "$failed_file" "$extension_file"
printf '%s targeted_liquid_extension_started temperature=1100 steps=%s\n' \
    "$(date -Iseconds)" "$extension_steps" > "$log"

mark_failed() {
    local rc=$?
    trap - ERR
    if [[ ! -e $done_file && ! -e $extension_file ]]; then
        printf '%s targeted_liquid_extension_failed exit_code=%s\n' \
            "$(date -Iseconds)" "$rc" | tee -a "$log" "$failed_file"
    fi
    exit "$rc"
}
trap mark_failed ERR

test -x "$binary"
test -x "$mpirun"
test -s "$workspace/audit/step4_enthalpy_T1100_recovery6000.needs_extension"
test -s "$source_analysis/manifest.json"
test -s "$source_analysis/discard_convergence_summary.json"
test -s "$source_root/confirmation_manifest.json"
test -s "$source_root/confirmation_summary.json"
test -s "$source_sha_file"
if pgrep -f "$binary" >/dev/null; then
    printf '%s refusing_existing_abacus_process\n' "$(date -Iseconds)" \
        | tee -a "$log"
    exit 2
fi

python3 - "$source_analysis/discard_convergence_summary.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "extension_required":
    raise SystemExit("source convergence report does not require extension")
critical = report.get("critical_temperatures", {})
if set(critical) != {"1100.000000000"}:
    raise SystemExit(f"unexpected critical temperatures: {sorted(critical)}")
point = report["points"][0]
if point.get("extension_reasons") != ["discard_sensitivity"]:
    raise SystemExit("T1100 extension is not limited to discard sensitivity")
PY

python3 - "$source_root/confirmation_summary.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
rows = {row["phase"]: row for row in report.get("phase_results", [])}
liquid = rows.get("liquid", {})
if report.get("status") != "all_confirmations_passed":
    raise SystemExit("T1100 source pair is not physically verified")
if (
    liquid.get("status") != "confirmation_passed"
    or liquid.get("max_step") != 6000
    or liquid.get("phase_status") != "liquid_verified"
    or float(liquid.get("nearest_neighbor_A", 0.0)) <= 2.0
    or not all(liquid.get("checks", {}).values())
):
    raise SystemExit("T1100 source liquid phase is not a valid extension source")
PY

source_dump=$(find "$source_root/liquid" -path '*/OUT.*/MD_dump' -type f)
test -n "$source_dump"
test "$(printf '%s\n' "$source_dump" | wc -l)" -eq 1
sha256sum -c "$source_sha_file" >> "$log" 2>&1

if [[ ! -e $out/confirmation_manifest.json ]]; then
    env PYTHONPATH="$repo" python3 \
        "$repo/scripts/prepare_wt_enthalpy_phase_extension.py" \
        --source "$source_root" --out "$out" --phase liquid \
        --steps "$extension_steps" --csvr-tau 5 --seed 2026122110 \
        --config "$config" >> "$log" 2>&1
fi

point=$out/liquid
test -s "$point/INPUT"
test -s "$point/STRU"
test -s "$point/KPT"
python3 - "$out/confirmation_manifest.json" "$source_dump" <<'PY'
import json
import math
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
if manifest.get("source_velocities_discarded") is not False:
    raise SystemExit("source velocities were not preserved")
if manifest.get("steps") != 3000:
    raise SystemExit("targeted extension step count differs")
if [row.get("phase") for row in manifest.get("phases", [])] != ["liquid"]:
    raise SystemExit("targeted extension must contain only liquid")
row = manifest["phases"][0]
if row.get("source_step") != 5995:
    raise SystemExit(f"expected source step 5995, got {row.get('source_step')}")
if not math.isclose(
    float(row["volume_per_atom_A3"]), 19.106478865122236,
    rel_tol=1.0e-12, abs_tol=1.0e-12,
):
    raise SystemExit("targeted liquid volume changed")
if row.get("source") != sys.argv[2]:
    raise SystemExit("targeted extension source MD_dump differs from preflight")
PY
if find "$point" -maxdepth 1 -type d -name 'OUT.*' | grep -q .; then
    printf '%s refusing_existing_output point=%s\n' \
        "$(date -Iseconds)" "$point" | tee -a "$log"
    exit 2
fi

(
    cd "$point"
    exec env \
        PATH="$runtime/conda_prefix/bin:$PATH" \
        LD_LIBRARY_PATH="$runtime/conda_prefix/lib:${LD_LIBRARY_PATH:-}" \
        OMP_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 \
        taskset -c 38-73 "$mpirun" -np 36 --bind-to none "$binary" \
        > run.stdout 2>&1
)
printf '%s targeted_liquid_extension_finished\n' "$(date -Iseconds)" \
    | tee -a "$log"

env PYTHONPATH="$repo" python3 \
    "$repo/scripts/analyze_wt_zero_pressure_confirmation.py" \
    "$out" --temperature-tolerance 20 --pressure-tolerance 2.5 \
    >> "$log" 2>&1
python3 - "$out/confirmation_summary.json" "$extension_steps" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
rows = report.get("phase_results", [])
if report.get("status") != "all_confirmations_passed" or len(rows) != 1:
    raise SystemExit("targeted liquid confirmation did not pass")
row = rows[0]
if (
    row.get("phase") != "liquid"
    or row.get("max_step") != int(sys.argv[2])
    or row.get("status") != "confirmation_passed"
    or not all(row.get("checks", {}).values())
):
    raise SystemExit("targeted liquid gate details did not all pass")
PY

manifest=$analysis/manifest.json
env PYTHONPATH="$repo" python3 \
    "$repo/scripts/build_wt_targeted_enthalpy_manifest.py" \
    "$source_analysis/manifest.json" "$out" --out "$manifest" \
    >> "$log" 2>&1

python3 - "$manifest" <<'PY'
import json
import sys

point = json.load(open(sys.argv[1], encoding="utf-8"))["points"][0]
if point.get("solid_segment_steps") != [1700, 6000]:
    raise SystemExit("unexpected solid segment chain")
if point.get("liquid_segment_steps") != [1700, 6000, 3000]:
    raise SystemExit("unexpected liquid segment chain")
if point.get("solid_steps") != 7700 or point.get("liquid_steps") != 10700:
    raise SystemExit("unexpected targeted cumulative lengths")
if int(0.25 * point["solid_steps"]) <= 1700:
    raise SystemExit("solid d25 still includes the recovered parent")
if int(0.25 * point["liquid_steps"]) <= 1700:
    raise SystemExit("liquid d25 still includes the recovered parent")
PY

reports=()
for suffix in d25 d50 d75; do
    case $suffix in
        d25) discard=0.25 ;;
        d50) discard=0.50 ;;
        d75) discard=0.75 ;;
    esac
    report=$analysis/enthalpy_${suffix}.json
    env PYTHONPATH="$repo" python3 \
        "$repo/scripts/analyze_wt_fusion_enthalpy_series.py" \
        "$manifest" --out "$report" --discard-fraction "$discard" \
        --temperature-tolerance 20 --pressure-tolerance 2.5 \
        --maximum-half-drift 5 >> "$log" 2>&1
    reports+=("$report")
done

convergence=$analysis/discard_convergence_summary.json
env PYTHONPATH="$repo" python3 \
    "$repo/scripts/summarize_wt_enthalpy_convergence.py" \
    "${reports[@]}" --out "$convergence" >> "$log" 2>&1

(
    cd "$prod"
    find critical_extensions/T1100_liquid_steps3000_recovered_round2 \
        analysis_T1100_targeted_liquid_round2 \
        -type f -print0 | sort -z | xargs -0 sha256sum
) > "$prod/SHA256SUMS_targeted_T1100_liquid_round2"

status=$(python3 - "$convergence" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])
PY
)
if [[ $status != verified ]]; then
    printf '%s targeted_liquid_extension_needs_extension status=%s\n' \
        "$(date -Iseconds)" "$status" | tee -a "$log" "$extension_file"
    trap - ERR
    exit 3
fi

date -Iseconds > "$done_file"
printf '%s targeted_liquid_extension_verified\n' "$(date -Iseconds)" \
    | tee -a "$log"
