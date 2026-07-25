#!/usr/bin/env bash
set -euo pipefail

workspace=${WORKSPACE:-/home/shenwei01/WT_Al_melting_workspace_20260724}
repo=${REPO:-$workspace/repository}
runtime=${RUNTIME:-/home/shenwei01/wt_melting_runtime_20260724}
root=${RUN_ROOT:-$workspace/runs/step4_rebuild}
source_root=${SOURCE_ROOT:-/home/shenwei01/wt_melting_local_runs_20260724/T1100_resume_s1300_to_s3000}
prod=$root/enthalpy_production
parent=$prod/T1100_recovered_parent1700
extension=$prod/critical_extensions/T1100_steps6000_recovered_round1
analysis=$prod/analysis_T1100_recovered_extension_round1
config=$repo/config/abacus_wt_node01_local_cpu18.json
binary=$runtime/build-abacus-wt-cpu/source/abacus_pw_para
mpirun=$runtime/conda_prefix/bin/mpirun
log=$workspace/audit/step4_enthalpy_T1100_recovery6000.log
done_file=$workspace/audit/step4_enthalpy_T1100_recovery6000.done
failed_file=$workspace/audit/step4_enthalpy_T1100_recovery6000.failed
extension_file=$workspace/audit/step4_enthalpy_T1100_recovery6000.needs_extension

solid_sha=0c6bfa92e50be644ea5459414f67016fb4d2cead12690713466dc18228f90f2d
liquid_sha=bea6019daa2bca2b73cad1da438d31fae366dc7526fbf7786a2afe83e518ea76
solid_volume=18.368954645234547
liquid_volume=19.106478865122236
parent_steps=1700
source_step=1695
extension_steps=6000

mkdir -p "$prod/critical_extensions" "$analysis" "$workspace/audit"
rm -f "$done_file" "$failed_file" "$extension_file"
printf '%s T1100_recovery_started\n' "$(date -Iseconds)" > "$log"

mark_failed() {
    local rc=$?
    trap - ERR
    if [[ ! -e $done_file && ! -e $extension_file ]]; then
        printf '%s T1100_recovery_failed exit_code=%s\n' \
            "$(date -Iseconds)" "$rc" | tee -a "$log" "$failed_file"
    fi
    exit "$rc"
}
trap mark_failed ERR

test -x "$binary"
test -x "$mpirun"
test -s "$config"
test -d "$source_root/solid"
test -d "$source_root/liquid"

if [[ ! -e $parent/confirmation_manifest.json ]]; then
    env PYTHONPATH="$repo" python3 \
        "$repo/scripts/prepare_wt_recovered_confirmation.py" \
        --source-root "$source_root" --out "$parent" \
        --temperature 1100 --steps "$parent_steps" \
        --expected-last-step "$source_step" --expected-natoms 108 \
        --csvr-tau 5 --solid-volume "$solid_volume" \
        --liquid-volume "$liquid_volume" \
        --solid-sha256 "$solid_sha" --liquid-sha256 "$liquid_sha" \
        >> "$log" 2>&1
fi

env PYTHONPATH="$repo" python3 \
    "$repo/scripts/analyze_wt_zero_pressure_confirmation.py" \
    "$parent" --temperature-tolerance 20 --pressure-tolerance 2.5 \
    >> "$log" 2>&1

python3 - "$parent/confirmation_summary.json" "$parent_steps" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "all_confirmations_passed":
    raise SystemExit("recovered T1100 parent did not pass physical gates")
if any(
    row.get("max_step") != int(sys.argv[2])
    or row.get("status") != "confirmation_passed"
    or not all(row.get("checks", {}).values())
    for row in report.get("phase_results", [])
):
    raise SystemExit("recovered T1100 parent gate details did not all pass")
PY

if [[ ! -e $extension/confirmation_manifest.json ]]; then
    env PYTHONPATH="$repo" python3 \
        "$repo/scripts/prepare_wt_enthalpy_extension.py" \
        --source "$parent" --out "$extension" --steps "$extension_steps" \
        --csvr-tau 5 --seed 2026092110 --config "$config" \
        >> "$log" 2>&1
fi

python3 - "$extension/confirmation_manifest.json" "$source_step" \
    "$solid_volume" "$liquid_volume" <<'PY'
import json
import math
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
expected_step = int(sys.argv[2])
volumes = {"solid": float(sys.argv[3]), "liquid": float(sys.argv[4])}
if manifest.get("source_velocities_discarded") is not False:
    raise SystemExit("T1100 recovery extension discarded source velocities")
if manifest.get("steps") != 6000:
    raise SystemExit("T1100 recovery extension step count is not 6000")
for row in manifest.get("phases", []):
    if row.get("source_step") != expected_step:
        raise SystemExit(f"{row.get('phase')} source step is not {expected_step}")
    if not math.isclose(
        float(row["volume_per_atom_A3"]), volumes[row["phase"]],
        rel_tol=1.0e-12, abs_tol=1.0e-12,
    ):
        raise SystemExit(f"{row['phase']} volume changed during recovery")
PY

points=("$extension/solid" "$extension/liquid")
cpu_ranges=(0-35 38-73)
for point in "${points[@]}"; do
    test -s "$point/INPUT"
    test -s "$point/STRU"
    test -s "$point/KPT"
    if find "$point" -maxdepth 1 -type d -name 'OUT.*' | grep -q .; then
        printf '%s refusing_existing_output point=%s\n' \
            "$(date -Iseconds)" "$point" | tee -a "$log"
        exit 2
    fi
done

pids=()
for index in "${!points[@]}"; do
    point=${points[$index]}
    cpus=${cpu_ranges[$index]}
    (
        cd "$point"
        exec env \
            PATH="$runtime/conda_prefix/bin:$PATH" \
            LD_LIBRARY_PATH="$runtime/conda_prefix/lib:${LD_LIBRARY_PATH:-}" \
            OMP_NUM_THREADS=1 \
            OPENBLAS_NUM_THREADS=1 \
            MKL_NUM_THREADS=1 \
            taskset -c "$cpus" "$mpirun" -np 36 --bind-to none "$binary" \
            > run.stdout 2>&1
    ) &
    pids+=("$!")
    printf '%s started point=%s cpus=%s pid=%s\n' \
        "$(date -Iseconds)" "$point" "$cpus" "$!" | tee -a "$log"
done

run_failed=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        printf '%s finished point=%s\n' \
            "$(date -Iseconds)" "${points[$index]}" | tee -a "$log"
    else
        rc=$?
        printf '%s failed point=%s exit_code=%s\n' \
            "$(date -Iseconds)" "${points[$index]}" "$rc" | tee -a "$log"
        run_failed=1
    fi
done
if ((run_failed != 0)); then
    exit 2
fi

env PYTHONPATH="$repo" python3 \
    "$repo/scripts/analyze_wt_zero_pressure_confirmation.py" \
    "$extension" --temperature-tolerance 20 --pressure-tolerance 2.5 \
    >> "$log" 2>&1

python3 - "$extension/confirmation_summary.json" "$extension_steps" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "all_confirmations_passed":
    raise SystemExit("T1100 extension did not pass physical gates")
if any(
    row.get("max_step") != int(sys.argv[2])
    or row.get("status") != "confirmation_passed"
    or not all(row.get("checks", {}).values())
    for row in report.get("phase_results", [])
):
    raise SystemExit("T1100 extension gate details did not all pass")
PY

manifest=$analysis/manifest.json
env PYTHONPATH="$repo" python3 \
    "$repo/scripts/build_wt_fusion_enthalpy_manifest.py" \
    "$extension" --out "$manifest" >> "$log" 2>&1

python3 - "$manifest" "$parent_steps" "$extension_steps" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
point = report["points"][0]
expected = [int(sys.argv[2]), int(sys.argv[3])]
if point.get("segment_steps") != expected:
    raise SystemExit(f"unexpected recovery segment lengths: {point.get('segment_steps')}")
if point.get("steps") != sum(expected):
    raise SystemExit("unexpected total recovery chain length")
discard_25_start = int(0.25 * point["steps"])
if discard_25_start <= expected[0]:
    raise SystemExit("d25 still includes recovered parent samples")
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
    find T1100_recovered_parent1700 \
        critical_extensions/T1100_steps6000_recovered_round1 \
        analysis_T1100_recovered_extension_round1 \
        -type f -print0 | sort -z | xargs -0 sha256sum
) > "$prod/SHA256SUMS_T1100_recovery6000"

status=$(python3 - "$convergence" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])
PY
)
if [[ $status != verified ]]; then
    printf '%s T1100_recovery_needs_extension status=%s\n' \
        "$(date -Iseconds)" "$status" | tee -a "$log" "$extension_file"
    trap - ERR
    exit 3
fi

date -Iseconds > "$done_file"
printf '%s T1100_recovery_verified\n' "$(date -Iseconds)" | tee -a "$log"
