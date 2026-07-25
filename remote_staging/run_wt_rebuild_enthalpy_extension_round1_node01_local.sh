#!/usr/bin/env bash
set -euo pipefail

workspace=${WORKSPACE:-/home/shenwei01/WT_Al_melting_workspace_20260724}
repo=${REPO:-$workspace/repository}
runtime=${RUNTIME:-/home/shenwei01/wt_melting_runtime_20260724}
root=${RUN_ROOT:-$workspace/runs/step4_rebuild}
prod=$root/enthalpy_production
extensions=$prod/critical_extensions
round=${ROUND:-1}
extension_steps=${EXTENSION_STEPS:-3000}
parent_steps=${PARENT_STEPS:-3000}
if ! [[ $round =~ ^[1-9][0-9]*$ ]]; then
    printf 'ROUND must be a positive integer, got %s\n' "$round" >&2
    exit 2
fi
if ! [[ $extension_steps =~ ^[1-9][0-9]*$ ]]; then
    printf 'EXTENSION_STEPS must be a positive integer, got %s\n' \
        "$extension_steps" >&2
    exit 2
fi
if ! [[ $parent_steps =~ ^[1-9][0-9]*$ ]]; then
    printf 'PARENT_STEPS must be a positive integer, got %s\n' \
        "$parent_steps" >&2
    exit 2
fi
previous_round=$((round - 1))
if ((round == 1)); then
    source_analysis=$prod/analysis_T0975_T1050
    source_975=$prod/T0975_steps3000
    source_1050=$prod/T1050_steps3000
    prior_marker=$workspace/audit/step4_enthalpy_prod_975_1050.needs_extension
else
    source_analysis=$prod/analysis_T0975_T1050_extension_round${previous_round}
    source_975=$extensions/T0975_steps${parent_steps}_round${previous_round}
    source_1050=$extensions/T1050_steps${parent_steps}_round${previous_round}
    prior_marker=$workspace/audit/step4_enthalpy_extension_round${previous_round}.needs_extension
fi
analysis=$prod/analysis_T0975_T1050_extension_round${round}
config=$repo/config/abacus_wt_node01_local_cpu18.json
binary=$runtime/build-abacus-wt-cpu/source/abacus_pw_para
mpirun=$runtime/conda_prefix/bin/mpirun
selection=$analysis/extension_selection.json
log=$workspace/audit/step4_enthalpy_extension_round${round}.log
done_file=$workspace/audit/step4_enthalpy_extension_round${round}.done
failed_file=$workspace/audit/step4_enthalpy_extension_round${round}.failed
extension_file=$workspace/audit/step4_enthalpy_extension_round${round}.needs_extension

mkdir -p "$extensions" "$analysis" "$workspace/audit"
rm -f "$done_file" "$failed_file" "$extension_file"
printf '%s enthalpy_extension_started round=%s\n' \
    "$(date -Iseconds)" "$round" > "$log"
printf '%s extension_steps=%s parent_steps=%s\n' \
    "$(date -Iseconds)" "$extension_steps" "$parent_steps" >> "$log"

mark_failed() {
    local rc=$?
    trap - ERR
    if [[ ! -e $done_file && ! -e $extension_file ]]; then
        printf '%s enthalpy_extension_failed round=%s exit_code=%s\n' \
            "$(date -Iseconds)" "$round" "$rc" | tee -a "$log" "$failed_file"
    fi
    exit "$rc"
}
trap mark_failed ERR

test -x "$binary"
test -x "$mpirun"
test -s "$source_analysis/discard_convergence_summary.json"
test -s "$prior_marker"

env PYTHONPATH="$repo" python3 \
    "$repo/scripts/select_wt_enthalpy_extensions.py" \
    "$source_analysis/discard_convergence_summary.json" \
    --out "$selection" >> "$log" 2>&1

python3 - "$selection" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "statistical_extension_required":
    raise SystemExit("extension selection is not statistical-only")
temperatures = [round(float(value)) for value in report["critical_temperatures_k"]]
if temperatures != [975, 1050]:
    raise SystemExit(f"unexpected extension temperatures: {temperatures}")
PY

prepare_extension() {
    local temperature=$1
    local seed=$2
    local source=$3
    local out=$extensions/T${temperature}_steps${extension_steps}_round${round}

    test -s "$source/confirmation_manifest.json"
    test -s "$source/confirmation_summary.json"
    if [[ ! -e $out/confirmation_manifest.json ]]; then
        env PYTHONPATH="$repo" python3 \
            "$repo/scripts/prepare_wt_enthalpy_extension.py" \
            --source "$source" --out "$out" --steps "$extension_steps" \
            --csvr-tau 5 --seed "$seed" --config "$config" \
            >> "$log" 2>&1
    fi
}

prepare_extension 0975 "$((2026091975 + previous_round * 10000))" "$source_975"
prepare_extension 1050 "$((2026092050 + previous_round * 10000))" "$source_1050"

roots=(
    "$extensions/T0975_steps${extension_steps}_round${round}"
    "$extensions/T1050_steps${extension_steps}_round${round}"
)
points=(
    "${roots[0]}/solid"
    "${roots[0]}/liquid"
    "${roots[1]}/solid"
    "${roots[1]}/liquid"
)
cpu_ranges=(0-17 19-36 38-55 57-74)

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
            taskset -c "$cpus" "$mpirun" -np 18 --bind-to none "$binary" \
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

for pair in "${roots[@]}"; do
    env PYTHONPATH="$repo" python3 \
        "$repo/scripts/analyze_wt_zero_pressure_confirmation.py" \
        "$pair" --temperature-tolerance 20 --pressure-tolerance 2.5 \
        >> "$log" 2>&1
    python3 - "$pair/confirmation_summary.json" "$extension_steps" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "all_confirmations_passed":
    raise SystemExit("extension phase pair did not pass")
if any(
    row.get("max_step") != int(sys.argv[2])
    or row.get("status") != "confirmation_passed"
    or not all(row.get("checks", {}).values())
    for row in report.get("phase_results", [])
):
    raise SystemExit("extension phase gate details did not all pass")
PY
done

manifest=$analysis/manifest.json
env PYTHONPATH="$repo" python3 \
    "$repo/scripts/build_wt_fusion_enthalpy_manifest.py" \
    "${roots[@]}" --out "$manifest" >> "$log" 2>&1

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
    find critical_extensions/T0975_steps${extension_steps}_round${round} \
        critical_extensions/T1050_steps${extension_steps}_round${round} \
        analysis_T0975_T1050_extension_round${round} \
        -type f -print0 | sort -z | xargs -0 sha256sum
) > "$prod/SHA256SUMS_extension_round${round}"

status=$(python3 - "$convergence" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])
PY
)
if [[ $status != verified ]]; then
    printf '%s enthalpy_extension_needs_extension round=%s status=%s\n' \
        "$(date -Iseconds)" "$round" "$status" | tee -a "$log" "$extension_file"
    trap - ERR
    exit 3
fi

date -Iseconds > "$done_file"
printf '%s enthalpy_extension_verified round=%s\n' \
    "$(date -Iseconds)" "$round" | tee -a "$log"
