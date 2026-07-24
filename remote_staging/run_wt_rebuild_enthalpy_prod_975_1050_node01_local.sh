#!/usr/bin/env bash
set -euo pipefail

workspace=${WORKSPACE:-/home/shenwei01/WT_Al_melting_workspace_20260724}
repo=${REPO:-$workspace/repository}
runtime=${RUNTIME:-/home/shenwei01/wt_melting_runtime_20260724}
root=${RUN_ROOT:-$workspace/runs/step4_rebuild}
prod=$root/enthalpy_production
analysis=$prod/analysis_T0975_T1050
config=$repo/config/abacus_wt_node01_local_cpu18.json
binary=$runtime/build-abacus-wt-cpu/source/abacus_pw_para
mpirun=$runtime/conda_prefix/bin/mpirun
log=$workspace/audit/step4_enthalpy_prod_975_1050.log
done_file=$workspace/audit/step4_enthalpy_prod_975_1050.done
failed_file=$workspace/audit/step4_enthalpy_prod_975_1050.failed
extension_file=$workspace/audit/step4_enthalpy_prod_975_1050.needs_extension

mkdir -p "$prod" "$analysis" "$workspace/audit"
rm -f "$done_file" "$failed_file" "$extension_file"
printf '%s enthalpy_prod_975_1050_started\n' "$(date -Iseconds)" > "$log"

mark_failed() {
    local rc=$?
    trap - ERR
    if [[ ! -e $done_file && ! -e $extension_file ]]; then
        printf '%s enthalpy_prod_975_1050_failed exit_code=%s\n' \
            "$(date -Iseconds)" "$rc" | tee -a "$log" "$failed_file"
    fi
    exit "$rc"
}
trap mark_failed ERR

test -x "$binary"
test -x "$mpirun"

source_975_solid=$root/T0975_preequil500_round2_preservevel/solid
source_975_liquid=$root/T0975_liquid_preequil500_round3_preservevel
source_1050_solid=$root/T1050_preequil500_round2_preservevel/solid
source_1050_liquid=$root/T1050_liquid_zeroP_fitted_confirmation500

python3 - \
    "$source_975_solid/confirmation_result.json" \
    "$source_975_liquid/phase_confirmation.json" \
    "$source_1050_solid/confirmation_result.json" \
    "$source_1050_liquid/phase_confirmation.json" <<'PY'
import json
import sys

for path in sys.argv[1:]:
    report = json.load(open(path, encoding="utf-8"))
    if report.get("status") not in {"confirmation_passed"}:
        raise SystemExit(f"source gate did not pass: {path}")
    checks = report.get("checks", {})
    if not checks or not all(checks.values()):
        raise SystemExit(f"source checks did not all pass: {path}")
PY

find_dump() {
    find "$1" -path '*/MD_dump' -type f | head -1
}

prepare_pair() {
    local temperature=$1
    local solid_source_root=$2
    local liquid_source_root=$3
    local solid_volume=$4
    local liquid_volume=$5
    local seed=$6
    local out=$prod/T${temperature}_steps3000
    local solid_dump
    local liquid_dump

    solid_dump=$(find_dump "$solid_source_root")
    liquid_dump=$(find_dump "$liquid_source_root")
    test -s "$solid_dump"
    test -s "$liquid_dump"
    if [[ ! -e $out/confirmation_manifest.json ]]; then
        env PYTHONPATH="$repo" python3 \
            "$repo/scripts/prepare_wt_zero_pressure_confirmation.py" \
            --out "$out" \
            --solid-source "$solid_dump" \
            --liquid-source "$liquid_dump" \
            --solid-volume "$solid_volume" \
            --liquid-volume "$liquid_volume" \
            --temperature "$((10#$temperature))" \
            --steps 3000 \
            --csvr-tau 5 \
            --seed "$seed" \
            --config "$config" >> "$log" 2>&1
    fi
}

prepare_pair 0975 "$source_975_solid" "$source_975_liquid" \
    18.08480608406171 18.891575304254044 2026081975
prepare_pair 1050 "$source_1050_solid" "$source_1050_liquid" \
    18.250457879770096 19.09530702583622 2026082050

points=(
    "$prod/T0975_steps3000/solid"
    "$prod/T0975_steps3000/liquid"
    "$prod/T1050_steps3000/solid"
    "$prod/T1050_steps3000/liquid"
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

confirmation_roots=("$prod/T0975_steps3000" "$prod/T1050_steps3000")
for pair in "${confirmation_roots[@]}"; do
    env PYTHONPATH="$repo" python3 \
        "$repo/scripts/analyze_wt_zero_pressure_confirmation.py" \
        "$pair" --temperature-tolerance 20 --pressure-tolerance 2.5 \
        >> "$log" 2>&1
    python3 - "$pair/confirmation_summary.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "all_confirmations_passed":
    raise SystemExit("3000-step phase pair did not pass")
PY
done

manifest=$analysis/manifest.json
env PYTHONPATH="$repo" python3 \
    "$repo/scripts/build_wt_fusion_enthalpy_manifest.py" \
    "${confirmation_roots[@]}" --out "$manifest" >> "$log" 2>&1

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
    find T0975_steps3000 T1050_steps3000 analysis_T0975_T1050 \
        -type f -print0 | sort -z | xargs -0 sha256sum
) > "$prod/SHA256SUMS"

status=$(python3 - "$convergence" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])
PY
)
if [[ $status != verified ]]; then
    printf '%s enthalpy_prod_975_1050_needs_extension status=%s\n' \
        "$(date -Iseconds)" "$status" | tee -a "$log" "$extension_file"
    trap - ERR
    exit 3
fi

date -Iseconds > "$done_file"
printf '%s enthalpy_prod_975_1050_verified\n' "$(date -Iseconds)" | tee -a "$log"
