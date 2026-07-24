#!/usr/bin/env bash
set -euo pipefail

workspace=${WORKSPACE:-/home/shenwei01/WT_Al_melting_workspace_20260724}
repo=${REPO:-$workspace/repository}
runtime=${RUNTIME:-/home/shenwei01/wt_melting_runtime_20260724}
root=${RUN_ROOT:-$workspace/runs/step4_rebuild}
scan=$root/T1050_liquid_volume_rescan_from_round3
source_root=$scan/vpa_19p120
out=$root/T1050_liquid_zeroP_fitted_confirmation500
config=$repo/config/abacus_wt_node01_local_cpu18.json
binary=$runtime/build-abacus-wt-cpu/source/abacus_pw_para
mpirun=$runtime/conda_prefix/bin/mpirun
log=$workspace/audit/step4_T1050_liquid_fitted_confirmation.log
done_file=$workspace/audit/step4_T1050_liquid_fitted_confirmation.done
failed_file=$workspace/audit/step4_T1050_liquid_fitted_confirmation.failed

mkdir -p "$workspace/audit"
rm -f "$done_file" "$failed_file"
printf '%s T1050_liquid_fitted_confirmation_started\n' "$(date -Iseconds)" > "$log"

mark_failed() {
    local rc=$?
    trap - ERR
    if [[ ! -e $done_file ]]; then
        printf '%s T1050_liquid_fitted_confirmation_failed exit_code=%s\n' \
            "$(date -Iseconds)" "$rc" | tee -a "$log" "$failed_file"
    fi
    exit "$rc"
}
trap mark_failed ERR

test -s "$scan/volume_scan_summary.json"
volume=$(python3 - "$scan/volume_scan_summary.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "zero_pressure_volume_verified":
    raise SystemExit("volume scan is not verified")
print(f"{report['linear_zero_pressure_volume_A3_per_atom']:.12f}")
PY
)
source=$(find "$source_root" -path '*/MD_dump' -type f | head -1)
test -s "$source"

if [[ ! -e $out/phase_continuation_manifest.json ]]; then
    env PYTHONPATH="$repo" python3 \
        "$repo/scripts/prepare_wt_phase_continuation.py" \
        --source "$source" --out "$out" --phase liquid \
        --volume "$volume" --temperature 1050 \
        --steps 500 --csvr-tau 2 --seed 2026081050 --config "$config" \
        >> "$log" 2>&1
fi

test -s "$out/INPUT"
test -s "$out/STRU"
test -s "$out/KPT"
if find "$out" -maxdepth 1 -type d -name 'OUT.*' | grep -q .; then
    printf '%s refusing_existing_output out=%s\n' \
        "$(date -Iseconds)" "$out" | tee -a "$log"
    exit 2
fi

printf '%s started out=%s volume_A3_per_atom=%s\n' \
    "$(date -Iseconds)" "$out" "$volume" | tee -a "$log"
(
    cd "$out"
    exec env \
        PATH="$runtime/conda_prefix/bin:$PATH" \
        LD_LIBRARY_PATH="$runtime/conda_prefix/lib:${LD_LIBRARY_PATH:-}" \
        OMP_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 \
        taskset -c 0-35 "$mpirun" -np 36 --bind-to none "$binary" \
        > run.stdout 2>&1
)

env PYTHONPATH="$repo" python3 \
    "$repo/scripts/analyze_wt_phase_confirmation.py" "$out" \
    --expected liquid --temperature 1050 --pressure 0 --steps 500 \
    --temperature-tolerance 20 --pressure-tolerance 2.5 \
    --out "$out/phase_confirmation.json" >> "$log" 2>&1

python3 - "$out/phase_confirmation.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "confirmation_passed":
    raise SystemExit("fitted-volume confirmation did not pass")
PY

(
    cd "$out"
    find . -type f -print0 | sort -z | xargs -0 sha256sum
) > "$out/SHA256SUMS"
date -Iseconds > "$done_file"
printf '%s T1050_liquid_fitted_confirmation_verified\n' "$(date -Iseconds)" \
    | tee -a "$log"
