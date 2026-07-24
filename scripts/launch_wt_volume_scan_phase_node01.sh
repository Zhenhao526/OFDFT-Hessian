#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/scratch/xzh/OFDFT-Hessian}
run_root=${1:?usage: launch_wt_volume_scan_phase_node01.sh RUN_ROOT PHASE [MPI_RANKS]}
phase=${2:?usage: launch_wt_volume_scan_phase_node01.sh RUN_ROOT PHASE [MPI_RANKS]}
ranks=${3:-24}
binary=${ABACUS_BINARY:-$repo/external/abacus-develop/build-wt-ti-cpu2/source/abacus_wt_ti_cpu}
mpirun=${MPIRUN:-/scratch/xzh/conda/envs/abacus-gpu/bin/mpirun}
phase_root=$run_root/$phase
log=$run_root/run_${phase}_volume_scan_phase_node01.log

if [[ $phase != solid && $phase != liquid ]]; then
    printf 'phase must be solid or liquid, got %s\n' "$phase" >&2
    exit 2
fi
if [[ ! -f $phase_root/manifest.json ]]; then
    printf 'missing manifest below %s\n' "$phase_root" >&2
    exit 2
fi

mapfile -t points < <(
    python3 - "$phase_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
for point in manifest["points"]:
    print(root / point["label"])
PY
)
if ((${#points[@]} < 2 || ${#points[@]} > 3)); then
    printf 'expected 2-3 points, found %s\n' "${#points[@]}" >&2
    exit 2
fi
if ((ranks < 1 || ranks * ${#points[@]} > 72)); then
    printf 'invalid rank allocation: ranks=%s points=%s\n' "$ranks" "${#points[@]}" >&2
    exit 2
fi
for point in "${points[@]}"; do
    if [[ ! -f $point/INPUT || ! -f $point/STRU ]]; then
        printf 'missing prepared input below %s\n' "$point" >&2
        exit 2
    fi
    if find "$point" -maxdepth 1 -type d -name 'OUT.*' | grep -q .; then
        printf 'refusing existing output below %s\n' "$point" >&2
        exit 2
    fi
done

: > "$log"
pids=()
if ((ranks == 36 && ${#points[@]} == 2)); then
    cpu_ranges=(0-35 38-73)
elif ((ranks == 12 && ${#points[@]} == 3)); then
    cpu_ranges=(0-11 12-23 38-49)
elif ((ranks == 12 && ${#points[@]} == 2)); then
    cpu_ranges=(0-11 38-49)
else
    cpu_ranges=()
fi
for slot in "${!points[@]}"; do
    point=${points[$slot]}
    if ((${#cpu_ranges[@]})); then
        cpus=${cpu_ranges[$slot]}
    else
        lo=$((slot * ranks))
        hi=$((lo + ranks - 1))
        cpus=$lo-$hi
    fi
    (
        cd "$point"
        exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            taskset -c "$cpus" "$mpirun" -np "$ranks" --bind-to none "$binary" \
            > run.stdout 2>&1
    ) &
    pids+=("$!")
    printf '%s started point=%s cpus=%s ranks=%s pid=%s\n' \
        "$(date -Iseconds)" "$point" "$cpus" "$ranks" "$!" | tee -a "$log"
done

failed=0
for slot in "${!pids[@]}"; do
    if wait "${pids[$slot]}"; then
        printf '%s finished point=%s\n' \
            "$(date -Iseconds)" "${points[$slot]}" | tee -a "$log"
    else
        rc=$?
        printf '%s failed point=%s exit_code=%s\n' \
            "$(date -Iseconds)" "${points[$slot]}" "$rc" | tee -a "$log"
        failed=1
    fi
done
if ((failed != 0)); then
    exit 2
fi

cd "$repo"
env PYTHONPATH="$repo" python3 scripts/prepare_al108_volume_scan.py analyze-md "$phase_root" \
    >> "$log" 2>&1
python3 - "$phase_root/nvt_volume_scan_result.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "zero_pressure_volume_verified":
    raise SystemExit("zero-pressure volume scan is not verified")
if not report.get("checks") or not all(report["checks"].values()):
    raise SystemExit("zero-pressure volume checks failed")
PY
printf '%s phase_volume_scan_verified phase=%s root=%s\n' \
    "$(date -Iseconds)" "$phase" "$phase_root" | tee -a "$log"
