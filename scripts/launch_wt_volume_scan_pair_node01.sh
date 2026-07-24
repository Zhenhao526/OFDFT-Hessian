#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/scratch/xzh/OFDFT-Hessian}
run_root=${1:?usage: launch_wt_volume_scan_pair_node01.sh RUN_ROOT}
binary=${ABACUS_BINARY:-$repo/external/abacus-develop/build-wt-ti-cpu2/source/abacus_wt_ti_cpu}
mpirun=${MPIRUN:-/scratch/xzh/conda/envs/abacus-gpu/bin/mpirun}
log=$run_root/run_wt_volume_scan_pair_node01.log

for phase in solid liquid; do
    if [[ ! -f $run_root/$phase/manifest.json ]]; then
        printf 'missing manifest for %s below %s\n' "$phase" "$run_root" >&2
        exit 2
    fi
done

mapfile -t points < <(
    python3 - "$run_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
temperatures = set()
for phase in ("solid", "liquid"):
    manifest = json.loads((root / phase / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("phase") != phase or manifest.get("target_kedf") != "wt":
        raise SystemExit(f"invalid {phase} manifest")
    temperatures.add(float(manifest["target_temperature_K"]))
    for point in manifest["points"]:
        print(root / phase / point["label"])
if len(temperatures) != 1:
    raise SystemExit("solid and liquid temperatures do not match")
PY
)

if ((${#points[@]} < 4 || ${#points[@]} > 6)); then
    printf 'expected 4-6 total scan points, found %s\n' "${#points[@]}" >&2
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
cpu_ranges=(0-11 12-23 24-35 38-49 50-61 62-73)
for slot in "${!points[@]}"; do
    point=${points[$slot]}
    cpus=${cpu_ranges[$slot]}
    (
        cd "$point"
        exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            taskset -c "$cpus" "$mpirun" -np 12 --bind-to none "$binary" \
            > run.stdout 2>&1
    ) &
    pids+=("$!")
    printf '%s started point=%s cpus=%s pid=%s\n' \
        "$(date -Iseconds)" "$point" "$cpus" "$!" | tee -a "$log"
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
for phase in solid liquid; do
    env PYTHONPATH="$repo" python3 scripts/prepare_al108_volume_scan.py analyze-md \
        "$run_root/$phase" >> "$log" 2>&1
done

python3 - "$run_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
for phase in ("solid", "liquid"):
    report = json.loads(
        (root / phase / "nvt_volume_scan_result.json").read_text(encoding="utf-8")
    )
    if report.get("status") != "zero_pressure_volume_verified":
        raise SystemExit(f"{phase} zero-pressure volume gate failed")
    if not all(report.get("checks", {}).values()):
        raise SystemExit(f"{phase} zero-pressure volume checks failed")
PY

printf '%s wt_volume_scan_pair_complete\n' "$(date -Iseconds)" | tee -a "$log"
