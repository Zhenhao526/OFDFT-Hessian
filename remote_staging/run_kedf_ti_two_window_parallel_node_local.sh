#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 BATCH_ROOT REPOSITORY PAIR_DAT SOLID_RUN_ROOT LIQUID_RUN_ROOT" >&2
  exit 2
fi

batch_root=$1
repository=$2
pair_dat=$3
solid_root=$4
liquid_root=$5
phase_roots=("$solid_root" "$liquid_root")
cpu_ranges=(0-35 38-73)
log=$batch_root/two_window_pipeline.log

[[ -f "$pair_dat" ]]
for root in "${phase_roots[@]}"; do
  [[ -f "$root/manifest.json" ]]
done

mapfile -t points < <(
  python3 - "${phase_roots[@]}" <<'PY'
import json
import sys
from pathlib import Path

for argument in sys.argv[1:]:
    root = Path(argument)
    manifest = json.loads((root / "manifest.json").read_text())
    windows = manifest.get("windows", [])
    if len(windows) != 1:
        raise SystemExit(f"{root}: expected exactly one window, found {len(windows)}")
    print(root / windows[0]["label"])
PY
)
mapfile -t lambdas < <(
  python3 - "${phase_roots[@]}" <<'PY'
import json
import sys
from pathlib import Path

for argument in sys.argv[1:]:
    root = Path(argument)
    manifest = json.loads((root / "manifest.json").read_text())
    windows = manifest.get("windows", [])
    if len(windows) != 1:
        raise SystemExit(f"{root}: expected exactly one window, found {len(windows)}")
    print(windows[0]["lambda"])
PY
)

if [[ ${#points[@]} -ne 2 || ${#lambdas[@]} -ne 2 ]]; then
  echo "invalid two-window batch" >&2
  exit 2
fi

for point in "${points[@]}"; do
  [[ -x "$point/run_local.sh" ]]
  if find "$point" -maxdepth 1 -type d -name 'OUT.*' | grep -q .; then
    echo "refusing existing output below $point" >&2
    exit 2
  fi
done

: >"$log"
pids=()
for index in 0 1; do
  point=${points[$index]}
  lambda=${lambdas[$index]}
  cpus=${cpu_ranges[$index]}
  (
    cd "$point"
    exec taskset -c "$cpus" env \
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      MPN_TI_LAMBDA="$lambda" MPN_TI_PAIR_MODEL="$pair_dat" \
      ./run_local.sh
  ) >"$point/run.stdout" 2>&1 &
  pids+=("$!")
  printf '%s started point=%s lambda=%s cpus=%s pid=%s\n' \
    "$(date -Iseconds)" "$point" "$lambda" "$cpus" "$!" >>"$log"
done

failed=0
for index in 0 1; do
  if wait "${pids[$index]}"; then
    printf '%s finished point=%s\n' \
      "$(date -Iseconds)" "${points[$index]}" >>"$log"
  else
    status=$?
    failed=1
    printf '%s failed point=%s status=%s\n' \
      "$(date -Iseconds)" "${points[$index]}" "$status" >>"$log"
  fi
done

if [[ $failed -ne 0 ]]; then
  touch "$batch_root/two_window_pipeline.failed"
  exit 1
fi

cd "$repository"
for root in "${phase_roots[@]}"; do
  env PYTHONPATH=. python3 scripts/prepare_al108_ti_windows.py \
    analyze "$root" >>"$log" 2>&1
done
find "$batch_root" -type f \
  \( -name INPUT -o -name STRU -o -name KPT -o -name metadata.json \
     -o -name manifest.json -o -name phase_analysis.json \
     -o -name ti_analysis.json \) \
  -print0 | sort -z | xargs -0 sha256sum >"$batch_root/SHA256SUMS"
touch "$batch_root/two_window_pipeline.completed"
