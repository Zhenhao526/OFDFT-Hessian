#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 && $# -ne 4 ]]; then
  echo "usage: $0 PILOT_ROOT REPOSITORY SOLID_PAIR_DAT [LIQUID_PAIR_DAT]" >&2
  exit 2
fi

pilot_root=$1
repository=$2
solid_pair_dat=$3
liquid_pair_dat=${4:-$solid_pair_dat}
log=$pilot_root/pilot_pipeline.log
cpu_ranges=(0-11 12-23 24-35 38-49 50-61 62-73)

[[ -f "$pilot_root/pilot_manifest.json" ]]
[[ -f "$solid_pair_dat" ]]
[[ -f "$liquid_pair_dat" ]]
for phase in solid liquid; do
  [[ -f "$pilot_root/$phase/manifest.json" ]]
done

mapfile -t points < <(
  python3 - "$pilot_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for phase in ("solid", "liquid"):
    manifest = json.loads((root / phase / "manifest.json").read_text())
    for window in manifest["windows"]:
        print(root / phase / window["label"])
PY
)
mapfile -t lambdas < <(
  python3 - "$pilot_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for phase in ("solid", "liquid"):
    manifest = json.loads((root / phase / "manifest.json").read_text())
    for window in manifest["windows"]:
        print(window["lambda"])
PY
)

if [[ ${#points[@]} -ne 18 || ${#lambdas[@]} -ne 18 ]]; then
  echo "expected 18 phase/lambda windows" >&2
  exit 2
fi
for point in "${points[@]}"; do
  [[ -x "$point/run_local.sh" ]]
  if find "$point" -maxdepth 1 -type d -name 'OUT.*' | grep -q .; then
    echo "refusing existing output below $point" >&2
    exit 2
  fi
done

run_wave() {
  local first=$1
  local count=$2
  local pids=()
  local index point lambda cpus status phase_pair_dat

  for ((slot=0; slot<count; slot++)); do
    index=$((first + slot))
    point=${points[$index]}
    lambda=${lambdas[$index]}
    cpus=${cpu_ranges[$slot]}
    phase_pair_dat=$solid_pair_dat
    ((index >= 9)) && phase_pair_dat=$liquid_pair_dat
    (
      cd "$point"
      exec taskset -c "$cpus" env \
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      MPN_TI_LAMBDA="$lambda" MPN_TI_PAIR_MODEL="$phase_pair_dat" \
        ./run_local.sh
    ) >"$point/run.stdout" 2>&1 &
    pids+=("$!")
    printf '%s started point=%s lambda=%s cpus=%s pid=%s\n' \
      "$(date -Iseconds)" "$point" "$lambda" "$cpus" "$!" >>"$log"
  done

  for ((slot=0; slot<count; slot++)); do
    index=$((first + slot))
    if wait "${pids[$slot]}"; then
      printf '%s finished point=%s\n' \
        "$(date -Iseconds)" "${points[$index]}" >>"$log"
    else
      status=$?
      printf '%s failed point=%s status=%s\n' \
        "$(date -Iseconds)" "${points[$index]}" "$status" >>"$log"
      touch "$pilot_root/pilot_pipeline.failed"
      return 1
    fi
  done
}

: >"$log"
run_wave 0 6
run_wave 6 6
run_wave 12 6

cd "$repository"
for phase in solid liquid; do
  env PYTHONPATH=. python3 scripts/prepare_al108_ti_windows.py \
    analyze "$pilot_root/$phase" >>"$log" 2>&1
done
find "$pilot_root" -type f \
  \( -name INPUT -o -name STRU -o -name KPT -o -name metadata.json \
     -o -name manifest.json -o -name pilot_manifest.json \
     -o -name phase_analysis.json -o -name ti_analysis.json \) \
  -print0 | sort -z | xargs -0 sha256sum >"$pilot_root/SHA256SUMS"
touch "$pilot_root/pilot_pipeline.completed"
