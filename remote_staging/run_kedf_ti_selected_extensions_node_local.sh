#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 BATCH_ROOT REPOSITORY PAIR_DAT PHASE_RUN_ROOT..." >&2
  exit 2
fi

batch_root=$1
repository=$2
pair_dat=$3
shift 3
phase_roots=("$@")
log=$batch_root/extension_pipeline.log
if [[ -n ${KEDF_TI_CPU_RANGES:-} ]]; then
  read -r -a cpu_ranges <<<"$KEDF_TI_CPU_RANGES"
else
  cpu_ranges=(0-11 12-23 24-35 38-49 50-61 62-73)
fi

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
    for window in manifest["windows"]:
        print(root / window["label"])
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
    for window in manifest["windows"]:
        print(window["lambda"])
PY
)

if [[ ${#points[@]} -eq 0 || ${#points[@]} -ne ${#lambdas[@]} ]]; then
  echo "invalid selected-window batch" >&2
  exit 2
fi
slots_needed=$((${#points[@]} < 6 ? ${#points[@]} : 6))
if [[ ${#points[@]} -gt 1 && ${#cpu_ranges[@]} -lt $slots_needed ]]; then
  echo "insufficient KEDF_TI_CPU_RANGES entries" >&2
  exit 2
fi
for cpus in "${cpu_ranges[@]}"; do
  if [[ ! $cpus =~ ^[0-9]+-[0-9]+$ ]]; then
    echo "invalid CPU range: $cpus" >&2
    exit 2
  fi
done
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
  local index point lambda cpus status

  for ((slot=0; slot<count; slot++)); do
    index=$((first + slot))
    point=${points[$index]}
    lambda=${lambdas[$index]}
    if [[ ${#points[@]} -eq 1 && -z ${KEDF_TI_CPU_RANGES:-} ]]; then
      cpus=0-73
    else
      cpus=${cpu_ranges[$slot]}
    fi
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

  for ((slot=0; slot<count; slot++)); do
    index=$((first + slot))
    if wait "${pids[$slot]}"; then
      printf '%s finished point=%s\n' \
        "$(date -Iseconds)" "${points[$index]}" >>"$log"
    else
      status=$?
      printf '%s failed point=%s status=%s\n' \
        "$(date -Iseconds)" "${points[$index]}" "$status" >>"$log"
      touch "$batch_root/extension_pipeline.failed"
      return 1
    fi
  done
}

: >"$log"
for ((first=0; first<${#points[@]}; first+=6)); do
  remaining=$((${#points[@]} - first))
  count=$((remaining < 6 ? remaining : 6))
  run_wave "$first" "$count"
done

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
touch "$batch_root/extension_pipeline.completed"
