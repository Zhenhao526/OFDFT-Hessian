#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_ROOT REPOSITORY" >&2
  exit 2
fi

run_root=$1
repository=$2
IFS=',' read -r -a cpu_ranges <<<"${CPU_RANGES:-0-35,38-73}"
log=$run_root/enthalpy_pipeline.log

[[ -f "$run_root/confirmation_manifest.json" ]]
[[ -f "$run_root/INPUT_SHA256SUMS" ]]
(cd "$run_root" && sha256sum -c INPUT_SHA256SUMS)

mapfile -t phases < <(
  python3 - "$run_root/confirmation_manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
if manifest.get("target_kedf") not in {"xwm", "lkt"}:
    raise SystemExit("unsupported target KEDF")
phases = [item["phase"] for item in manifest["phases"]]
if sorted(phases) != ["liquid", "solid"]:
    raise SystemExit("expected exactly one solid and one liquid phase")
print("\n".join(phases))
PY
)

if [[ ${#phases[@]} -ne 2 || ${#cpu_ranges[@]} -ne 2 ]]; then
  echo "expected two phases and two CPU ranges" >&2
  exit 2
fi

points=()
for phase in "${phases[@]}"; do
  point=$run_root/$phase
  [[ -x "$point/run_local.sh" ]]
  if find "$point" -maxdepth 1 -type d -name 'OUT.*' | grep -q .; then
    echo "refusing existing output below $point" >&2
    exit 2
  fi
  points+=("$point")
done

: >"$log"
pids=()
for index in "${!points[@]}"; do
  point=${points[$index]}
  cpus=${cpu_ranges[$index]}
  (
    cd "$point"
    exec taskset -c "$cpus" env \
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      ./run_local.sh
  ) >"$point/run.stdout" 2>&1 &
  pids+=("$!")
  printf '%s started phase=%s point=%s cpus=%s pid=%s\n' \
    "$(date -Iseconds)" "${phases[$index]}" "$point" "$cpus" "$!" \
    >>"$log"
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    printf '%s finished phase=%s point=%s\n' \
      "$(date -Iseconds)" "${phases[$index]}" "${points[$index]}" \
      >>"$log"
  else
    status=$?
    printf '%s failed phase=%s point=%s status=%s\n' \
      "$(date -Iseconds)" "${phases[$index]}" "${points[$index]}" "$status" \
      >>"$log"
    failed=1
  fi
done
if ((failed)); then
  touch "$run_root/enthalpy_pipeline.failed"
  exit 1
fi

cd "$repository"
env PYTHONPATH=. python3 scripts/prepare_kedf_volume_confirmation.py \
  analyze "$run_root" | tee "$run_root/analysis.stdout"

find "$run_root" -type f \
  \( -name INPUT -o -name STRU -o -name KPT -o -name metadata.json \
     -o -name confirmation_manifest.json -o -name confirmation_result.json \
     -o -name confirmation_summary.json -o -name phase_analysis.json \) \
  -print0 | sort -z | xargs -0 sha256sum >"$run_root/SHA256SUMS"

status=$(
  python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$run_root/confirmation_summary.json"
)
if [[ "$status" != "volume_confirmation_verified" ]]; then
  touch "$run_root/enthalpy_pipeline.physical_failed"
  exit 1
fi
touch "$run_root/enthalpy_pipeline.physical_verified"
