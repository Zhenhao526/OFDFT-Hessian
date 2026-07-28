#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 REPOSITORY RUN_ROOT [RUN_ROOT ...]" >&2
  exit 2
fi

repository=$1
shift
run_roots=("$@")
IFS=',' read -r -a cpu_ranges <<<"${CPU_RANGES:-0-35,38-55,56-73}"
points=()
point_roots=()
phases=()
ranks=()

cpu_count() {
  python3 - "$1" <<'PY'
import sys

cpus = set()
for field in sys.argv[1].split(","):
    bounds = [int(item) for item in field.split("-", 1)]
    if len(bounds) == 1:
        cpus.add(bounds[0])
    else:
        cpus.update(range(bounds[0], bounds[1] + 1))
print(len(cpus))
PY
}

for root in "${run_roots[@]}"; do
  [[ -f "$root/confirmation_manifest.json" ]]
  [[ -f "$root/INPUT_SHA256SUMS" ]]
  [[ ! -e "$root/confirmation.done" ]]
  [[ ! -e "$root/confirmation.failed" ]]
  (cd "$root" && sha256sum -c INPUT_SHA256SUMS)
  while IFS=$'\t' read -r phase rank; do
    point=$root/$phase
    [[ -x "$point/run_local.sh" ]]
    if find "$point" -maxdepth 1 -type d -name 'OUT.*' | grep -q .; then
      echo "refusing existing output below $point" >&2
      exit 2
    fi
    points+=("$point")
    point_roots+=("$root")
    phases+=("$phase")
    ranks+=("$rank")
  done < <(
    python3 - "$root/confirmation_manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
if manifest.get("target_kedf") not in {"xwm", "lkt"}:
    raise SystemExit("unsupported target KEDF")
for row in manifest["phases"]:
    metadata = json.load(open(f'{row["run"]}/metadata.json'))
    print(f'{row["phase"]}\t{metadata["mpi_ranks"]}')
PY
  )
done

if (( ${#points[@]} != ${#cpu_ranges[@]} )); then
  echo "expected ${#points[@]} CPU ranges, found ${#cpu_ranges[@]}" >&2
  exit 2
fi

for index in "${!points[@]}"; do
  assigned=$(cpu_count "${cpu_ranges[$index]}")
  if (( assigned != ranks[index] )); then
    echo "CPU range ${cpu_ranges[$index]} has $assigned CPUs; ${points[$index]} requires ${ranks[$index]}" >&2
    exit 2
  fi
done

batch_log=${BATCH_LOG:-${run_roots[0]}/cpu_volume_batch.log}
: >"$batch_log"
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
  printf '%s started root=%s phase=%s point=%s cpus=%s ranks=%s pid=%s\n' \
    "$(date -Iseconds)" "${point_roots[$index]}" "${phases[$index]}" \
    "$point" "$cpus" "${ranks[$index]}" "$!" >>"$batch_log"
done

runtime_failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    printf '%s finished root=%s phase=%s point=%s\n' \
      "$(date -Iseconds)" "${point_roots[$index]}" \
      "${phases[$index]}" "${points[$index]}" >>"$batch_log"
  else
    status=$?
    printf '%s failed root=%s phase=%s point=%s status=%s\n' \
      "$(date -Iseconds)" "${point_roots[$index]}" \
      "${phases[$index]}" "${points[$index]}" "$status" >>"$batch_log"
    runtime_failed=1
  fi
done
if (( runtime_failed )); then
  for root in "${run_roots[@]}"; do
    touch "$root/confirmation.failed"
  done
  exit 1
fi

cd "$repository"
physical_failed=0
for root in "${run_roots[@]}"; do
  env PYTHONPATH=. python3 scripts/prepare_kedf_volume_confirmation.py \
    analyze "$root" | tee "$root/analysis.stdout"
  find "$root" -type f \
    \( -name INPUT -o -name STRU -o -name KPT -o -name metadata.json \
       -o -name confirmation_manifest.json -o -name confirmation_result.json \
       -o -name confirmation_summary.json -o -name phase_analysis.json \) \
    -print0 | sort -z | xargs -0 sha256sum >"$root/SHA256SUMS"
  status=$(
    python3 -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
      "$root/confirmation_summary.json"
  )
  if [[ "$status" == "volume_confirmation_verified" ]]; then
    touch "$root/confirmation.done"
  else
    touch "$root/confirmation.failed"
    physical_failed=1
  fi
done
exit "$physical_failed"
