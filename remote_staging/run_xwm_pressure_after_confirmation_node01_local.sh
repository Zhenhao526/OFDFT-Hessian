#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 CONFIRMATION_ROOT PRESSURE_ROOT REPOSITORY CONFIG" >&2
  exit 2
fi

confirmation_root=$1
pressure_root=$2
repository=$3
config=$4
temperature_K=${TEMPERATURE_K:-975}
pressure_cpu_start=${PRESSURE_CPU_START:-0}

if [[ -e "$pressure_root" ]]; then
  echo "refusing to overwrite $pressure_root" >&2
  exit 2
fi

deadline=$((SECONDS + 3600))
while [[ ! -f "$confirmation_root/confirmation.done" ]]; do
  if [[ -f "$confirmation_root/confirmation.failed" ]]; then
    echo "volume confirmation failed; pressure sampling not started" >&2
    exit 1
  fi
  if ((SECONDS >= deadline)); then
    echo "timed out waiting for volume confirmation" >&2
    exit 1
  fi
  sleep 15
done

status=$(
  python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$confirmation_root/confirmation_summary.json"
)
if [[ "$status" != "volume_confirmation_verified" ]]; then
  echo "unexpected confirmation status: $status" >&2
  exit 1
fi

cd "$repository"
for phase in solid liquid; do
  mapfile -t dumps < <(
    find "$confirmation_root/$phase" -type f -name MD_dump | sort
  )
  if [[ ${#dumps[@]} -ne 1 ]]; then
    echo "expected one MD_dump for $phase, found ${#dumps[@]}" >&2
    exit 1
  fi
  env PYTHONPATH=. python3 scripts/prepare_kedf_pressure_samples.py prepare \
    --out "$pressure_root/$phase" \
    --source "${dumps[0]}" \
    --config "$config" \
    --phase "$phase" \
    --temperature "$temperature_K" \
    --frame-indices 40 45 50 55 59 \
    --linear-strain 0.001 \
    --ranks 4
done

PRESSURE_CPU_START="$pressure_cpu_start" \
  bash remote_staging/run_kedf_pressure_samples_pair_node_local.sh \
  "$pressure_root" "$repository"

env PYTHONPATH=. python3 scripts/finalize_kedf_zero_pressure_confirmation.py \
  --confirmation-root "$confirmation_root" \
  --pressure-samples-root "$pressure_root" \
  --out "$confirmation_root/zero_pressure_confirmation_summary.json" \
  >"$confirmation_root/finalize_zero_pressure.stdout"

final_status=$(
  python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$confirmation_root/zero_pressure_confirmation_summary.json"
)
if [[ "$final_status" != "all_confirmations_passed" ]]; then
  touch "$pressure_root/pressure_samples.needs_adjustment"
  exit 1
fi
touch "$pressure_root/pressure_samples.verified"
