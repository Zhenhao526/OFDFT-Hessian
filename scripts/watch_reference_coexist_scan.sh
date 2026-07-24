#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 ROOT SPLIT [TEMPERATURE ...]" >&2
  exit 2
fi

root=$1
split=$2
shift 2
temperatures=("${@:-850 900 950 1000}")
python_bin=${PYTHON_BIN:-python3}
target_steps=${TARGET_STEPS:-5000}
poll_seconds=${POLL_SECONDS:-60}
log_file=${WATCH_LOG:-$root/watch_v4.log}

while true; do
  all_done=1
  line="$(date --iso-8601=seconds)"
  for temperature in "${temperatures[@]}"; do
    label=$(printf "%04d" "$temperature")
    run_dir="$root/T$label"
    md_log=$(find "$run_dir" -path '*/running_md.log' -type f -print -quit 2>/dev/null || true)
    steps=0
    if [[ -n "$md_log" ]]; then
      steps=$(grep -c 'PAIR_REFERENCE_COMPONENTS' "$md_log" || true)
    fi
    line+=" T$label=$steps/$target_steps"
    if (( steps < target_steps )); then
      all_done=0
    fi
  done
  printf '%s\n' "$line" >> "$log_file"
  if (( all_done )); then
    break
  fi
  sleep "$poll_seconds"
done

cd "$(dirname "$(dirname "$(realpath "$0")")")"
for temperature in "${temperatures[@]}"; do
  label=$(printf "%04d" "$temperature")
  run_dir="$root/T$label"
  PYTHONPATH=. "$python_bin" scripts/analyze_two_phase_run.py \
    "$run_dir" --split "$split" --out "$run_dir/two_phase_analysis.json" \
    >> "$log_file" 2>&1
done
printf '%s analysis_complete\n' "$(date --iso-8601=seconds)" >> "$log_file"
