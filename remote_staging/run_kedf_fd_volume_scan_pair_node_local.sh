#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_ROOT REPOSITORY" >&2
  exit 2
fi

run_root=$1
repository=$2
mapfile -t jobs < <(
  find "$run_root" -mindepth 4 -maxdepth 4 -type f -name run_local.sh | sort
)
if [[ ${#jobs[@]} -ne 18 ]]; then
  echo "expected 18 finite-difference SCF jobs, found ${#jobs[@]}" >&2
  exit 2
fi

pids=()
for index in "${!jobs[@]}"; do
  start=$((index * 4))
  end=$((start + 3))
  directory=$(dirname "${jobs[$index]}")
  (
    cd "$directory"
    exec taskset -c "$start-$end" env OMP_NUM_THREADS=1 ./run_local.sh
  ) >"$directory/run.stdout" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

cd "$repository"
for phase in solid liquid; do
  for point in "$run_root/$phase"/vpa_*; do
    env PYTHONPATH=. python3 scripts/check_kedf_pressure_finite_difference.py \
      analyze "$point" | tee "$point/analysis.stdout"
  done
done
env PYTHONPATH=. python3 scripts/analyze_kedf_fd_volume_scan.py \
  "$run_root" | tee "$run_root/analysis.stdout"
find "$run_root" -type f \
  \( -name INPUT -o -name STRU -o -name KPT -o -name manifest.json \
     -o -name pressure_fd_result.json -o -name fd_volume_scan_summary.json \) \
  -print0 | sort -z | xargs -0 sha256sum >"$run_root/SHA256SUMS"
touch "$run_root/fd_volume_scan.done"
