#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 METHOD RUN_ROOT REPOSITORY" >&2
  exit 2
fi

method=$1
run_root=$2
repository=$3

case "$method" in
  xwm|lkt) ;;
  *)
    echo "unsupported KEDF: $method" >&2
    exit 2
    ;;
esac

force_root="$run_root/force_fd_d005"
for point in minus center plus; do
  [[ -x "$force_root/$point/run_local.sh" ]]
done
if [[ "$method" == "lkt" ]]; then
  pressure_root="$run_root/pressure_fd_solid975"
  for point in minus center plus; do
    [[ -x "$pressure_root/$point/run_local.sh" ]]
  done
fi

run_point() {
  local cpus=$1
  local directory=$2
  (
    cd "$directory"
    exec taskset -c "$cpus" env OMP_NUM_THREADS=1 ./run_local.sh
  ) >"$directory/run.stdout" 2>&1
}

run_point 0-7 "$force_root/minus" &
force_minus_pid=$!
run_point 8-15 "$force_root/center" &
force_center_pid=$!
run_point 16-23 "$force_root/plus" &
force_plus_pid=$!

pressure_pids=()
if [[ "$method" == "lkt" ]]; then
  run_point 24-35 "$pressure_root/minus" &
  pressure_pids+=("$!")
  run_point 36-47 "$pressure_root/center" &
  pressure_pids+=("$!")
  run_point 48-59 "$pressure_root/plus" &
  pressure_pids+=("$!")
fi

wait "$force_minus_pid"
wait "$force_center_pid"
wait "$force_plus_pid"
for pid in "${pressure_pids[@]}"; do
  wait "$pid"
done

cd "$repository"
env PYTHONPATH=. python3 scripts/check_mpn_force_finite_difference.py \
  analyze "$force_root" | tee "$force_root/analysis.stdout"
if [[ "$method" == "lkt" ]]; then
  env PYTHONPATH=. python3 scripts/check_mpn_pressure_finite_difference.py \
    analyze "$pressure_root" | tee "$pressure_root/analysis.stdout"
fi

find "$run_root" -type f \
  \( -name INPUT -o -name STRU -o -name KPT -o -name manifest.json \
     -o -name force_fd_result.json -o -name pressure_fd_result.json \) \
  -print0 | sort -z | xargs -0 sha256sum >"$run_root/SHA256SUMS"
touch "$run_root/fd_validation.done"
