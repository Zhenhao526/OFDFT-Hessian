#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_ROOT REPOSITORY" >&2
  exit 2
fi

run_root=$1
repository=$2
for phase in solid liquid; do
  for point in minus center plus; do
    [[ -x "$run_root/$phase/$point/run_local.sh" ]]
  done
done

run_point() {
  local cpus=$1
  local directory=$2
  (
    cd "$directory"
    exec taskset -c "$cpus" env OMP_NUM_THREADS=1 ./run_local.sh
  ) >"$directory/run.stdout" 2>&1
}

run_point 0-11 "$run_root/solid/minus" &
p1=$!
run_point 12-23 "$run_root/solid/center" &
p2=$!
run_point 24-35 "$run_root/solid/plus" &
p3=$!
run_point 38-49 "$run_root/liquid/minus" &
p4=$!
run_point 50-61 "$run_root/liquid/center" &
p5=$!
run_point 62-73 "$run_root/liquid/plus" &
p6=$!

wait "$p1"
wait "$p2"
wait "$p3"
wait "$p4"
wait "$p5"
wait "$p6"

cd "$repository"
for phase in solid liquid; do
  env PYTHONPATH=. python3 scripts/check_kedf_pressure_finite_difference.py \
    analyze "$run_root/$phase" | tee "$run_root/$phase/analysis.stdout"
done
find "$run_root" -type f \
  \( -name INPUT -o -name STRU -o -name KPT -o -name manifest.json \
     -o -name pressure_fd_result.json \) \
  -print0 | sort -z | xargs -0 sha256sum >"$run_root/SHA256SUMS"
touch "$run_root/pressure_pair.done"
