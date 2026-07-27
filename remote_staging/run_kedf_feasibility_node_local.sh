#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_ROOT REPOSITORY" >&2
  exit 2
fi

run_root=$1
repository=$2
for phase in solid liquid; do
  [[ -x "$run_root/$phase/run_local.sh" ]]
done

run_phase() {
  local cpus=$1
  local phase=$2
  (
    cd "$run_root/$phase"
    exec taskset -c "$cpus" env OMP_NUM_THREADS=1 ./run_local.sh
  ) >"$run_root/$phase/run.stdout" 2>&1
}

run_phase 0-35 solid &
solid_pid=$!
run_phase 38-73 liquid &
liquid_pid=$!

wait "$solid_pid"
wait "$liquid_pid"

cd "$repository"
env PYTHONPATH=. python3 scripts/prepare_kedf_feasibility.py \
  analyze "$run_root" | tee "$run_root/analysis.stdout"

find "$run_root" -type f \
  \( -name INPUT -o -name STRU -o -name KPT -o -name metadata.json \
     -o -name feasibility_manifest.json -o -name feasibility_result.json \
     -o -name feasibility_summary.json -o -name phase_analysis.json \) \
  -print0 | sort -z | xargs -0 sha256sum >"$run_root/SHA256SUMS"

status=$(
  python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$run_root/feasibility_summary.json"
)
if [[ "$status" != "feasibility_verified" ]]; then
  touch "$run_root/feasibility.failed"
  exit 1
fi
touch "$run_root/feasibility.done"
