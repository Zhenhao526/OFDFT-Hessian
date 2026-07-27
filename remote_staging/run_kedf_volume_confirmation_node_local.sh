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

solid_cpus=${SOLID_CPUS:-0-35}
liquid_cpus=${LIQUID_CPUS:-38-73}
single_cpus=${SINGLE_CPUS:-0-73}
mapfile -t phases < <(
  python3 -c \
    'import json,sys; print("\n".join(x["phase"] for x in json.load(open(sys.argv[1]))["phases"]))' \
    "$run_root/confirmation_manifest.json"
)
if [[ ${#phases[@]} -eq 2 ]]; then
  run_phase "$solid_cpus" "${phases[0]}" &
  first_pid=$!
  run_phase "$liquid_cpus" "${phases[1]}" &
  second_pid=$!
  wait "$first_pid"
  wait "$second_pid"
elif [[ ${#phases[@]} -eq 1 ]]; then
  run_phase "$single_cpus" "${phases[0]}"
else
  echo "expected one or two phases, found ${#phases[@]}" >&2
  exit 2
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
  touch "$run_root/confirmation.failed"
  exit 1
fi
touch "$run_root/confirmation.done"
