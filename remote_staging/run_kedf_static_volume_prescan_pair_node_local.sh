#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_ROOT REPOSITORY" >&2
  exit 2
fi

run_root=$1
repository=$2
mapfile -t solid_labels < <(
  python3 -c \
    'import json,sys; print(*[p["label"] for p in json.load(open(sys.argv[1]))["points"]], sep="\n")' \
    "$run_root/solid/manifest.json"
)
mapfile -t liquid_labels < <(
  python3 -c \
    'import json,sys; print(*[p["label"] for p in json.load(open(sys.argv[1]))["points"]], sep="\n")' \
    "$run_root/liquid/manifest.json"
)
if [[ ${#solid_labels[@]} -ne 3 || ${#liquid_labels[@]} -ne 3 ]]; then
  echo "launcher requires exactly three volume points per phase" >&2
  exit 2
fi

run_point() {
  local cpus=$1
  local directory=$2
  (
    cd "$directory"
    exec taskset -c "$cpus" env OMP_NUM_THREADS=1 ./run_local.sh
  ) >"$directory/run.stdout" 2>&1
}

run_point 0-11 "$run_root/solid/${solid_labels[0]}" &
p1=$!
run_point 12-23 "$run_root/solid/${solid_labels[1]}" &
p2=$!
run_point 24-35 "$run_root/solid/${solid_labels[2]}" &
p3=$!
run_point 38-49 "$run_root/liquid/${liquid_labels[0]}" &
p4=$!
run_point 50-61 "$run_root/liquid/${liquid_labels[1]}" &
p5=$!
run_point 62-73 "$run_root/liquid/${liquid_labels[2]}" &
p6=$!

wait "$p1"
wait "$p2"
wait "$p3"
wait "$p4"
wait "$p5"
wait "$p6"

cd "$repository"
for phase in solid liquid; do
  env PYTHONPATH=. python3 scripts/prepare_al108_volume_scan.py \
    analyze "$run_root/$phase" | tee "$run_root/$phase/analysis.stdout"
done
find "$run_root" -type f \
  \( -name INPUT -o -name STRU -o -name KPT -o -name manifest.json \
     -o -name volume_prescan_result.json \) \
  -print0 | sort -z | xargs -0 sha256sum >"$run_root/SHA256SUMS"
touch "$run_root/static_prescan.done"
