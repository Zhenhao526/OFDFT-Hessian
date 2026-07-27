#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_ROOT REPOSITORY" >&2
  exit 2
fi

run_root=$1
repository=$2
cpu_start=${PRESSURE_CPU_START:-0}
mapfile -t jobs < <(
  find "$run_root" -mindepth 4 -maxdepth 4 -type f -name run_local.sh | sort
)
if [[ ${#jobs[@]} -ne 15 && ${#jobs[@]} -ne 30 ]]; then
  echo "expected 15 or 30 snapshot-pressure SCF jobs, found ${#jobs[@]}" >&2
  exit 2
fi

for ((batch_start = 0; batch_start < ${#jobs[@]}; batch_start += 18)); do
  pids=()
  batch_end=$((batch_start + 18))
  if ((batch_end > ${#jobs[@]})); then
    batch_end=${#jobs[@]}
  fi
  for ((index = batch_start; index < batch_end; index++)); do
    slot=$((index - batch_start))
    start=$((cpu_start + slot * 4))
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
done

cd "$repository"
mapfile -t phases < <(
  find "$run_root" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
)
for phase in "${phases[@]}"; do
  env PYTHONPATH=. python3 scripts/prepare_kedf_pressure_samples.py \
    analyze "$run_root/$phase" | tee "$run_root/$phase/analysis.stdout"
done
find "$run_root" -type f \
  \( -name INPUT -o -name STRU -o -name KPT -o -name manifest.json \
     -o -name samples_manifest.json -o -name pressure_fd_result.json \
     -o -name pressure_samples_summary.json \) \
  -print0 | sort -z | xargs -0 sha256sum >"$run_root/SHA256SUMS"
touch "$run_root/pressure_samples.done"
