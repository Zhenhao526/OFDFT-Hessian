#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 RUN_ROOT LANE CPU_LIST" >&2
  exit 2
fi

root=$1
lane=$2
cpu_list=$3
manifest="$root/manifests/$lane.txt"
mpirun=/home/shenwei01/wt_melting_runtime_20260724/conda_prefix/bin/mpirun
abacus=/home/shenwei01/wt_melting_runtime_20260724/build-abacus-wt-cpu/source/abacus_pw_para

[[ -f "$manifest" ]] || { echo "missing manifest: $manifest" >&2; exit 2; }
mkdir -p "$root/lane_status"
status="$root/lane_status/$lane.tsv"
printf 'job\texit_code\n' > "$status"

while IFS= read -r rel; do
  [[ -n "$rel" ]] || continue
  job="$root/$rel"
  if [[ -f "$job/sp.done" ]]; then
    printf '%s\t0\n' "$rel" >> "$status"
    continue
  fi
  if find "$job" -maxdepth 1 -type d -name 'OUT.*' -print -quit | grep -q .; then
    echo "refusing pre-existing incomplete output: $job" >&2
    exit 3
  fi
  date --iso-8601=seconds > "$job/started_at.txt"
  (
    cd "$job"
    set +e
    "$mpirun" --map-by "pe-list=$cpu_list:ordered" --bind-to core -np 36 "$abacus" > run.stdout 2>&1 < /dev/null
    rc=$?
    set -e
    printf '%s\n' "$rc" > exit_code.txt
    date --iso-8601=seconds > finished_at.txt
    find . -maxdepth 1 -type f \( -name INPUT -o -name STRU -o -name KPT -o -name metadata.json \) -print0 | sort -z | xargs -0 sha256sum > RUN_INPUT_SHA256SUMS
    if [[ "$rc" -eq 0 ]] && grep -R -q 'FINAL_ETOT_IS' OUT.* 2>/dev/null; then
      touch sp.done
    else
      touch sp.failed
    fi
    exit "$rc"
  )
  rc=$?
  printf '%s\t%s\n' "$rel" "$rc" >> "$status"
  if [[ "$rc" -ne 0 ]]; then
    exit "$rc"
  fi
done < "$manifest"

touch "$root/lane_status/$lane.done"
