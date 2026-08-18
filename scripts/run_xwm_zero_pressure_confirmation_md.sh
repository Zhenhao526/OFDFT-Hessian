#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_DIR CPU_LIST" >&2
  exit 2
fi

run_dir=$1
cpu_list=$2
mpirun=/home/shenwei01/wt_melting_runtime_20260724/conda_prefix/bin/mpirun
abacus=/home/shenwei01/wt_melting_runtime_20260724/build-abacus-wt-cpu/source/abacus_pw_para

if [[ -e "$run_dir/md.done" || -e "$run_dir/md.failed" ]] || find "$run_dir" -maxdepth 1 -type d -name 'OUT.*' -print -quit | grep -q .; then
  echo "refusing an already started run: $run_dir" >&2
  exit 3
fi

cd "$run_dir"
date --iso-8601=seconds > started_at.txt
find . -maxdepth 1 -type f \( -name INPUT -o -name STRU -o -name KPT -o -name metadata.json -o -name mg_phase_md_manifest.json \) -print0 \
  | sort -z | xargs -0 sha256sum > RUN_INPUT_SHA256SUMS

set +e
"$mpirun" --map-by "pe-list=$cpu_list:ordered" --bind-to core -np 36 "$abacus" > run.stdout 2>&1 < /dev/null
rc=$?
set -e

printf '%s\n' "$rc" > exit_code.txt
date --iso-8601=seconds > finished_at.txt
if [[ "$rc" -eq 0 ]]; then
  touch md.done
else
  touch md.failed
fi
exit "$rc"
