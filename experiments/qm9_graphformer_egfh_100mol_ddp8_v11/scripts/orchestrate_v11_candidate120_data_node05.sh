#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname -s)" != "node05" ]]; then
  echo "ERROR: v11 data orchestrator is bound to node05" >&2
  exit 2
fi

root=/home/shenwei01/xzh_node02_20260724
egf_run="$root/runs/qm9_graphformer_egfh_100mol_ddp8_v11/labelgen_candidate120"
hessian_run="$root/runs/qm9_graphformer_egfh_100mol_ddp8_v11/hessian_candidate120_ddp8"
work="$root/work/qm9_graphformer_egfh_100mol_ddp8_v11"
run_dir="$root/runs/qm9_graphformer_egfh_100mol_ddp8_v11/data_orchestrator"
expected_egf_pid=3086323

for path_value in "$root" "$egf_run" "$hessian_run" "$work" "$run_dir"; do
  if [[ "$path_value" == /scratch* ]]; then
    echo "ERROR: shared scratch paths are forbidden for v11" >&2
    exit 3
  fi
done

mkdir -p "$run_dir"
status_file="$run_dir/status.txt"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then printf "failed rc=%s at=%s\n" "$rc" "$(date --iso-8601=seconds)" > "$status_file"; fi' EXIT
printf 'running stage=wait_egf at=%s\n' "$(date --iso-8601=seconds)" > "$status_file"
date --iso-8601=seconds > "$run_dir/started_at.txt"

while true; do
  egf_status=$(cat "$egf_run/status.txt" 2>/dev/null || true)
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$egf_status" >> "$run_dir/wait.log"
  if [[ "$egf_status" == complete* ]]; then
    break
  fi
  if [[ "$egf_status" == failed* ]]; then
    echo "ERROR: E/G/F generation failed: $egf_status" >&2
    exit 4
  fi
  if ! kill -0 "$expected_egf_pid" 2>/dev/null; then
    echo "ERROR: E/G/F generator disappeared without completing" >&2
    exit 5
  fi
  sleep 30
done

if [[ -e "$hessian_run/status.txt" ]]; then
  echo "ERROR: Hessian run status already exists; refusing duplicate launch" >&2
  exit 6
fi
printf 'running stage=hessian_ddp8 at=%s\n' "$(date --iso-8601=seconds)" > "$status_file"
"$work/scripts/run_v11_candidate120_hessian_ddp8_node05.sh" \
  > "$run_dir/hessian_launcher.log" 2>&1

if [[ "$(cat "$hessian_run/status.txt")" != complete* ]]; then
  echo "ERROR: Hessian run returned without a complete status" >&2
  exit 7
fi
date --iso-8601=seconds > "$run_dir/completed_at.txt"
printf 'complete egf=120 hessian=120 at=%s\n' "$(date --iso-8601=seconds)" > "$status_file"
