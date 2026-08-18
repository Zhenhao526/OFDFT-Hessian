#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 ROOT MANIFEST OUTPUT" >&2
  exit 2
fi

ROOT=$1
MANIFEST=$2
OUTPUT=$3
count=0
{
  echo "schema=mg-xwm-ksdft-assigned-output-validation-v1"
  echo "host=$(hostname)"
  echo "manifest=$MANIFEST"
  while IFS= read -r relative; do
    [[ -n "$relative" ]] || continue
    job="$ROOT/$relative"
    [[ -f "$job/sp.done" && ! -e "$job/sp.failed" ]]
    [[ "$(<"$job/exit_code.txt")" == 0 ]]
    (cd "$job" && sha256sum -c OUTPUT_SHA256SUMS >/dev/null)
    mapfile -t logs < <(find "$job" -path '*/OUT.*/running_scf.log' -type f)
    [[ ${#logs[@]} -eq 1 ]]
    [[ $(grep -c '!FINAL_ETOT_IS' "${logs[0]}") -eq 1 ]]
    [[ $(grep -c 'E_entropy(-TS)' "${logs[0]}") -eq 1 ]]
    ! grep -Eq 'SCF IS NOT CONVERGED|NaN|nan|FATAL|ERROR' "${logs[0]}"
    mapfile -t occupations < <(find "$job" -path '*/OUT.*/eig_occ.txt' -type f)
    [[ ${#occupations[@]} -eq 1 ]]
    echo "PASS $relative log_sha=$(sha256sum "${logs[0]}" | awk '{print $1}')"
    count=$((count + 1))
  done < "$MANIFEST"
  echo "validated_jobs=$count"
  [[ $count -eq 30 ]]
  echo "status=verified"
} > "$OUTPUT"
sha256sum "$OUTPUT" > "$OUTPUT.sha256"
cat "$OUTPUT.sha256"
