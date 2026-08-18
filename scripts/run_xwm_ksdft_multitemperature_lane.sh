#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 ROOT MANIFEST" >&2
  exit 2
fi

ROOT=$1
MANIFEST=$2
MPIRUN=/home/shenwei01/wt_melting_runtime_20260724/conda_prefix/bin/mpirun
ABACUS=/home/shenwei01/wt_melting_runtime_20260724/build-abacus-wt-cpu/source/abacus_pw_para
CPU_SETS=(0-35 38-73)
mapfile -t JOBS < <(sed '/^[[:space:]]*$/d' "$MANIFEST")

failed=0
for ((start=0; start<${#JOBS[@]}; start+=2)); do
  pids=()
  labels=()
  for lane in 0 1; do
    index=$((start + lane))
    (( index < ${#JOBS[@]} )) || continue
    relative=${JOBS[$index]}
    job="$ROOT/$relative"
    if [[ -f "$job/exit_code.txt" ]] \
       && [[ "$(<"$job/exit_code.txt")" == 0 ]] \
       && grep -Rqs '!FINAL_ETOT_IS' "$job"/OUT.*/running_scf.log 2>/dev/null \
       && grep -Rqs 'E_entropy(-TS)' "$job"/OUT.*/running_scf.log 2>/dev/null; then
      echo "SKIP complete $relative"
      continue
    fi
    (
      cd "$job"
      sha256sum -c INPUT_SHA256SUMS > input_sha_check.txt
      date -u +%Y-%m-%dT%H:%M:%SZ > started_at.txt
      set +e
      /usr/bin/time -v -o resource_usage.txt \
        env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        taskset -c "${CPU_SETS[$lane]}" \
        "$MPIRUN" --bind-to none -np 36 "$ABACUS" > run.stdout 2>&1
      rc=$?
      set -e
      echo "$rc" > exit_code.txt
      date -u +%Y-%m-%dT%H:%M:%SZ > finished_at.txt
      find . -maxdepth 2 -type f ! -name OUTPUT_SHA256SUMS -print0 \
        | sort -z | xargs -0 sha256sum > OUTPUT_SHA256SUMS
      if [[ $rc -eq 0 ]] \
         && grep -Rqs '!FINAL_ETOT_IS' OUT.*/running_scf.log \
         && grep -Rqs 'E_entropy(-TS)' OUT.*/running_scf.log \
         && ! grep -Rqs -E 'SCF IS NOT CONVERGED|nan|NaN|FATAL|ERROR' OUT.*/running_scf.log; then
        touch sp.done
        exit 0
      fi
      touch sp.failed
      exit 1
    ) &
    pids+=("$!")
    labels+=("$relative")
  done
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      echo "PASS ${labels[$i]}"
    else
      echo "FAIL ${labels[$i]}"
      failed=1
    fi
  done
  (( failed == 0 )) || exit 1
done

echo "ALL_ASSIGNED_COMPLETE $(hostname) $MANIFEST"
