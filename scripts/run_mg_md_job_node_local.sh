#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 JOB_DIR CPUSET MPI_RANKS" >&2
  exit 2
fi

JOB_DIR=$1
CPUSET=$2
MPI_RANKS=$3
BIN=${ABACUS_BIN:-/home/shenwei01/wt_melting_runtime_20260724/build-abacus-wt-cpu/source/abacus_pw_para}
MPI=${MPIRUN_BIN:-/home/shenwei01/wt_melting_runtime_20260724/conda_prefix/bin/mpirun}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1

test -x "$BIN"
test -x "$MPI"
test -f "$JOB_DIR/INPUT"
test -f "$JOB_DIR/STRU"
test -f "$JOB_DIR/KPT"
test -f "$JOB_DIR/mg_phase_md_manifest.json"
if find "$JOB_DIR" -maxdepth 1 -type d -name 'OUT.*' | grep -q .; then
  echo "refusing to overwrite existing output: $JOB_DIR" >&2
  exit 3
fi

cd "$JOB_DIR"
date -Is > started_at.txt
sha256sum INPUT STRU KPT mg_phase_md_manifest.json > RUN_INPUT_SHA256SUMS
set +e
/usr/bin/time -v -o resource_usage.txt \
  taskset -c "$CPUSET" "$MPI" --bind-to none -np "$MPI_RANKS" "$BIN" \
  > run.stdout 2>&1
rc=$?
set -e
date -Is > finished_at.txt
printf '%s\n' "$rc" > exit_code.txt
if [[ $rc -eq 0 ]]; then
  touch md.done
fi
exit "$rc"
