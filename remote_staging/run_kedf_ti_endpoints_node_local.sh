#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 ENDPOINT_ROOT REPOSITORY PAIR_JSON PAIR_DAT PYTHON" >&2
  exit 2
fi

endpoint_root=$1
repository=$2
pair_json=$3
pair_dat=$4
python=$5
log=$endpoint_root/endpoint_pipeline.log

points=(
  "$endpoint_root/solid/baseline/lambda_1p000"
  "$endpoint_root/solid/ti/lambda_0p000"
  "$endpoint_root/solid/ti/lambda_1p000"
  "$endpoint_root/liquid/baseline/lambda_1p000"
  "$endpoint_root/liquid/ti/lambda_0p000"
  "$endpoint_root/liquid/ti/lambda_1p000"
)
modes=(baseline 0.0 1.0 baseline 0.0 1.0)
cpu_ranges=(0-11 12-23 24-35 38-49 50-61 62-73)

for point in "${points[@]}"; do
  [[ -x "$point/run_local.sh" ]]
  if find "$point" -maxdepth 1 -type d -name 'OUT.*' | grep -q .; then
    echo "refusing existing output below $point" >&2
    exit 2
  fi
done

: >"$log"
pids=()
for index in "${!points[@]}"; do
  point=${points[$index]}
  mode=${modes[$index]}
  cpus=${cpu_ranges[$index]}
  (
    cd "$point"
    if [[ "$mode" == baseline ]]; then
      exec taskset -c "$cpus" env \
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        ./run_local.sh
    fi
    exec taskset -c "$cpus" env \
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      MPN_TI_LAMBDA="$mode" MPN_TI_PAIR_MODEL="$pair_dat" \
      ./run_local.sh
  ) >"$point/run.stdout" 2>&1 &
  pids+=("$!")
  printf '%s started point=%s mode=%s cpus=%s pid=%s\n' \
    "$(date -Iseconds)" "$point" "$mode" "$cpus" "$!" >>"$log"
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    printf '%s finished point=%s\n' \
      "$(date -Iseconds)" "${points[$index]}" >>"$log"
  else
    status=$?
    printf '%s failed point=%s status=%s\n' \
      "$(date -Iseconds)" "${points[$index]}" "$status" >>"$log"
    failed=1
  fi
done

if ((failed)); then
  touch "$endpoint_root/endpoint_pipeline.failed"
  exit 1
fi

cd "$repository"
for phase in solid liquid; do
  env PYTHONPATH=. "$python" scripts/validate_al108_ti_endpoints.py \
    "$endpoint_root/$phase" --pair-model "$pair_json" \
    --expected-steps 10 >>"$log" 2>&1
  grep -q '"status": "endpoint_validation_passed"' \
    "$endpoint_root/$phase/endpoint_validation.json"
done

find "$endpoint_root" -type f \
  \( -name INPUT -o -name STRU -o -name KPT -o -name metadata.json \
     -o -name manifest.json -o -name endpoint_manifest.json \
     -o -name endpoint_validation.json \) \
  -print0 | sort -z | xargs -0 sha256sum >"$endpoint_root/SHA256SUMS"
touch "$endpoint_root/endpoint_pipeline.done"
