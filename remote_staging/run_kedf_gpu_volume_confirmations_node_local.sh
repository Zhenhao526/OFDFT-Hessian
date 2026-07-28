#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 REPOSITORY RUN_ROOT [RUN_ROOT ...]" >&2
  exit 2
fi

repository=$1
shift
run_roots=("$@")
IFS=',' read -r -a gpu_ids <<<"${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a cpu_ids <<<"${CPU_IDS:-36,37,74,75}"
points=()
point_roots=()

for root in "${run_roots[@]}"; do
  [[ -f "$root/confirmation_manifest.json" ]]
  [[ -f "$root/GPU_INPUT_SHA256SUMS" ]]
  (cd "$root" && sha256sum -c GPU_INPUT_SHA256SUMS)
  mapfile -t phases < <(
    python3 -c \
      'import json,sys; print("\n".join(x["phase"] for x in json.load(open(sys.argv[1]))["phases"]))' \
      "$root/confirmation_manifest.json"
  )
  for phase in "${phases[@]}"; do
    point=$root/$phase
    [[ -x "$point/run_local.sh" ]]
    if find "$point" -maxdepth 1 -type d -name 'OUT.*' | grep -q .; then
      echo "refusing existing output below $point" >&2
      exit 2
    fi
    grep -q '^device gpu$' "$point/INPUT"
    grep -q 'abacus_pw_gpu' "$point/run_local.sh"
    python3 -c \
      'import json,sys; assert json.load(open(sys.argv[1]))["mpi_ranks"] == 1' \
      "$point/metadata.json"
    points+=("$point")
    point_roots+=("$root")
  done
done

if (( ${#points[@]} > ${#gpu_ids[@]} || ${#points[@]} > ${#cpu_ids[@]} )); then
  echo "not enough GPU or CPU assignments for ${#points[@]} points" >&2
  exit 2
fi
for index in "${!points[@]}"; do
  gpu=${gpu_ids[$index]}
  if nvidia-smi -i "$gpu" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then
    echo "GPU $gpu already has a compute process" >&2
    exit 2
  fi
done

pids=()
for index in "${!points[@]}"; do
  point=${points[$index]}
  gpu=${gpu_ids[$index]}
  cpu=${cpu_ids[$index]}
  (
    cd "$point"
    exec taskset -c "$cpu" env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      ./run_local.sh
  ) >"$point/run.stdout" 2>&1 &
  pids+=("$!")
  printf '%s point=%s gpu=%s cpu=%s pid=%s\n' \
    "$(date -Iseconds)" "$point" "$gpu" "$cpu" "$!" \
    >>"${run_roots[0]}/gpu_volume_pipeline.log"
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    printf '%s finished point=%s\n' \
      "$(date -Iseconds)" "${points[$index]}" \
      >>"${run_roots[0]}/gpu_volume_pipeline.log"
  else
    status=$?
    printf '%s failed point=%s status=%s\n' \
      "$(date -Iseconds)" "${points[$index]}" "$status" \
      >>"${run_roots[0]}/gpu_volume_pipeline.log"
    failed=1
  fi
done
if (( failed )); then
  for root in "${run_roots[@]}"; do
    touch "$root/confirmation.failed"
  done
  exit 1
fi

cd "$repository"
for root in "${run_roots[@]}"; do
  env PYTHONPATH=. python3 scripts/prepare_kedf_volume_confirmation.py \
    analyze "$root" | tee "$root/analysis.stdout"
  find "$root" -type f \
    \( -name INPUT -o -name STRU -o -name KPT -o -name metadata.json \
       -o -name confirmation_manifest.json -o -name confirmation_result.json \
       -o -name confirmation_summary.json -o -name phase_analysis.json \) \
    -print0 | sort -z | xargs -0 sha256sum >"$root/SHA256SUMS"
  status=$(
    python3 -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
      "$root/confirmation_summary.json"
  )
  if [[ "$status" == "volume_confirmation_verified" ]]; then
    touch "$root/confirmation.done"
  else
    touch "$root/confirmation.failed"
    failed=1
  fi
done
exit "$failed"
