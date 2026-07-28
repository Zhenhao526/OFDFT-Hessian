#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_ROOT" >&2
  exit 2
fi

run_root=$1
expected_jobs=${EXPECTED_JOBS:?EXPECTED_JOBS is required}
IFS=',' read -r -a gpu_ids <<<"${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a cpu_ids <<<"${CPU_IDS:-36,37,74,75}"

if [[ ${#gpu_ids[@]} -eq 0 || ${#gpu_ids[@]} -ne ${#cpu_ids[@]} ]]; then
  echo "GPU_IDS and CPU_IDS must contain the same nonzero number of slots" >&2
  exit 2
fi

mapfile -t jobs < <(find "$run_root" -type f -name run_local.sh | sort)
if [[ ${#jobs[@]} -ne $expected_jobs ]]; then
  echo "expected $expected_jobs SCF jobs, found ${#jobs[@]}" >&2
  exit 2
fi

for job in "${jobs[@]}"; do
  directory=$(dirname "$job")
  [[ -x "$job" ]]
  grep -q '^device gpu$' "$directory/INPUT"
  grep -q 'abacus_pw_gpu' "$job"
  grep -Eq -- '-np[[:space:]]+1([[:space:]]|$)' "$job"
  if find "$directory" -maxdepth 1 -type d -name 'OUT.*' | grep -q .; then
    echo "refusing existing output below $directory" >&2
    exit 2
  fi
done

for gpu in "${gpu_ids[@]}"; do
  if nvidia-smi -i "$gpu" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then
    echo "GPU $gpu already has a compute process" >&2
    exit 2
  fi
done

: >"$run_root/gpu_scf_pipeline.log"
for ((batch_start = 0; batch_start < ${#jobs[@]}; batch_start += ${#gpu_ids[@]})); do
  pids=()
  batch_jobs=()
  batch_end=$((batch_start + ${#gpu_ids[@]}))
  if ((batch_end > ${#jobs[@]})); then
    batch_end=${#jobs[@]}
  fi
  for ((index = batch_start; index < batch_end; index++)); do
    slot=$((index - batch_start))
    directory=$(dirname "${jobs[$index]}")
    gpu=${gpu_ids[$slot]}
    cpu=${cpu_ids[$slot]}
    (
      cd "$directory"
      exec taskset -c "$cpu" env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        ./run_local.sh
    ) >"$directory/run.stdout" 2>&1 &
    pids+=("$!")
    batch_jobs+=("$directory")
    printf '%s started job=%s gpu=%s cpu=%s pid=%s\n' \
      "$(date -Iseconds)" "$directory" "$gpu" "$cpu" "$!" \
      >>"$run_root/gpu_scf_pipeline.log"
  done

  failed=0
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      printf '%s finished job=%s\n' \
        "$(date -Iseconds)" "${batch_jobs[$index]}" \
        >>"$run_root/gpu_scf_pipeline.log"
    else
      status=$?
      printf '%s failed job=%s status=%s\n' \
        "$(date -Iseconds)" "${batch_jobs[$index]}" "$status" \
        >>"$run_root/gpu_scf_pipeline.log"
      failed=1
    fi
  done
  if ((failed)); then
    touch "$run_root/gpu_scf_pipeline.failed"
    exit 1
  fi
done
touch "$run_root/gpu_scf_pipeline.done"
