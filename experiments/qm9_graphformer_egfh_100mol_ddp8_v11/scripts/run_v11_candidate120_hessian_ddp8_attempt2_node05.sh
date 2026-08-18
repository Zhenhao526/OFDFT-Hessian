#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname -s)" != "node05" ]]; then
  echo "ERROR: Hessian generation is bound to node05" >&2
  exit 2
fi

root=/home/shenwei01/xzh_node02_20260724
repo="$root/work/structures25_autodiff_v8_20260805"
v11_work="$root/work/qm9_graphformer_egfh_100mol_ddp8_v11"
v11_data="$root/data/qm9_graphformer_egfh_100mol_ddp8_v11"
dataset_dir="$v11_data/fresh_labels/QM9GraphformerEGFH100MolDDP8V11Candidate120"
candidate_manifest="$v11_data/candidate120/candidate120_manifest.json"
output_dir="$v11_data/fresh_hessians_candidate120"
run_dir="$root/runs/qm9_graphformer_egfh_100mol_ddp8_v11/hessian_candidate120_ddp8_attempt2"
main_py="$root/work/structures25/.venv/bin/python"
gpu_env=/home/shenwei01/.gpu_spacier_pipeline_runtime/env_archive_cache/6eb2903d89c8e098b8a5217d6137c3015e54241052d94c54b66bb5cd2715c97c/env
gpu_site="$gpu_env/lib/python3.11/site-packages"
expected_candidate_sha=f1a6a16be453bfbf9898831922b6e482d5581c268ecaaf77d60dff18a97eb0e8

for path_value in "$root" "$repo" "$v11_work" "$v11_data" "$dataset_dir" "$output_dir" "$run_dir"; do
  if [[ "$path_value" == /scratch* ]]; then
    echo "ERROR: shared scratch paths are forbidden for v11" >&2
    exit 3
  fi
done

status_file="$run_dir/status.txt"
mkdir -p "$run_dir" "$output_dir" "$root/cache/cupy_v11" "$root/tmp/pyscf_v11"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then printf "failed rc=%s at=%s\n" "$rc" "$(date --iso-8601=seconds)" > "$status_file"; fi' EXIT
printf 'running stage=preflight at=%s\n' "$(date --iso-8601=seconds)" > "$status_file"

if [[ "$(sha256sum "$candidate_manifest" | awk '{print $1}')" != "$expected_candidate_sha" ]]; then
  echo "ERROR: frozen candidate manifest SHA256 mismatch" >&2
  exit 4
fi
if [[ "$(find "$dataset_dir/kohn_sham" -maxdepth 1 -type f -name '*.chk' | wc -l)" != 120 ]]; then
  echo "ERROR: exactly 120 fresh SCF checkpoints are required" >&2
  exit 5
fi
if [[ "$(find "$dataset_dir/labels" -maxdepth 1 -type f -name '*.zarr.zip' | wc -l)" != 120 ]]; then
  echo "ERROR: exactly 120 fresh E/G/F label archives are required" >&2
  exit 6
fi
if find "$output_dir" -mindepth 1 -print -quit | grep -q .; then
  echo "ERROR: Hessian output directory is not empty; refusing overwrite/recompute" >&2
  exit 7
fi

mapfile -t gpu_rows < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
if [[ ${#gpu_rows[@]} -ne 8 ]]; then
  echo "ERROR: node05 must expose exactly 8 GPUs, found ${#gpu_rows[@]}" >&2
  exit 8
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')" ]]; then
  echo "ERROR: at least one GPU compute process exists; refusing to share or preempt" >&2
  exit 9
fi
for row in "${gpu_rows[@]}"; do
  memory_used=$(awk -F, '{gsub(/ /,"",$2); print $2}' <<< "$row")
  if (( memory_used > 256 )); then
    echo "ERROR: GPU is not cleanly idle: $row" >&2
    exit 10
  fi
done

main_site="$root/work/structures25/.venv/lib/python3.11/site-packages"
cuda_dirs=()
for lib_dir in "$main_site"/nvidia/*/lib; do
  [[ ! -d "$lib_dir" ]] || cuda_dirs+=("$lib_dir")
done
if [[ ${#cuda_dirs[@]} -eq 0 ]]; then
  echo "ERROR: local CUDA runtime directories are missing" >&2
  exit 11
fi
export LD_LIBRARY_PATH="$(IFS=:; echo "${cuda_dirs[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$gpu_site:$repo"
export CUPY_CACHE_DIR="$root/cache/cupy_v11"
export PYSCF_TMPDIR="$root/tmp/pyscf_v11"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

date --iso-8601=seconds > "$run_dir/started_at.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader > "$run_dir/gpu_before.csv"
sha256sum "$candidate_manifest" > "$run_dir/candidate_manifest_sha256.txt"
sha256sum "$dataset_dir/kohn_sham"/*.chk > "$run_dir/kohn_sham_sha256_before.txt"
sha256sum "$dataset_dir/labels"/*.zarr.zip > "$run_dir/labels_sha256.txt"

"$main_py" "$v11_work/scripts/make_hessian_inventory_split.py" \
  --dataset-dir "$dataset_dir" \
  --dataset-name QM9GraphformerEGFH100MolDDP8V11Candidate120 \
  --output "$dataset_dir/split.pkl" \
  > "$run_dir/split.log" 2>&1
sha256sum "$dataset_dir/split.pkl" > "$run_dir/split_sha256.txt"

"$main_py" "$v11_work/scripts/plan_candidate120_hessian_shards.py" plan \
  --dataset-dir "$dataset_dir" \
  --candidate-manifest "$candidate_manifest" \
  --shards 8 \
  --output "$run_dir/shard_plan.json" \
  > "$run_dir/shard_plan.log" 2>&1
sha256sum "$run_dir/shard_plan.json" > "$run_dir/shard_plan_sha256.txt"

printf 'running stage=hessian_ddp8 at=%s\n' "$(date --iso-8601=seconds)" > "$status_file"
pids=()
for rank in $(seq 0 7); do
  rank_dir="$run_dir/rank$rank"
  mkdir -p "$rank_dir"
  molecule_csv=$("$main_py" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(",".join(x["molecule_id"] for x in p["shards"][int(sys.argv[2])]["molecules"]))' "$run_dir/shard_plan.json" "$rank")
  if [[ -z "$molecule_csv" ]]; then
    echo "ERROR: empty Hessian shard $rank" >&2
    exit 12
  fi
  (
    export CUDA_VISIBLE_DEVICES="$rank"
    /usr/bin/time -v -o "$rank_dir/time.txt" \
      "$main_py" "$repo/scripts/qm9_pbe_hessian_reference_set.py" \
      --dataset-dir "$dataset_dir" \
      --output-dir "$output_dir" \
      --manifest-json "$rank_dir/manifest.json" \
      --manifest-csv "$rank_dir/manifest.csv" \
      --molecules "$molecule_csv" \
      --max-molecules 120 \
      --sample-id 0 \
      --split train \
      --workers 1 \
      --backend gpu4pyscf \
      --recompute \
      > "$rank_dir/hessian.log" 2>&1
  ) &
  pids+=("$!")
  printf '%s,%s\n' "$rank" "$!" >> "$run_dir/rank_pids.csv"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "ERROR: at least one Hessian rank failed" >&2
  exit 13
fi

printf 'running stage=verify at=%s\n' "$(date --iso-8601=seconds)" > "$status_file"
"$main_py" "$v11_work/scripts/plan_candidate120_hessian_shards.py" verify \
  --plan "$run_dir/shard_plan.json" \
  --manifest-dir "$run_dir" \
  --symmetry-tolerance 1e-10 \
  --output "$run_dir/combined_manifest.json" \
  > "$run_dir/verify.log" 2>&1

sha256sum "$dataset_dir/kohn_sham"/*.chk > "$run_dir/kohn_sham_sha256_after.txt"
if ! cmp -s "$run_dir/kohn_sham_sha256_before.txt" "$run_dir/kohn_sham_sha256_after.txt"; then
  echo "ERROR: at least one frozen SCF checkpoint changed" >&2
  exit 14
fi
sha256sum "$output_dir"/*.npz > "$run_dir/hessian_sha256.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader > "$run_dir/gpu_after.csv"
du -sb "$output_dir" > "$run_dir/output_size_bytes.txt"
date --iso-8601=seconds > "$run_dir/completed_at.txt"
printf 'complete hessians=120 at=%s\n' "$(date --iso-8601=seconds)" > "$status_file"
