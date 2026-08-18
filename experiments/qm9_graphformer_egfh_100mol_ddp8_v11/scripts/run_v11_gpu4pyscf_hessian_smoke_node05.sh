#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname -s)" != "node05" ]]; then
  echo "ERROR: GPU Hessian smoke is bound to node05" >&2
  exit 2
fi

root=/home/shenwei01/xzh_node02_20260724
repo="$root/work/structures25_autodiff_v8_20260805"
main_py="$root/work/structures25/.venv/bin/python"
gpu_env=/home/shenwei01/.gpu_spacier_pipeline_runtime/env_archive_cache/6eb2903d89c8e098b8a5217d6137c3015e54241052d94c54b66bb5cd2715c97c/env
gpu_site="$gpu_env/lib/python3.11/site-packages"
dataset_dir="$root/data/qm9_graphformer_egfh_100mol_ddp8_v11/labelgen_smoke/QM9GraphformerEGFH100MolDDP8V11Smoke"
run_dir="$root/runs/qm9_graphformer_egfh_100mol_ddp8_v11/hessian_smoke_gpu4pyscf"
output_dir="$root/data/qm9_graphformer_egfh_100mol_ddp8_v11/hessian_smoke_gpu4pyscf"
chk="$dataset_dir/kohn_sham/qm9_graphformer_egfh_100mol_ddp8_v11_smoke_0000005.0000000.chk"
expected_chk_sha=b0f09e2b70d69cb125c66cbd4e586b88c1992d6b0d5f06865ac0cc02d8cad272

for path_value in "$root" "$repo" "$dataset_dir" "$run_dir" "$output_dir"; do
  if [[ "$path_value" == /scratch* ]]; then
    echo "ERROR: shared scratch paths are forbidden for v11" >&2
    exit 3
  fi
done

mkdir -p "$run_dir" "$output_dir" "$root/cache/cupy_v11" "$root/tmp/pyscf_v11"
actual_chk_sha=$(sha256sum "$chk" | awk '{print $1}')
if [[ "$actual_chk_sha" != "$expected_chk_sha" ]]; then
  echo "ERROR: smoke checkpoint SHA256 mismatch: $actual_chk_sha" >&2
  exit 4
fi

main_site="$root/work/structures25/.venv/lib/python3.11/site-packages"
cuda_dirs=()
for lib_dir in "$main_site"/nvidia/*/lib; do
  [[ ! -d "$lib_dir" ]] || cuda_dirs+=("$lib_dir")
done
if [[ ${#cuda_dirs[@]} -eq 0 ]]; then
  echo "ERROR: local CUDA runtime directories are missing" >&2
  exit 5
fi
export LD_LIBRARY_PATH="$(IFS=:; echo "${cuda_dirs[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$gpu_site:$repo"
export CUDA_VISIBLE_DEVICES=0
export CUPY_CACHE_DIR="$root/cache/cupy_v11"
export PYSCF_TMPDIR="$root/tmp/pyscf_v11"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

status_file="$run_dir/status.txt"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then printf "failed rc=%s at=%s\n" "$rc" "$(date --iso-8601=seconds)" > "$status_file"; fi' EXIT
printf 'running at=%s\n' "$(date --iso-8601=seconds)" > "$status_file"

"$main_py" "$root/work/qm9_graphformer_egfh_100mol_ddp8_v11/scripts/make_hessian_inventory_split.py" \
  --dataset-dir "$dataset_dir" \
  --dataset-name QM9GraphformerEGFH100MolDDP8V11Smoke \
  --output "$dataset_dir/split.pkl" \
  > "$run_dir/split.log" 2>&1
sha256sum "$dataset_dir/split.pkl" > "$run_dir/split_sha256.txt"

nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader > "$run_dir/gpu_before.csv"
/usr/bin/time -v -o "$run_dir/hessian.time.txt" \
  "$main_py" "$repo/scripts/qm9_pbe_hessian_reference_set.py" \
  --dataset-dir "$dataset_dir" \
  --output-dir "$output_dir" \
  --manifest-json "$run_dir/manifest.json" \
  --manifest-csv "$run_dir/manifest.csv" \
  --molecules 5 \
  --max-molecules 1 \
  --sample-id 0 \
  --split train \
  --workers 1 \
  --backend gpu4pyscf \
  --recompute \
  > "$run_dir/hessian.log" 2>&1
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader > "$run_dir/gpu_after.csv"

after_chk_sha=$(sha256sum "$chk" | awk '{print $1}')
if [[ "$after_chk_sha" != "$expected_chk_sha" ]]; then
  echo "ERROR: frozen checkpoint changed during Hessian calculation" >&2
  exit 6
fi
if ! grep -q '"success": true' "$run_dir/manifest.json"; then
  echo "ERROR: GPU4PySCF Hessian manifest did not report success" >&2
  exit 7
fi
sha256sum "$output_dir"/*.npz "$run_dir/manifest.json" "$run_dir/manifest.csv" > "$run_dir/output_sha256.txt"
printf 'complete at=%s\n' "$(date --iso-8601=seconds)" > "$status_file"
