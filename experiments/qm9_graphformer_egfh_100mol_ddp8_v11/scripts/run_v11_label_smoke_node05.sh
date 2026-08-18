#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname -s)" != "node05" ]]; then
  echo "ERROR: this smoke test is bound to node05" >&2
  exit 2
fi

root=/home/shenwei01/xzh_node02_20260724
repo="$root/work/structures25_autodiff_v8_20260805"
data_root="$root/data/qm9_graphformer_egfh_100mol_ddp8_v11/labelgen_smoke"
source_raw="$root/data/qm9_graphformer_egfh_100mol_ddp8_v11/candidate120/raw/dsgdb9nsd_000005.xyz"
smoke_raw="$data_root/raw"
run_dir="$root/runs/qm9_graphformer_egfh_100mol_ddp8_v11/labelgen_smoke"
dataset_name=QM9GraphformerEGFH100MolDDP8V11Smoke
dataset_filename=qm9_graphformer_egfh_100mol_ddp8_v11_smoke

for path_value in "$root" "$repo" "$data_root" "$source_raw" "$run_dir"; do
  if [[ "$path_value" == /scratch* ]]; then
    echo "ERROR: shared scratch paths are forbidden for v11" >&2
    exit 3
  fi
done

mkdir -p "$smoke_raw" "$run_dir/hydra" "$root/tmp/qm9_v11_label_smoke"
cp -p "$source_raw" "$smoke_raw/"

cd "$repo"
source scripts/activate_qm9_node_local.sh
export DFT_DATA="$data_root"
export DFT_MODELS="$root/models"
export TMPDIR="$root/tmp/qm9_v11_label_smoke"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

common=(
  preset=qm9_pbe_force_full
  "dataset.name=$dataset_name"
  "dataset.filename=$dataset_filename"
  "dataset.raw_data_dir=$smoke_raw"
  dataset.num_perturbations=0
  dataset.include_reference=true
  start_idx=0
  n_molecules=1
  num_processes=1
  num_threads_per_process=1
  max_memory_per_process=12000
  verify_files=true
  remove_broken_files=false
)

date --iso-8601=seconds > "$run_dir/started_at.txt"
sha256sum "$source_raw" > "$run_dir/source_sha256.txt"

/usr/bin/time -v -o "$run_dir/kohn_sham.time.txt" \
  "$PYTHON_BIN" -m mldft.datagen.kohn_sham_dataset \
  "${common[@]}" \
  "hydra.run.dir=$run_dir/hydra/kohn_sham" \
  > "$run_dir/kohn_sham.log" 2>&1

/usr/bin/time -v -o "$run_dir/labels.time.txt" \
  "$PYTHON_BIN" -m mldft.datagen.generate_labels_dataset \
  "${common[@]}" \
  "hydra.run.dir=$run_dir/hydra/labels" \
  > "$run_dir/labels.log" 2>&1

find "$data_root/$dataset_name/kohn_sham" -maxdepth 1 -type f -name '*.chk' -print -exec sha256sum {} \; \
  > "$run_dir/kohn_sham_files.txt"
find "$data_root/$dataset_name/labels" -maxdepth 1 -type f -name '*.zarr.zip' -print -exec sha256sum {} \; \
  > "$run_dir/label_files.txt"

chk_count=$(grep -c '\.chk$' "$run_dir/kohn_sham_files.txt" || true)
label_count=$(grep -c '\.zarr\.zip$' "$run_dir/label_files.txt" || true)
if [[ "$chk_count" != 1 || "$label_count" != 1 ]]; then
  echo "ERROR: expected one checkpoint and one label, found chk=$chk_count label=$label_count" >&2
  exit 4
fi

date --iso-8601=seconds > "$run_dir/completed_at.txt"
echo complete > "$run_dir/status.txt"
