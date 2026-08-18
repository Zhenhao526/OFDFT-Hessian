#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname -s)" != "node05" ]]; then
  echo "ERROR: candidate label generation is bound to node05" >&2
  exit 2
fi

root=/home/shenwei01/xzh_node02_20260724
repo="$root/work/structures25_autodiff_v8_20260805"
v11_data="$root/data/qm9_graphformer_egfh_100mol_ddp8_v11"
raw_dir="$v11_data/candidate120/raw"
manifest="$v11_data/candidate120/candidate120_manifest.json"
data_root="$v11_data/fresh_labels"
run_dir="$root/runs/qm9_graphformer_egfh_100mol_ddp8_v11/labelgen_candidate120"
dataset_name=QM9GraphformerEGFH100MolDDP8V11Candidate120
dataset_filename=qm9_graphformer_egfh_100mol_ddp8_v11_candidate120
dataset_root="$data_root/$dataset_name"
expected_manifest_sha=f1a6a16be453bfbf9898831922b6e482d5581c268ecaaf77d60dff18a97eb0e8
num_processes=${V11_LABEL_NUM_PROCESSES:-20}

for path_value in "$root" "$repo" "$v11_data" "$raw_dir" "$data_root" "$run_dir"; do
  if [[ "$path_value" == /scratch* ]]; then
    echo "ERROR: shared scratch paths are forbidden for v11" >&2
    exit 3
  fi
done

actual_manifest_sha=$(sha256sum "$manifest" | awk '{print $1}')
if [[ "$actual_manifest_sha" != "$expected_manifest_sha" ]]; then
  echo "ERROR: candidate120 manifest SHA256 mismatch: $actual_manifest_sha" >&2
  exit 4
fi
raw_count=$(find "$raw_dir" -maxdepth 1 -type f -name 'dsgdb9nsd_*.xyz' | wc -l)
if [[ "$raw_count" != 120 ]]; then
  echo "ERROR: expected 120 raw XYZ files, found $raw_count" >&2
  exit 5
fi

mkdir -p "$data_root" "$run_dir/hydra" "$root/tmp/qm9_v11_candidate120_labels"
cd "$repo"
source scripts/activate_qm9_node_local.sh
export DFT_DATA="$data_root"
export DFT_MODELS="$root/models"
export TMPDIR="$root/tmp/qm9_v11_candidate120_labels"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

common=(
  preset=qm9_pbe_force_full
  "dataset.name=$dataset_name"
  "dataset.filename=$dataset_filename"
  "dataset.raw_data_dir=$raw_dir"
  dataset.num_perturbations=0
  dataset.include_reference=true
  start_idx=0
  n_molecules=120
  "num_processes=$num_processes"
  num_threads_per_process=1
  max_memory_per_process=12000
  verify_files=true
  remove_broken_files=false
)

status_file="$run_dir/status.txt"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then printf "failed rc=%s at=%s\n" "$rc" "$(date --iso-8601=seconds)" > "$status_file"; fi' EXIT
printf 'running stage=preflight at=%s\n' "$(date --iso-8601=seconds)" > "$status_file"
date --iso-8601=seconds > "$run_dir/started_at.txt"
sha256sum "$manifest" > "$run_dir/input_manifest_sha256.txt"
sha256sum "$raw_dir"/*.xyz > "$run_dir/input_xyz_sha256.txt"
hostname > "$run_dir/hostname.txt"
free -b > "$run_dir/memory_before.txt"
df -B1 "$root" > "$run_dir/disk_before.txt"

printf 'running stage=kohn_sham at=%s\n' "$(date --iso-8601=seconds)" > "$status_file"
/usr/bin/time -v -o "$run_dir/kohn_sham.time.txt" \
  "$PYTHON_BIN" -m mldft.datagen.kohn_sham_dataset \
  "${common[@]}" \
  "hydra.run.dir=$run_dir/hydra/kohn_sham" \
  > "$run_dir/kohn_sham.log" 2>&1

chk_count=$(find "$dataset_root/kohn_sham" -maxdepth 1 -type f -name '*.chk' | wc -l)
if [[ "$chk_count" != 120 ]]; then
  echo "ERROR: expected 120 Kohn-Sham checkpoints, found $chk_count" >&2
  exit 6
fi

printf 'running stage=labels at=%s\n' "$(date --iso-8601=seconds)" > "$status_file"
/usr/bin/time -v -o "$run_dir/labels.time.txt" \
  "$PYTHON_BIN" -m mldft.datagen.generate_labels_dataset \
  "${common[@]}" \
  "hydra.run.dir=$run_dir/hydra/labels" \
  > "$run_dir/labels.log" 2>&1

label_count=$(find "$dataset_root/labels" -maxdepth 1 -type f -name '*.zarr.zip' | wc -l)
if [[ "$label_count" != 120 ]]; then
  echo "ERROR: expected 120 E/G/F label archives, found $label_count" >&2
  exit 7
fi

sha256sum "$dataset_root/kohn_sham"/*.chk > "$run_dir/kohn_sham_sha256.txt"
sha256sum "$dataset_root/labels"/*.zarr.zip > "$run_dir/labels_sha256.txt"
du -sb "$dataset_root" > "$run_dir/dataset_size_bytes.txt"
free -b > "$run_dir/memory_after.txt"
df -B1 "$root" > "$run_dir/disk_after.txt"
date --iso-8601=seconds > "$run_dir/completed_at.txt"
printf 'complete chk=120 labels=120 at=%s\n' "$(date --iso-8601=seconds)" > "$status_file"
