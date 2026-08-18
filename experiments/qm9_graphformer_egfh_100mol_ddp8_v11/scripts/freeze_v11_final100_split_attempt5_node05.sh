#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname -s)" != "node05" ]]; then
  echo "ERROR: v11 split recovery is bound to node05" >&2
  exit 2
fi

root=/home/shenwei01/xzh_node02_20260724
v11_data="$root/data/qm9_graphformer_egfh_100mol_ddp8_v11"
v11_work="$root/work/qm9_graphformer_egfh_100mol_ddp8_v11"
run_dir="$root/runs/qm9_graphformer_egfh_100mol_ddp8_v11/post_hessian_split_attempt5"
dataset_dir="$v11_data/fresh_labels/QM9GraphformerEGFH100MolDDP8V11Candidate120"
hessian_dir="$v11_data/fresh_hessians_candidate120"
candidate_manifest="$v11_data/candidate120/candidate120_manifest.json"
direction_dir="$v11_data/candidate120_internal_bases"
split_dir="$v11_data/frozen_split"
split_manifest="$split_dir/final100_split_manifest.json"
main_py="$root/work/structures25/.venv/bin/python"

for path_value in "$root" "$v11_data" "$v11_work" "$run_dir" "$direction_dir" "$split_dir"; do
  if [[ "$path_value" == /scratch* ]]; then
    echo "ERROR: shared scratch paths are forbidden for v11" >&2
    exit 3
  fi
done

mkdir -p "$run_dir"
status_file="$run_dir/status.txt"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then printf "failed rc=%s at=%s\n" "$rc" "$(date --iso-8601=seconds)" > "$status_file"; fi' EXIT
printf 'running stage=preflight at=%s\n' "$(date --iso-8601=seconds)" > "$status_file"
date --iso-8601=seconds > "$run_dir/started_at.txt"

if [[ ! -f "$direction_dir/manifest.json" || ! -f "$direction_dir/summary.json" ]]; then
  echo "ERROR: verified complete internal-basis inventory is missing" >&2
  exit 4
fi
if [[ "$(find "$direction_dir/artifacts" -maxdepth 1 -type f -name '*.npz' | wc -l)" != 120 ]]; then
  echo "ERROR: exactly 120 internal-basis artifacts are required" >&2
  exit 5
fi
if [[ -e "$split_dir" ]]; then
  echo "ERROR: split output already exists; refusing overwrite" >&2
  exit 6
fi
sha256sum "$direction_dir/manifest.json" "$direction_dir/summary.json" \
  > "$run_dir/internal_bases_manifest_sha256.txt"

printf 'running stage=freeze_split at=%s\n' "$(date --iso-8601=seconds)" > "$status_file"
mkdir "$split_dir"
"$main_py" "$v11_work/scripts/freeze_v11_final100_split.py" \
  --candidate-manifest "$candidate_manifest" \
  --dataset-dir "$dataset_dir" \
  --hessian-dir "$hessian_dir" \
  --seed 20260817 \
  --output "$split_manifest" \
  > "$run_dir/freeze_split.log" 2>&1
sha256sum "$split_manifest" > "$run_dir/final100_split_manifest_sha256.txt"
date --iso-8601=seconds > "$run_dir/completed_at.txt"
printf 'complete final100_split_frozen training_not_started at=%s\n' "$(date --iso-8601=seconds)" > "$status_file"
