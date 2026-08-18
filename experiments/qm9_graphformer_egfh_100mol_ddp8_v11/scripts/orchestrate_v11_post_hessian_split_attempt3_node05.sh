#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname -s)" != "node05" ]]; then
  echo "ERROR: v11 post-Hessian recovery is bound to node05" >&2
  exit 2
fi

root=/home/shenwei01/xzh_node02_20260724
v11_data="$root/data/qm9_graphformer_egfh_100mol_ddp8_v11"
v11_work="$root/work/qm9_graphformer_egfh_100mol_ddp8_v11"
source_repo="$root/work/structures25_autodiff_v8_20260805"
hessian_run="$root/runs/qm9_graphformer_egfh_100mol_ddp8_v11/hessian_candidate120_ddp8_attempt2"
run_dir="$root/runs/qm9_graphformer_egfh_100mol_ddp8_v11/post_hessian_split_attempt3"
dataset_dir="$v11_data/fresh_labels/QM9GraphformerEGFH100MolDDP8V11Candidate120"
hessian_dir="$v11_data/fresh_hessians_candidate120"
candidate_manifest="$v11_data/candidate120/candidate120_manifest.json"
direction_dir="$v11_data/candidate120_internal_bases"
split_dir="$v11_data/frozen_split"
split_manifest="$split_dir/final100_split_manifest.json"
main_py="$root/work/structures25/.venv/bin/python"

for path_value in "$root" "$v11_data" "$v11_work" "$source_repo" "$run_dir" "$direction_dir" "$split_dir"; do
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

upstream_status=$(cat "$hessian_run/status.txt" 2>/dev/null || true)
if [[ "$upstream_status" != complete* ]]; then
  echo "ERROR: verified Hessian attempt2 is not complete: $upstream_status" >&2
  exit 4
fi
if [[ -e "$direction_dir" || -e "$split_dir" ]]; then
  echo "ERROR: recovery output already exists; refusing overwrite" >&2
  exit 5
fi
if [[ "$(find "$hessian_dir" -maxdepth 1 -type f -name '*.npz' | wc -l)" != 120 ]]; then
  echo "ERROR: exactly 120 verified Hessians are required" >&2
  exit 6
fi

export PYTHONPATH="$source_repo${PYTHONPATH:+:$PYTHONPATH}"
"$main_py" -c 'from mldft.ofdft.internal_directions import build_structured_internal_direction_bank; print("mldft_import_ok")' \
  > "$run_dir/import_smoke.log" 2>&1

printf 'running stage=internal_bases at=%s\n' "$(date --iso-8601=seconds)" > "$status_file"
"$main_py" "$v11_work/scripts/prepare_v11_candidate120_internal_bases.py" \
  --candidate-manifest "$candidate_manifest" \
  --dataset-dir "$dataset_dir" \
  --hessian-dir "$hessian_dir" \
  --output-dir "$direction_dir" \
  --seed 20260817 \
  > "$run_dir/internal_bases.log" 2>&1
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
