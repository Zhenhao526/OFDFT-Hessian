#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
source "${ENV_FILE:-/scratch/xzh/env.sh}"

export DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
export DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

RUN_ROOT="${RUN_ROOT:-${DFT_MODELS}/hvp100/20260716}"
SELECTION_DIR="${SELECTION_DIR:-${RUN_ROOT}/selection}"
VALIDATION_JSON="${VALIDATION_JSON:-${DFT_MODELS}/eval/qm9_random1000_validation_protocol/validation_representatives_20.json}"
DATASET_DIR="${DFT_DATA}/QM9PBEForceRandom1000"
OUT="${RUN_ROOT}/pbe_hessians"
CACHE="${OUT}/cache"
SHARDS="${GPU_SHARDS:-8}"
mkdir -p "${OUT}" "${CACHE}" "${OUT}/logs"

python - "${SELECTION_DIR}/train_parent_ids.txt" "${VALIDATION_JSON}" "${OUT}" "${DATASET_DIR}" "${SHARDS}" <<'PY'
import json, pickle, sys
from pathlib import Path
import zarr

train_ids_path = Path(sys.argv[1])
val_json = Path(sys.argv[2])
out_dir = Path(sys.argv[3])
dataset_dir = Path(sys.argv[4])
nshards = int(sys.argv[5])
requested = {
    "train": set(train_ids_path.read_text().split()),
    "val": {row["molecule_id"] for row in json.loads(val_json.read_text())},
}
split = pickle.loads((dataset_dir / "split.pkl").read_bytes())
for split_name, ids in requested.items():
    rows = []
    seen = set()
    for _, label_name, _ in split[split_name]:
        molecule_id, sample_text, *_ = label_name.removesuffix(".zarr.zip").split(".")
        if molecule_id not in ids or int(sample_text) != 0 or molecule_id in seen:
            continue
        seen.add(molecule_id)
        root = zarr.open(dataset_dir / "labels" / label_name, mode="r")
        rows.append((molecule_id, int(root["geometry/atomic_numbers"].shape[0])))
    if seen != ids:
        raise SystemExit(f"{split_name} missing ids: {sorted(ids-seen)}")
    chunks = [[] for _ in range(nshards)]
    loads = [0 for _ in range(nshards)]
    for molecule_id, natoms in sorted(rows, key=lambda item: (-item[1], item[0])):
        shard = min(range(nshards), key=lambda index: (loads[index], index))
        chunks[shard].append(molecule_id)
        loads[shard] += natoms**3
    chunk_dir = out_dir / f"{split_name}_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for shard, chunk in enumerate(chunks):
        (chunk_dir / f"chunk_{shard:02d}.txt").write_text(",".join(chunk) + "\n")
    print(split_name, "count", len(rows), "chunk_sizes", [len(x) for x in chunks], "loads", loads)
PY

run_split() {
  local split_name="$1"
  local chunk_dir="${OUT}/${split_name}_chunks"
  local pids=()
  for shard in $(seq 0 $((SHARDS - 1))); do
    local molecules
    molecules="$(cat "${chunk_dir}/chunk_$(printf '%02d' "${shard}").txt")"
    [[ -z "${molecules}" ]] && continue
    local count
    count="$(awk -F, '{print NF}' "${chunk_dir}/chunk_$(printf '%02d' "${shard}").txt")"
    (
      export CUDA_VISIBLE_DEVICES="${shard}"
      /usr/bin/time -v -o "${OUT}/logs/${split_name}_$(printf '%02d' "${shard}").time" \
        python scripts/qm9_pbe_hessian_reference_set.py \
          --dataset-dir "${DATASET_DIR}" \
          --output-dir "${CACHE}" \
          --manifest-json "${chunk_dir}/manifest_$(printf '%02d' "${shard}").json" \
          --manifest-csv "${chunk_dir}/manifest_$(printf '%02d' "${shard}").csv" \
          --molecules "${molecules}" \
          --max-molecules "${count}" \
          --sample-id 0 \
          --split "${split_name}" \
          --workers 1 \
          --backend gpu4pyscf
    ) >"${OUT}/logs/${split_name}_$(printf '%02d' "${shard}").log" 2>&1 &
    pids+=("$!")
  done
  local failures=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failures=$((failures + 1))
  done
  [[ "${failures}" == 0 ]] || { echo "${split_name}: ${failures} failed shards" >&2; return 1; }
}

run_split train
run_split val

python - "${OUT}" <<'PY'
import hashlib, json, sys
from pathlib import Path

out = Path(sys.argv[1])
combined = []
summary = {}
for split_name in ("train", "val"):
    rows = []
    for path in sorted((out / f"{split_name}_chunks").glob("manifest_*.json")):
        rows.extend(json.loads(path.read_text()))
    rows.sort(key=lambda row: row["molecule_id"])
    manifest = out / f"pbe_hessian_manifest_{split_name}.json"
    manifest.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    summary[split_name] = {
        "count": len(rows),
        "success": sum(bool(row.get("success")) for row in rows),
        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    combined.extend(rows)
combined.sort(key=lambda row: row["molecule_id"])
manifest = out / "pbe_hessian_manifest_train100_val20.json"
manifest.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")
summary["combined_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
if summary["train"]["success"] != 100 or summary["val"]["success"] != 20:
    raise SystemExit(1)
PY
