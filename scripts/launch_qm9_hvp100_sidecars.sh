#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
source "${ENV_FILE:-/scratch/xzh/env.sh}"

export DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
export DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

RUN_ROOT="${RUN_ROOT:-${DFT_MODELS}/hvp100/20260716}"
SELECTION="${RUN_ROOT}/selection"
REFERENCES="${RUN_ROOT}/pbe_hessians/pbe_hessian_manifest_train100_val20.json"
OUT="${RUN_ROOT}/hvp_sidecars"
SHARDS="${GPU_SHARDS:-8}"
BASE_RUN="${BASE_RUN:-${DFT_MODELS}/train/runs/qm9_random1000_egf_forcew1p0_e10_20260714_202203}"
BASE_CKPT="${BASE_CKPT:-${BASE_RUN}/checkpoints/epoch_009.ckpt}"
mkdir -p "${OUT}/chunks" "${OUT}/sidecars" "${OUT}/logs"

python - "${REFERENCES}" "${OUT}/chunks" "${SHARDS}" <<'PY'
import json, sys
from pathlib import Path

rows = [row for row in json.loads(Path(sys.argv[1]).read_text()) if row.get("success")]
out = Path(sys.argv[2])
nshards = int(sys.argv[3])
chunks = [[] for _ in range(nshards)]
loads = [0 for _ in range(nshards)]
for row in sorted(rows, key=lambda item: (-int(item["natoms"]), item["molecule_id"])):
    shard = min(range(nshards), key=lambda index: (loads[index], index))
    chunks[shard].append(row["molecule_id"])
    loads[shard] += int(row["natoms"])**3
for shard, chunk in enumerate(chunks):
    (out / f"chunk_{shard:02d}.txt").write_text(",".join(chunk) + "\n")
print("chunk_sizes", [len(chunk) for chunk in chunks], "loads", loads)
PY

pids=()
for shard in $(seq 0 $((SHARDS - 1))); do
  molecules="$(cat "${OUT}/chunks/chunk_$(printf '%02d' "${shard}").txt")"
  [[ -z "${molecules}" ]] && continue
  (
    export CUDA_VISIBLE_DEVICES="${shard}"
    /usr/bin/time -v -o "${OUT}/logs/shard_$(printf '%02d' "${shard}").time" \
      python scripts/qm9_build_hvp_sidecars.py \
        --reference-manifest "${REFERENCES}" \
        --split-file "${SELECTION}/split.pkl" \
        --data-dir "${DFT_DATA}" \
        --paired-label-dir "${DFT_DATA}/QM9PBEForceRandom1000PairedTrain/labels_local_frames_global_symmetric_natrep" \
        --train-parent-ids "${SELECTION}/train_parent_ids.txt" \
        --run-dir "${BASE_RUN}" \
        --checkpoint "${BASE_CKPT}" \
        --output-dir "${OUT}/chunks/shard_$(printf '%02d' "${shard}")" \
        --molecules "${molecules}" \
        --device cuda:0 \
        --hvp-step 3e-5 \
        --integral-derivative-step 1e-4 \
        --integral-derivative-workers 8 \
        --correction-mode none
  ) >"${OUT}/logs/shard_$(printf '%02d' "${shard}").log" 2>&1 &
  pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failures=$((failures + 1))
done
[[ "${failures}" == 0 ]] || { echo "${failures} sidecar shards failed" >&2; exit 1; }

python - "${OUT}" <<'PY'
import hashlib, json, shutil, sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for manifest_path in sorted((out / "chunks").glob("shard_*/manifest.json")):
    manifest = json.loads(manifest_path.read_text())
    rows.extend(manifest["rows"])
    for source in manifest_path.parent.glob("*.npz"):
        target = out / "sidecars" / source.name
        if target.exists() and hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(source.read_bytes()).digest():
            raise SystemExit(f"Conflicting sidecar: {target}")
        if not target.exists():
            shutil.copy2(source, target)
rows.sort(key=lambda row: row["molecule_id"])
manifest = {
    "definition": "Merged train100+validation20 learned-energy fixed-density direct-HVP sidecars with PBE analytic total-HVP targets; strict relaxed complete-total HVP is validation-only.",
    "count": len(rows),
    "test_accessed": False,
    "rows": rows,
}
path = out / "manifest.json"
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
summary = {
    "count": len(rows),
    "sidecar_count": len(list((out / "sidecars").glob("*.npz"))),
    "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "direction_kinds": {kind: sum(row["direction_kind"] == kind for row in rows) for kind in sorted({row["direction_kind"] for row in rows})},
    "mean_pbe_hvp_rms": sum(row["pbe_hvp_rms"] for row in rows) / len(rows),
    "mean_correction_hvp_rms": sum(row["correction_hvp_rms"] for row in rows) / len(rows),
}
(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
if len(rows) != 120 or summary["sidecar_count"] != 120:
    raise SystemExit(1)
PY
