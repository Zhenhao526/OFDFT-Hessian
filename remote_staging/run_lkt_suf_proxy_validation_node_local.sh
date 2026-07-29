#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/home/shenwei01/WT_Al_melting_workspace_20260724/repository}
run_root=${RUN_ROOT:-/home/shenwei01/WT_Al_melting_workspace_20260724/runs/xwm_lkt_20260727}
torch_runner=${TORCH_RUNNER:-$repo/remote_staging/run_node05_cached_cuda_torch.sh}
model=${MODEL:-$run_root/lkt/free_energy_T0900/liquid_suf_pair_proxy_v4_hardcore.json}
output=${OUTPUT:-$run_root/lkt/free_energy_T0900/liquid_suf_pair_proxy_v4_validation_steps20000}
steps=${STEPS:-20000}
temperature=${TEMPERATURE:-900}
cpu=${CPU:-36}

[[ -x $torch_runner ]] || {
  echo "missing torch runner: $torch_runner" >&2
  exit 2
}
[[ -f $model ]] || {
  echo "missing proxy model: $model" >&2
  exit 2
}
[[ ! -e $output ]] || {
  echo "refusing existing output: $output" >&2
  exit 2
}

dataset=$run_root/lkt/free_energy_T0900/reference_dataset_v1/frames.jsonl
[[ -f $dataset ]] || {
  echo "missing LKT reference dataset: $dataset" >&2
  exit 2
}

mkdir -p "$output"
exec > >(tee -a "$output/pipeline.log") 2>&1

mark_failed() {
  local rc=$?
  trap - ERR
  printf '%s failed exit_code=%s\n' "$(date -Iseconds)" "$rc" \
    | tee "$output/validation.failed"
  exit "$rc"
}
trap mark_failed ERR

cd "$repo"
python3 - "$model" <<'PY'
import json
import sys

model = json.load(open(sys.argv[1], encoding="utf-8"))
checks = {
    "target_kedf_lkt": model.get("target_kedf") == "lkt",
    "liquid_phase_specific": model.get("reference_phase") == "liquid",
    "suf_proxy": model.get("reference_kind") == "suf_radial_proxy",
    "static_gate": model.get("reference_gate_passed") is True,
}
print(json.dumps(checks, sort_keys=True))
if not all(checks.values()):
    raise SystemExit("proxy provenance or static gate failed")
PY

frame_index=$(
  python3 - "$dataset" <<'PY'
import json
import sys

rows = [
    json.loads(line)
    for line in open(sys.argv[1], encoding="utf-8")
    if line.strip()
]
indices = [index for index, row in enumerate(rows) if row.get("phase") == "liquid"]
if not indices:
    raise SystemExit("dataset has no liquid frame")
print(indices[-1])
PY
)

sha256sum "$model" "$dataset" > "$output/SOURCE_SHA256"
taskset -c "$cpu" env CUDA_VISIBLE_DEVICES= PYTHONPATH="$repo" \
  "$torch_runner" "$repo/scripts/run_pair_reference_md.py" \
  --model "$model" \
  --out "$output/run" \
  --dataset-frame "$dataset" \
  --frame-index "$frame_index" \
  --temperature "$temperature" \
  --steps "$steps" \
  --sample-every 10 \
  --seed 20260731 \
  --threads 1 \
  --device cpu \
  --store-positions

env PYTHONPATH="$repo" python3 "$repo/scripts/analyze_pair_reference_md.py" \
  "$output/run" \
  --expected liquid \
  --out "$output/run/phase_analysis.json"

python3 - "$output" "$steps" "$temperature" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
steps = int(sys.argv[2])
temperature = float(sys.argv[3])
summary = json.loads((root / "run" / "summary.json").read_text())
phase = json.loads((root / "run" / "phase_analysis.json").read_text())
checks = {
    "requested_steps": int(summary["steps"]) == steps,
    "stable": summary.get("stable") is True,
    "nearest_neighbor_gt_2_A": (
        float(summary["minimum_distance_angstrom"]) > 2.0
    ),
    "temperature_mean_within_25_K": (
        abs(float(summary["temperature_mean_k"]) - temperature) <= 25.0
    ),
    "liquid_verified": phase.get("status") == "liquid_verified",
}
payload = {
    "schema": "lkt-suf-pair-proxy-validation-v1",
    "status": "verified" if all(checks.values()) else "gate_failed",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "checks": checks,
    "summary": summary,
    "phase_analysis": phase,
}
path = root / "validation_summary.json"
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
(root / "SHA256SUMS").write_text(
    "\n".join(
        f"{hashlib.sha256(item.read_bytes()).hexdigest()}  {item}"
        for item in sorted(root.rglob("*"))
        if item.is_file() and item.name != "SHA256SUMS"
    )
    + "\n"
)
if payload["status"] != "verified":
    raise SystemExit("sUF pair proxy dynamics gate failed")
PY

printf '%s verified\n' "$(date -Iseconds)" | tee "$output/validation.done"
