#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/home/shenwei01/WT_Al_melting_workspace_20260724/repository}
run_root=${RUN_ROOT:-/home/shenwei01/WT_Al_melting_workspace_20260724/runs/xwm_lkt_20260727}
torch_runner=${TORCH_RUNNER:-$repo/remote_staging/run_node05_cached_cuda_torch.sh}
method=${METHOD:?set METHOD}
phase=${PHASE:?set PHASE}
parent=${PARENT:?set PARENT}
output=${OUTPUT:?set OUTPUT}
steps=${STEPS:-15000}
temperature=${TEMPERATURE:-900}
cpu=${CPU:-75}
seed=${SEED:-20260729}

[[ $method == xwm || $method == lkt ]] || {
  echo "METHOD must be xwm or lkt" >&2
  exit 2
}
[[ $phase == solid || $phase == liquid ]] || {
  echo "PHASE must be solid or liquid" >&2
  exit 2
}
[[ -x $torch_runner ]] || {
  echo "missing torch runner: $torch_runner" >&2
  exit 2
}
[[ -f $parent/checkpoint.json && -f $parent/summary.json ]] || {
  echo "missing parent checkpoint or summary below $parent" >&2
  exit 2
}
[[ ! -e $output ]] || {
  echo "refusing existing output: $output" >&2
  exit 2
}

model=$run_root/$method/free_energy_T0900/pair_reference_v1.json
[[ -f $model ]] || {
  echo "missing pair model: $model" >&2
  exit 2
}

mkdir -p "$output"
log=$output/pipeline.log
done_file=$output/pipeline.done
failed_file=$output/pipeline.failed
exec > >(tee -a "$log") 2>&1

mark_failed() {
  local rc=$?
  trap - ERR
  if [[ ! -e $done_file ]]; then
    printf '%s failed exit_code=%s\n' "$(date -Iseconds)" "$rc" \
      | tee "$failed_file"
  fi
  exit "$rc"
}
trap mark_failed ERR

python3 - "$parent" "$model" "$method" "$phase" "$steps" "$temperature" \
  "$cpu" "$seed" "$output/preflight.json" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

parent = Path(sys.argv[1])
model_path = Path(sys.argv[2])
method = sys.argv[3]
phase = sys.argv[4]
steps = int(sys.argv[5])
temperature = float(sys.argv[6])
cpu = int(sys.argv[7])
seed = int(sys.argv[8])
out = Path(sys.argv[9])

model = json.loads(model_path.read_text())
summary = json.loads((parent / "summary.json").read_text())
checkpoint = json.loads((parent / "checkpoint.json").read_text())
if not model.get("reference_gate_passed"):
    raise SystemExit("pair reference has not passed its gate")
if model.get("target_kedf") != method:
    raise SystemExit("pair-reference target_kedf mismatch")
if int(summary.get("natoms", 0)) != 108 or summary.get("stable") is not True:
    raise SystemExit("parent pair trajectory is not a stable 108-atom run")
if float(summary.get("minimum_distance_angstrom", 0.0)) < 2.0:
    raise SystemExit("parent nearest-neighbor distance is below 2 A")
if int(checkpoint.get("steps", -1)) != int(summary.get("steps", -2)):
    raise SystemExit("parent checkpoint and summary step counts differ")
if not checkpoint.get("velocities_angstrom_per_fs"):
    raise SystemExit("parent checkpoint lacks velocities")

inputs = {
    "model": model_path,
    "parent_checkpoint": parent / "checkpoint.json",
    "parent_summary": parent / "summary.json",
}
payload = {
    "schema": "kedf-pair-preequil-extension-preflight-v1",
    "status": "verified",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "target_kedf": method,
    "phase": phase,
    "steps": steps,
    "temperature_k": temperature,
    "cpu": cpu,
    "seed": seed,
    "source_velocities_discarded": False,
    "inputs": {
        label: {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for label, path in inputs.items()
    },
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

printf '%s start method=%s phase=%s steps=%s cpu=%s\n' \
  "$(date -Iseconds)" "$method" "$phase" "$steps" "$cpu"

taskset -c "$cpu" env CUDA_VISIBLE_DEVICES= PYTHONPATH="$repo" \
  "$torch_runner" "$repo/scripts/run_pair_reference_md.py" \
  --model "$model" \
  --out "$output/run" \
  --restart "$parent/checkpoint.json" \
  --temperature "$temperature" \
  --steps "$steps" \
  --sample-every 10 \
  --seed "$seed" \
  --threads 1 \
  --device cpu \
  --store-positions

cd "$repo"
env PYTHONPATH=. python3 scripts/analyze_pair_reference_md.py \
  "$output/run" --expected "$phase" --out "$output/run/phase_analysis.json"

python3 - "$output/run/summary.json" "$output/run/phase_analysis.json" \
  "$steps" "$temperature" "$phase" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
phase = json.load(open(sys.argv[2], encoding="utf-8"))
steps = int(sys.argv[3])
temperature = float(sys.argv[4])
expected = sys.argv[5]
checks = {
    "steps": int(summary["steps"]) == steps,
    "stable": summary.get("stable") is True,
    "nearest_neighbor_gt_2_A": float(summary["minimum_distance_angstrom"]) > 2.0,
    "temperature_mean_within_25_K": (
        abs(float(summary["temperature_mean_k"]) - temperature) <= 25.0
    ),
    "phase_verified": phase.get("status") == f"{expected}_verified",
}
if not all(checks.values()):
    failed = [name for name, passed in checks.items() if not passed]
    raise SystemExit(f"extension failed gates: {failed}")
PY

find "$output" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$output/SHA256SUMS"
printf '%s verified\n' "$(date -Iseconds)" | tee "$done_file"
