#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/shenwei01/xzh_node02_20260724
REPO="$ROOT/work/structures25"
PROTOCOL="$REPO/configs/audit/qm9_graphformer_0028399_full_network_capacity_smoke_v3.yaml"
ASSET_DIR="$ROOT/artifacts/graphformer_0028399_full_network_capacity_smoke_v3"
REGISTRATION="$ASSET_DIR/asset_registration.json"
RUN_ROOT="$ROOT/runs/graphformer_0028399_full_network_capacity_smoke_v3"
LOG_DIR="$RUN_ROOT/logs"
DEVICE="${FULL_NETWORK_CAPACITY_DEVICE:-cuda:2}"

if [[ "$(hostname)" != "node02" ]]; then
  echo "This smoke is authorized on node02 only." >&2
  exit 2
fi
for path in "$ROOT" "$REPO" "$ASSET_DIR" "$RUN_ROOT"; do
  if [[ "$path" == /scratch* ]]; then
    echo "Full-network capacity smoke must not use /scratch." >&2
    exit 2
  fi
done

mkdir -p "$LOG_DIR"
cd "$REPO"
source "$REPO/scripts/activate_qm9_node02_local.sh"
source "$REPO/.venv/bin/activate"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$ROOT/tmp"
export XDG_CACHE_HOME="$ROOT/cache"
export MLDFT_SYMMETRIC_MATRIX_POWER_MODE=eigh_second_order_audit

if [[ ! -f "$REGISTRATION" ]]; then
  python scripts/prepare_qm9_graphformer_0028399_capacity_assets.py \
    --protocol "$PROTOCOL" \
    --output-dir "$ASSET_DIR" \
    >"$LOG_DIR/prepare_assets.log" 2>&1
fi

run_stage() {
  local initialization=$1
  local mode=$2
  local output="$RUN_ROOT/$initialization/$mode"
  if [[ -f "$output/summary.json" || -f "$output/failure.json" ]]; then
    return 0
  fi
  if ! python scripts/qm9_graphformer_full39_capacity_only.py \
    --protocol "$PROTOCOL" \
    --asset-registration "$REGISTRATION" \
    --scope full_graphformer \
    --mode "$mode" \
    --optimizer adamw \
    --initialization "$initialization" \
    --output-dir "$output" \
    --device "$DEVICE" \
    >"$LOG_DIR/${initialization}_${mode}.log" 2>&1
  then
    echo "$initialization $mode failed closed; see $output/failure.json" >&2
    return 1
  fi
}

for initialization in pretrained scratch; do
  if ! run_stage "$initialization" density_preflight; then
    continue
  fi
  if ! run_stage "$initialization" gradient_smoke; then
    continue
  fi
  run_stage "$initialization" optimize || true
done

python - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
arms = {}
for initialization in ("pretrained", "scratch"):
    arm = {}
    for stage in ("density_preflight", "gradient_smoke", "optimize"):
        directory = root / initialization / stage
        summary = directory / "summary.json"
        failure = directory / "failure.json"
        if summary.exists():
            payload = json.loads(summary.read_text())
            arm[stage] = {
                "status": payload["status"],
                "final_relative_frobenius": (
                    payload["final"].get("full39_relative_frobenius")
                    if payload.get("final")
                    else None
                ),
                "capacity_passed": payload["capacity_passed"],
                "summary": summary.resolve().as_posix(),
            }
        elif failure.exists():
            arm[stage] = {
                "status": "failed_closed",
                "failure": failure.resolve().as_posix(),
                **json.loads(failure.read_text()),
            }
        else:
            arm[stage] = {"status": "not_run"}
    arms[initialization] = arm

result = {
    "protocol_id": "qm9_graphformer_0028399_full_network_capacity_smoke_v3",
    "definition": (
        "one-parent, complete-39-direction, Hessian-only, full-Graphformer "
        "one-update capacity smoke"
    ),
    "arms": arms,
    "capacity_proved": any(
        arm.get("optimize", {}).get("capacity_passed", False)
        for arm in arms.values()
    ),
    "insufficient_capacity_conclusion_allowed": False,
    "validation_accessed": False,
    "test100_accessed": False,
}
(root / "smoke_summary.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(result, indent=2, sort_keys=True))
PY
