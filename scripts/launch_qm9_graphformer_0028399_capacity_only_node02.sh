#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/shenwei01/xzh_node02_20260724
REPO="$ROOT/work/structures25"
PROTOCOL="$REPO/configs/audit/qm9_graphformer_0028399_full39_capacity_only_v2.yaml"
ASSET_DIR="$ROOT/artifacts/graphformer_0028399_full39_capacity_only_v2"
REGISTRATION="$ASSET_DIR/asset_registration.json"
DENSITY_RESCUE="$REPO/configs/audit/qm9_graphformer_0028399_density_solver_rescue_v1.yaml"
RUN_ROOT="$ROOT/runs/graphformer_0028399_full39_capacity_only_v2"
LOG_DIR="$RUN_ROOT/logs"
DEVICE="${CAPACITY_DEVICE:-cuda:0}"

if [[ "$(hostname)" != "node02" ]]; then
  echo "This capacity-only protocol is authorized on node02 only." >&2
  exit 2
fi
if [[ "$ROOT" == /scratch* || "$REPO" == /scratch* || "$RUN_ROOT" == /scratch* ]]; then
  echo "Capacity-only paths must not use /scratch." >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
exec 9>"$RUN_ROOT/supervisor.lock"
if ! flock -n 9; then
  echo "A capacity-only supervisor already owns the lock." >&2
  exit 3
fi

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

passed() {
  local summary=$1
  python - "$summary" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
raise SystemExit(0 if json.loads(path.read_text()).get("capacity_passed") else 1)
PY
}

run_arm() {
  local mode=$1
  local scope=$2
  local optimizer=$3
  local output=$4
  if [[ -f "$output/summary.json" ]]; then
    return
  fi
  python scripts/qm9_graphformer_full39_capacity_only.py \
    --protocol "$PROTOCOL" \
    --asset-registration "$REGISTRATION" \
    --density-rescue "$DENSITY_RESCUE" \
    --mode "$mode" \
    --scope "$scope" \
    --optimizer "$optimizer" \
    --output-dir "$output" \
    --device "$DEVICE" \
    >"$LOG_DIR/${scope}_${mode}_${optimizer}.log" 2>&1
}

scopes=(energy_readout readout_last1 readout_last2 full_graphformer)
optimizers=(adamw lbfgs levenberg_marquardt)

for scope in "${scopes[@]}"; do
  for optimizer in "${optimizers[@]}"; do
    output="$RUN_ROOT/$scope/$optimizer"
    run_arm optimize "$scope" "$optimizer" "$output"
    if passed "$output/summary.json"; then
      cp "$output/CAPACITY_PASSED_FROZEN.json" \
        "$RUN_ROOT/CAPACITY_PASSED_FROZEN.json"
      exit 0
    fi
  done
  linearized="$RUN_ROOT/$scope/linearized_fresh"
  run_arm linearize "$scope" adamw "$linearized"
done

python - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summaries = []
for path in sorted(root.glob("*/*/summary.json")):
    payload = json.loads(path.read_text())
    if payload.get("mode") == "optimize":
        summaries.append(
            {
                "path": path.as_posix(),
                "scope": payload["parameter_scope"],
                "optimizer": payload["optimizer"],
                "status": payload["status"],
                "final_relative_frobenius": payload["final"][
                    "full39_relative_frobenius"
                ],
                "capacity_passed": payload["capacity_passed"],
            }
        )
result = {
    "capacity_passed": any(row["capacity_passed"] for row in summaries),
    "inadequate_capacity_conclusion_allowed": False,
    "reason": (
        "A no-pass sweep is not by itself sufficient: the full-Graphformer "
        "optimizers and converged Jacobian minimum must all certify convergence."
    ),
    "arms": summaries,
    "validation_accessed": False,
    "test100_accessed": False,
}
(root / "sweep_status.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n"
)
PY
