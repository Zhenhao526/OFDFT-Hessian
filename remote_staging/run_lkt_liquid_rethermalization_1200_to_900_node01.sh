#!/usr/bin/env bash
set -euo pipefail

workspace=/home/shenwei01/WT_Al_melting_workspace_20260724
repository=$workspace/repository
root=$workspace/runs/xwm_lkt_20260727/lkt
source_run=$root/volume_confirmation_T0900_steps600_r2_tau1_node01/liquid
protocol=$root/free_energy_T0900/liquid_rethermalization_T1200_to_T0900_v1
heat=$protocol/heat_T1200_steps1000_tau2
cool=$protocol/cool_T1200_to_T0900_steps1000_tau2
hold=$protocol/hold_T0900_steps1200_tau1
config=$repository/config/abacus_lkt_ti_cpu12.json
python=$workspace/.venv-reference/bin/python
source_md_sha=1c86e048e7ec363cdc50a3c5197d397fc4c569c11d1d7f6481928a8011e20acd
log=$protocol/pipeline.log

mark_failed() {
  local rc=$?
  trap - ERR
  printf '%s failed exit_code=%s\n' "$(date -Iseconds)" "$rc" \
    >"$protocol/pipeline.failed"
  exit "$rc"
}
trap mark_failed ERR

[[ -x $python ]]
[[ -f $config ]]
[[ -f $source_run/phase_analysis.json ]]
[[ ! -e $protocol ]]
if pgrep -f '[a]bacus_pw' >/dev/null; then
  echo "refusing to share node01 with another ABACUS job" >&2
  exit 2
fi
source_md=$(find "$source_run" -path '*/OUT.*/MD_dump' -type f)
[[ $(printf '%s\n' "$source_md" | wc -l) -eq 1 ]]
printf '%s  %s\n' "$source_md_sha" "$source_md" | sha256sum -c -

mkdir -p "$protocol"
exec > >(tee -a "$log") 2>&1
cd "$repository"

env PYTHONPATH=. "$python" - "$source_run/phase_analysis.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
assert report["status"] == "liquid_verified"
assert report["trajectory"]["minimum_nearest_neighbor_A"] > 2.0
PY

prepare_stage() {
  local source=$1
  local out=$2
  local start=$3
  local end=$4
  local steps=$5
  local tau=$6
  local seed=$7
  env PYTHONPATH=. "$python" scripts/prepare_kedf_temperature_continuation.py \
    --source "$source" \
    --out "$out" \
    --config "$config" \
    --phase liquid \
    --volume 18.735 \
    --start-temperature "$start" \
    --end-temperature "$end" \
    --steps "$steps" \
    --csvr-tau "$tau" \
    --seed "$seed" \
    --ranks 72
}

gate_stage() {
  local run=$1
  local steps=$2
  local target=$3
  local mode=$4
  env PYTHONPATH=. "$python" scripts/analyze_phase_run.py \
    "$run" --expected liquid --thermalized-initial \
    --out "$run/phase_analysis.json" >/dev/null
  env PYTHONPATH=. "$python" - "$run" "$steps" "$target" "$mode" <<'PY'
import json
import sys
from pathlib import Path

from scripts.analyze_two_phase_run import parse_md_log, series_stats

run = Path(sys.argv[1])
expected_steps = int(sys.argv[2])
target = float(sys.argv[3])
mode = sys.argv[4]
phase = json.loads((run / "phase_analysis.json").read_text())
assert phase["status"] == "liquid_verified", (
    phase["status"],
    phase.get("phase_gate"),
)
assert phase["trajectory"]["minimum_nearest_neighbor_A"] > 2.0
logs = sorted(run.glob("OUT.*/running_md.log"))
rows, max_step = parse_md_log(logs[-1])
assert max_step >= expected_steps
selected = rows[len(rows) // 2 :] if mode == "half" else rows[-20:]
temperature = series_stats([row["temperature_K"] for row in selected])
assert abs(temperature["mean"] - target) <= 20.0, temperature
summary = {
    "status": "verified",
    "max_step": max_step,
    "temperature_gate_mode": mode,
    "temperature_K": temperature,
    "phase_status": phase["status"],
    "minimum_nearest_neighbor_A": phase["trajectory"][
        "minimum_nearest_neighbor_A"
    ],
    "diffusion_MSD_slope_A2_per_step": phase["trajectory"][
        "diffusion_MSD_slope_A2_per_step"
    ],
    "final_MSD_A2": phase["trajectory"]["non_affine_MSD_A2"],
}
(run / "rethermalization_gate.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
  if grep -RniE \
    'convergence has not been achieved|nan|fatal error' \
    "$run"/OUT.*/running_md.log "$run"/OUT.*/warning.log 2>/dev/null; then
    echo "electronic or numerical failure detected in $run" >&2
    return 1
  fi
}

run_stage() {
  local run=$1
  (
    cd "$run"
    exec taskset -c 0-73 env \
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      ./run_local.sh
  ) >"$run/run.stdout" 2>&1
}

prepare_stage "$source_run" "$heat" 1200 1200 1000 2 2026073050
run_stage "$heat"
gate_stage "$heat" 1000 1200 half

prepare_stage "$heat" "$cool" 1200 900 1000 2 2026073051
run_stage "$cool"
gate_stage "$cool" 1000 900 tail

prepare_stage "$cool" "$hold" 900 900 1200 1 2026073052
run_stage "$hold"
gate_stage "$hold" 1200 900 half

find "$protocol" -type f \
  \( -name INPUT -o -name STRU -o -name KPT -o -name metadata.json \
     -o -name temperature_continuation_manifest.json \
     -o -name phase_analysis.json -o -name rethermalization_gate.json \) \
  -print0 | sort -z | xargs -0 sha256sum >"$protocol/SHA256SUMS"
touch "$protocol/pipeline.done"
