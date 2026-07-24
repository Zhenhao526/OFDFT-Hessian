#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/scratch/xzh/OFDFT-Hessian}
base=$repo/runs/al/free_energy_wt/multitemp_volume_scan
original=${ORIGINAL_SCAN:-$base/T0975_prepared_v1_node01}
extension=${EXTENSION_SCAN:-$base/T0975_liquid_continuation_steps600_tau2_v1_node01}
combined=${COMBINED_SCAN:-$base/T0975_solid_original_liquid_continuation_v1_node01}
scan_1050=${SCAN_1050:-$base/T1050_independent_seeds_steps600_tau2_v2_node01}
scan_1100=${SCAN_1100:-$base/T1100_independent_seeds_steps600_tau2_v2_node01}
pipeline_root=$repo/runs/al/free_energy_wt/melting_pipeline_pairv2_v1_node01
log=${extension}.recovery.log
done_file=${extension}.recovery.done
failed_file=${extension}.recovery.failed

mkdir -p "$(dirname "$extension")"
mark_failed() {
    local rc=$?
    trap - ERR
    if [[ ! -e $done_file ]]; then
        printf '%s failed exit_code=%s\n' "$(date -Iseconds)" "$rc" \
            | tee -a "$log" "$failed_file"
    fi
    exit "$rc"
}
trap mark_failed ERR

if [[ -e $done_file ]]; then
    printf '%s already_complete file=%s\n' "$(date -Iseconds)" "$done_file"
    exit 0
fi
rm -f "$failed_file"

python3 - "$original" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
solid = json.loads((root / "solid" / "nvt_volume_scan_result.json").read_text())
liquid = json.loads((root / "liquid" / "nvt_volume_scan_result.json").read_text())
if solid.get("status") != "zero_pressure_volume_verified" or not all(solid.get("checks", {}).values()):
    raise SystemExit("original solid scan is not verified")
rows = liquid.get("rows", [])
if len(rows) < 2:
    raise SystemExit("original liquid scan has too few points")
for row in rows:
    if int(row.get("max_step", -1)) < int(liquid.get("steps", 0)):
        raise SystemExit("original liquid scan is incomplete")
    if row.get("phase_status") != "liquid_verified":
        raise SystemExit("original liquid phase gate failed")
    if float(row.get("nearest_neighbor_A", 0.0)) <= 2.0:
        raise SystemExit("original liquid nearest-neighbor gate failed")
    if row.get("temperature_mean_within_25_K"):
        raise SystemExit("recovery expected a correlated temperature-only failure")
ordered = sorted(rows, key=lambda row: float(row["volume_per_atom_A3"]))
pressures = [float(row["pressure_last_half_kbar"]["mean"]) for row in ordered]
if not all(left > right for left, right in zip(pressures, pressures[1:])):
    raise SystemExit("original liquid pressure is not monotonic with volume")
if not any(left * right <= 0.0 for left, right in zip(pressures, pressures[1:])):
    raise SystemExit("original liquid pressure does not bracket zero")
PY

if [[ ! -e $extension/liquid/manifest.json ]]; then
    if [[ -e $extension/solid || -e $extension/liquid ]]; then
        printf '%s refusing_extension reason=partial_preparation\n' "$(date -Iseconds)" | tee -a "$log"
        exit 2
    fi
    env PYTHONPATH="$repo" python3 "$repo/scripts/prepare_al108_volume_scan_extension.py" \
        --parent "$original" --out "$extension" --temperature 975 --steps 600 \
        --csvr-tau 2 --seed 2026072701 \
        --config "$repo/config/abacus_wt_ti_node04_cpu12.json" >> "$log" 2>&1
fi

if [[ ! -e $extension/liquid/nvt_volume_scan_result.json ]]; then
    bash "$repo/scripts/launch_wt_volume_scan_phase_node01.sh" "$extension" liquid 12 \
        >> "$log" 2>&1
fi
python3 - "$extension/liquid/nvt_volume_scan_result.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "zero_pressure_volume_verified":
    raise SystemExit("continued liquid scan is not verified")
if not report.get("checks") or not all(report["checks"].values()):
    raise SystemExit("continued liquid checks failed")
PY

mkdir -p "$combined"
if [[ ! -e $combined/solid && ! -L $combined/solid ]]; then
    ln -s "$original/solid" "$combined/solid"
fi
if [[ ! -e $combined/liquid && ! -L $combined/liquid ]]; then
    ln -s "$extension/liquid" "$combined/liquid"
fi
python3 - "$combined" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for phase in ("solid", "liquid"):
    report = json.loads((root / phase / "nvt_volume_scan_result.json").read_text())
    if report.get("status") != "zero_pressure_volume_verified":
        raise SystemExit(f"combined {phase} scan is not verified")
    if not report.get("checks") or not all(report["checks"].values()):
        raise SystemExit(f"combined {phase} checks failed")
PY

date -Iseconds > "$done_file"
printf '%s recovered_combined_scan root=%s\n' "$(date -Iseconds)" "$combined" | tee -a "$log"
rm -f "$pipeline_root/pipeline.failed"
if ! tmux list-sessions -F '#S' 2>/dev/null | grep -qx wt_melt_pipeline; then
    tmux new-session -d -s wt_melt_pipeline \
        "cd $repo && exec env SCAN_975=$combined SCAN_1050=$scan_1050 SCAN_1100=$scan_1100 bash scripts/watch_free_energy_then_multitemp_enthalpy_node01.sh"
fi
printf '%s restarted_pipeline scan_975=%s scan_1050=%s scan_1100=%s\n' \
    "$(date -Iseconds)" "$combined" "$scan_1050" "$scan_1100" | tee -a "$log"
