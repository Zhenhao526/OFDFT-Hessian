#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/scratch/xzh/OFDFT-Hessian}
base=$repo/runs/al/free_energy_wt/multitemp_volume_scan
original=${ORIGINAL_SCAN:-$base/T1100_independent_seeds_steps600_tau2_v2_node01}
recovery=${RECOVERY_SCAN:-$base/T1100_liquid_freshvel_tau1_steps1000_v1_node01}
combined=${COMBINED_SCAN:-$base/T1100_solid_original_liquid_freshvel_tau1_steps1000_v1_node01}
scan_975=${SCAN_975:-$base/T0975_solid_original_liquid_continuation_v1_node01}
scan_1050=${SCAN_1050:-$base/T1050_independent_seeds_steps600_tau2_v2_node01}
pipeline_root=${PIPELINE_ROOT:-$repo/runs/al/free_energy_wt/melting_pipeline_pairv2_recovery_T1100_v2_node01}
launcher_session=${LAUNCHER_SESSION:-wt_T1100_liq_recover}
pipeline_session=${PIPELINE_SESSION:-wt_melt_pipeline}
log=$recovery/recovery.log
done_file=$recovery/recovery.done
failed_file=$recovery/recovery.failed

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
printf '%s waiting_for_recovery session=%s\n' \
    "$(date -Iseconds)" "$launcher_session" > "$log"

while tmux list-sessions -F '#S' 2>/dev/null | grep -qx "$launcher_session"; do
    sleep 60
done

python3 - "$original" "$recovery" <<'PY' >> "$log" 2>&1
import json
import sys
from pathlib import Path

original = Path(sys.argv[1])
recovery = Path(sys.argv[2])
solid = json.loads((original / "solid" / "nvt_volume_scan_result.json").read_text())
if solid.get("status") != "zero_pressure_volume_verified":
    raise SystemExit("original solid zero-pressure volume is not verified")
if not solid.get("checks") or not all(solid["checks"].values()):
    raise SystemExit("original solid zero-pressure checks failed")

original_liquid = json.loads(
    (original / "liquid" / "nvt_volume_scan_result.json").read_text()
)
positive = next(
    (row for row in original_liquid.get("rows", []) if row["label"] == "vpa_18p850"),
    None,
)
if positive is None:
    raise SystemExit("missing original positive-pressure liquid point")
if int(positive.get("max_step", -1)) < 600:
    raise SystemExit("original positive-pressure liquid point is incomplete")
if positive.get("phase_status") != "liquid_verified":
    raise SystemExit("original positive-pressure point is not a verified liquid")
if positive.get("temperature_mean_within_25_K") is not True:
    raise SystemExit("original positive-pressure point failed its temperature gate")
if float(positive.get("pressure_last_half_kbar", {}).get("mean", 0.0)) <= 0.0:
    raise SystemExit("original liquid recovery point is not at positive pressure")

liquid = json.loads((recovery / "liquid" / "nvt_volume_scan_result.json").read_text())
rows = liquid.get("rows", [])
if len(rows) != 2:
    raise SystemExit("recovery result does not contain two points")
for row in rows:
    if int(row.get("max_step", -1)) < 1000:
        raise SystemExit(f"incomplete recovery point: {row.get('label')}")
    if row.get("phase_status") != "liquid_verified":
        raise SystemExit(f"recovery point is not liquid: {row.get('label')}")
    if row.get("temperature_mean_within_25_K") is not True:
        raise SystemExit(f"recovery temperature gate failed: {row.get('label')}")
    if float(row.get("nearest_neighbor_A", 0.0)) <= 2.0:
        raise SystemExit(f"recovery nearest-neighbor gate failed: {row.get('label')}")
manifest = json.loads((recovery / "liquid" / "manifest.json").read_text())
if int(manifest.get("steps", -1)) != 1000:
    raise SystemExit("recovery did not request 1000 steps")
seeds = [int(point["md_seed"]) for point in manifest.get("points", [])]
if len(seeds) != 2 or len(set(seeds)) != 2:
    raise SystemExit("recovery does not have two independent point seeds")
print(json.dumps({"solid_status": solid["status"], "recovery_rows": len(rows), "seeds": seeds}))
PY

mkdir -p "$combined/liquid"
if [[ ! -e $combined/solid && ! -L $combined/solid ]]; then
    ln -s "$original/solid" "$combined/solid"
fi
if [[ ! -e $combined/liquid/vpa_18p850 && ! -L $combined/liquid/vpa_18p850 ]]; then
    ln -s "$original/liquid/vpa_18p850" "$combined/liquid/vpa_18p850"
fi
for label in vpa_19p150 vpa_19p450; do
    if [[ ! -e $combined/liquid/$label && ! -L $combined/liquid/$label ]]; then
        ln -s "$recovery/liquid/$label" "$combined/liquid/$label"
    fi
done

python3 - "$original" "$recovery" "$combined" <<'PY'
import json
import sys
from pathlib import Path

original, recovery, combined = map(Path, sys.argv[1:])
original_manifest = json.loads((original / "liquid" / "manifest.json").read_text())
recovery_manifest = json.loads((recovery / "liquid" / "manifest.json").read_text())
positive = next(
    point for point in original_manifest["points"] if point["label"] == "vpa_18p850"
)
manifest = {
    "phase": "liquid",
    "target_kedf": "wt",
    "target_temperature_K": 1100.0,
    "natoms": int(recovery_manifest["natoms"]),
    "source_scans": {
        "positive_pressure": str((original / "liquid").resolve()),
        "long_recovery": str((recovery / "liquid").resolve()),
    },
    "thermalized_initial": True,
    "steps": 600,
    "point_requested_steps": {
        "vpa_18p850": 600,
        "vpa_19p150": 1000,
        "vpa_19p450": 1000,
    },
    "points": [positive, *recovery_manifest["points"]],
}
(combined / "liquid" / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

cd "$repo"
env PYTHONPATH="$repo" python3 scripts/prepare_al108_volume_scan.py analyze-md \
    "$combined/liquid" >> "$log" 2>&1

python3 - "$combined" <<'PY' >> "$log" 2>&1
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for phase in ("solid", "liquid"):
    report = json.loads((root / phase / "nvt_volume_scan_result.json").read_text())
    if report.get("status") != "zero_pressure_volume_verified":
        raise SystemExit(f"combined {phase} scan is not verified")
    checks = report.get("checks", {})
    if not checks or not all(checks.values()):
        raise SystemExit(f"combined {phase} checks failed")
PY

date -Iseconds > "$done_file"
printf '%s recovered_combined_scan root=%s\n' \
    "$(date -Iseconds)" "$combined" | tee -a "$log"
if ! tmux list-sessions -F '#S' 2>/dev/null | grep -qx "$pipeline_session"; then
    tmux new-session -d -s "$pipeline_session" \
        "cd $repo && exec env PIPELINE_ROOT=$pipeline_root SCAN_975=$scan_975 SCAN_1050=$scan_1050 SCAN_1100=$combined bash scripts/watch_free_energy_then_multitemp_enthalpy_node01.sh"
fi
printf '%s restarted_pipeline scan_975=%s scan_1050=%s scan_1100=%s pipeline_root=%s\n' \
    "$(date -Iseconds)" "$scan_975" "$scan_1050" "$combined" "$pipeline_root" \
    | tee -a "$log"
