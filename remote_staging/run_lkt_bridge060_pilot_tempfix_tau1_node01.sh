#!/usr/bin/env bash
set -euo pipefail

workspace=/home/shenwei01/WT_Al_melting_workspace_20260724
repository=$workspace/repository
root=$workspace/runs/xwm_lkt_20260727/lkt/free_energy_T0900
old_merged=$root/ti_pilot_merged_phase_specific_proxy6_verified_v3/merged_pilot_analysis.json
parent=$root/ti_pilot_phase_specific_bridge060_lambda9_steps300_v1/liquid
batch=$root/ti_pilot_tempfix_phase_specific_bridge060_tau1_v1
run_root=$batch/liquid_fix
bridge_dat=$root/liquid_variance_bridge_alpha060_v1.dat
merged=$root/ti_pilot_merged_phase_specific_bridge060_verified_v2_tau1
design=$root/ti_formal_design_phase_specific_bridge060_v2_tau1.json
runner=$repository/remote_staging/run_kedf_ti_selected_extensions_node_local.sh
python=$workspace/.venv-reference/bin/python
bridge_sha=1fec6fe7a6a0d0104a0d76a3266419da83baa7b4a002c40b02fa7844ecc8ea6f
labels=(lambda_0p250 lambda_0p875 lambda_1p000)
log=$root/lkt_bridge060_pilot_tempfix_tau1_retry.log

exec > >(tee -a "$log") 2>&1

mark_failed() {
  local rc=$?
  trap - ERR
  printf '%s failed exit_code=%s\n' "$(date -Iseconds)" "$rc" \
    >"$root/lkt_bridge060_pilot_tempfix_tau1_retry.failed"
  exit "$rc"
}
trap mark_failed ERR

[[ -x $python ]]
[[ -x $runner ]]
[[ -f $old_merged ]]
[[ -f $parent/manifest.json ]]
[[ -f $bridge_dat ]]
[[ ! -e $batch ]]
[[ ! -e $merged ]]
[[ ! -e $design ]]
if pgrep -f '[a]bacus_pw' >/dev/null; then
  echo "refusing to share node01 with another ABACUS job" >&2
  exit 2
fi
printf '%s  %s\n' "$bridge_sha" \
  "$root/liquid_variance_bridge_alpha060_v1.json" | sha256sum -c -

cd "$repository"

mapfile -t parent_labels < <(
  "$python" - "$parent/manifest.json" <<'PY'
import json
import sys

for window in json.load(open(sys.argv[1]))["windows"]:
    print(window["label"])
PY
)
for label in "${parent_labels[@]}"; do
  run=$parent/$label
  env PYTHONPATH=. "$python" scripts/analyze_phase_run.py \
    "$run" --expected liquid --thermalized-initial \
    --out "$run/phase_analysis.json" >/dev/null
  env PYTHONPATH=. "$python" - "$run/phase_analysis.json" "$label" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
label = sys.argv[2]
if label == "lambda_1p000":
    checks = report["phase_gate"]
    assert report["status"] == "liquid_not_verified"
    assert checks["final_MSD_gt_0_3_A2"] is False
    assert all(
        passed
        for name, passed in checks.items()
        if name != "final_MSD_gt_0_3_A2"
    ), (label, checks)
else:
    assert report["status"] == "liquid_verified", (
        label,
        report["status"],
        report.get("phase_gate"),
    )
assert report["trajectory"]["minimum_nearest_neighbor_A"] > 2.0
PY
done

env PYTHONPATH=. "$python" scripts/prepare_ti_window_extensions.py \
  --phase-root "liquid=$parent" \
  --out "$run_root" \
  --steps 300 \
  --csvr-tau 1 \
  --seed 2026073001 \
  --ranks 24 \
  --phases liquid \
  --labels "${labels[@]}" \
  --config config/abacus_lkt_ti_cpu12.json

env PYTHONPATH=. "$python" - \
  "$run_root/liquid/manifest.json" "${labels[@]}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
expected = set(sys.argv[2:])
manifest = json.loads(manifest_path.read_text())
assert manifest["target_kedf"] == "lkt"
assert manifest["phase"] == "liquid"
assert manifest["steps"] == 300
assert manifest["mpi_ranks"] == 24
assert {row["label"] for row in manifest["windows"]} == expected
for label in expected:
    run = manifest_path.parent / label
    metadata = json.loads((run / "metadata.json").read_text())
    assert metadata["source_step"] == 295
    assert metadata["source_velocities_discarded"] is False
    assert metadata["csvr_tau"] == 1.0
    assert metadata["target_kedf"] == "lkt"
    assert metadata["phase"] == "liquid"
    assert not list(run.glob("OUT.*"))
PY

find "$batch" -type f \
  \( -name INPUT -o -name STRU -o -name KPT -o -name metadata.json \
     -o -name manifest.json \) \
  -print0 | sort -z | xargs -0 sha256sum >"$batch/PRELAUNCH_SHA256SUMS"

KEDF_TI_CPU_RANGES="0-23 25-48 50-73" \
  bash "$runner" "$run_root" "$repository" "$bridge_dat" "$run_root/liquid"

for label in "${labels[@]}"; do
  run=$run_root/liquid/$label
  env PYTHONPATH=. "$python" scripts/analyze_phase_run.py \
    "$run" --expected liquid --thermalized-initial \
    --out "$run/phase_analysis.json" >/dev/null
  env PYTHONPATH=. "$python" - "$run/phase_analysis.json" "$label" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
assert report["status"] == "liquid_verified", (
    sys.argv[2],
    report["status"],
    report.get("phase_gate"),
)
assert report["trajectory"]["minimum_nearest_neighbor_A"] > 2.0
PY
done

mapfile -t replacements < <(
  env PYTHONPATH=. "$python" - \
    "$old_merged" "$parent" "$run_root/liquid" <<'PY'
import json
import sys
from pathlib import Path

old = json.loads(Path(sys.argv[1]).read_text())
parent = Path(sys.argv[2])
correction = Path(sys.argv[3])
corrected = {
    row["label"] for row in json.loads((correction / "manifest.json").read_text())["windows"]
}
for row in old["phase_results"]["solid"]["window_results"]:
    print(f"solid:{row['label']}={row['run']}")
for row in json.loads((parent / "manifest.json").read_text())["windows"]:
    label = row["label"]
    run = correction / label if label in corrected else parent / label
    print(f"liquid:{label}={run}")
PY
)
base=$(
  "$python" - "$old_merged" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1]))["base"])
PY
)
merge_args=()
for replacement in "${replacements[@]}"; do
  merge_args+=(--replacement "$replacement")
done
env PYTHONPATH=. "$python" scripts/merge_kedf_ti_pilot_replacements.py \
  --base "$base" \
  --out "$merged" \
  "${merge_args[@]}"

if grep -q '"status": "pilot_grid_verified"' \
  "$merged/merged_pilot_analysis.json"; then
  touch "$root/lkt_bridge060_pilot_tempfix_tau1_retry.verified"
else
  touch "$root/lkt_bridge060_pilot_tempfix_tau1_retry.needs_refinement"
  exit 3
fi

if env PYTHONPATH=. "$python" scripts/design_kedf_ti_production.py \
  --method "lkt_bridge060=$merged/merged_pilot_analysis.json" \
  --out "$design" \
  --candidate-steps 2000 3000 4000 6000 9000 12000 18000 24000; then
  touch "$root/lkt_bridge060_formal_design_v2.done"
else
  touch "$root/lkt_bridge060_formal_design_v2.needs_refinement"
fi

find "$parent" "$batch" "$merged" -type f \
  \( -name INPUT -o -name STRU -o -name KPT -o -name metadata.json \
     -o -name manifest.json -o -name phase_analysis.json \
     -o -name ti_analysis.json -o -name merged_pilot_analysis.json \) \
  -print0 | sort -z | xargs -0 sha256sum \
  >"$root/lkt_bridge060_pilot_tempfix_tau1_SHA256SUMS"
touch "$root/lkt_bridge060_pilot_tempfix_tau1_retry.done"
