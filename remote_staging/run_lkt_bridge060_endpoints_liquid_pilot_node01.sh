#!/usr/bin/env bash
set -euo pipefail

workspace=/home/shenwei01/WT_Al_melting_workspace_20260724
repository=$workspace/repository
run_root=$workspace/runs/xwm_lkt_20260727
root=$run_root/lkt/free_energy_T0900
confirmation=$run_root/lkt/volume_confirmation_T0900_steps600_r2_tau1_node01
solid_json=$root/pair_reference_v1.json
solid_dat=$root/pair_reference_v1.dat
bridge_json=$root/liquid_variance_bridge_alpha060_v1.json
bridge_dat=$root/liquid_variance_bridge_alpha060_v1.dat
bridge_sha=1fec6fe7a6a0d0104a0d76a3266419da83baa7b4a002c40b02fa7844ecc8ea6f
old_merged=$root/ti_pilot_merged_phase_specific_proxy6_verified_v3/merged_pilot_analysis.json
endpoints=$root/endpoints_phase_specific_v3_bridge060
pilot=$root/ti_pilot_phase_specific_bridge060_lambda9_steps300_v1
merged=$root/ti_pilot_merged_phase_specific_bridge060_verified_v1
design=$root/ti_formal_design_phase_specific_bridge060_v1.json
log=$root/lkt_bridge060_abacus_pilot_pipeline.log
python=/home/shenwei01/wt_melting_runtime_20260724/conda_prefix/bin/python

exec > >(tee -a "$log") 2>&1

mark_failed() {
  local rc=$?
  trap - ERR
  printf '%s failed exit_code=%s\n' "$(date -Iseconds)" "$rc" \
    >"$root/lkt_bridge060_abacus_pilot_pipeline.failed"
  exit "$rc"
}
trap mark_failed ERR

[[ -x $python ]]
[[ -f $confirmation/zero_pressure_confirmation_summary.json ]]
[[ -f $solid_json ]]
[[ -f $solid_dat ]]
[[ -f $bridge_json ]]
[[ -f $old_merged ]]
[[ ! -e $bridge_dat ]]
[[ ! -e $endpoints ]]
[[ ! -e $pilot ]]
[[ ! -e $merged ]]
[[ ! -e $design ]]
if pgrep -f '[a]bacus_pw' >/dev/null; then
  echo "refusing to share node01 with another ABACUS job" >&2
  exit 2
fi
printf '%s  %s\n' "$bridge_sha" "$bridge_json" | sha256sum -c -

cd "$repository"
env PYTHONPATH=. "$python" - "$bridge_json" "$confirmation" "$old_merged" <<'PY'
import json
import sys
from pathlib import Path

bridge = json.loads(Path(sys.argv[1]).read_text())
assert bridge["status"] == "verified"
assert bridge["reference_gate_passed"] is True
assert bridge["short_range_guard_passed"] is True
assert bridge["target_kedf"] == "lkt"
assert bridge["phase"] == "liquid"
assert bridge["alpha_correction"] == 0.6
confirmation = json.loads(
    (Path(sys.argv[2]) / "zero_pressure_confirmation_summary.json").read_text()
)
assert confirmation["status"] == "all_confirmations_passed"
assert confirmation["target_kedf"] == "lkt"
assert confirmation["temperature_K"] == 900.0
merged = json.loads(Path(sys.argv[3]).read_text())
assert merged["status"] == "pilot_grid_verified"
assert merged["phase_results"]["solid"]["status"] == "verified"
PY

env PYTHONPATH=. "$python" scripts/export_pair_reference_for_abacus.py \
  "$bridge_json" "$bridge_dat"
sha256sum "$bridge_json" "$bridge_dat" >"$root/liquid_variance_bridge_alpha060_v1.sha256"

env PYTHONPATH=. "$python" scripts/prepare_ti_endpoints_from_confirmations.py \
  --confirmation-root "$confirmation" \
  --out "$endpoints" \
  --solid-pair-model "$solid_json" \
  --liquid-pair-model "$bridge_json" \
  --steps 10 \
  --config config/abacus_lkt_ti_cpu12.json
bash remote_staging/run_kedf_ti_endpoints_node_local.sh \
  "$endpoints" "$repository" \
  "$solid_json" "$solid_dat" "$bridge_json" "$bridge_dat" "$python"

env PYTHONPATH=. "$python" scripts/prepare_ti_pilot_from_endpoints.py \
  --endpoint-root "$endpoints" \
  --out "$pilot" \
  --solid-pair-model "$solid_json" \
  --liquid-pair-model "$bridge_json" \
  --steps 300 \
  --csvr-tau 5 \
  --config config/abacus_lkt_ti_cpu12.json

env PYTHONPATH=. "$python" - "$pilot" "$bridge_json" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
bridge = str(Path(sys.argv[2]).resolve())
manifest = json.loads((root / "pilot_manifest.json").read_text())
assert manifest["target_kedf"] == "lkt"
assert manifest["steps"] == 300
assert manifest["csvr_tau_fs"] == 5.0
liquid = json.loads((root / "liquid" / "manifest.json").read_text())
assert liquid["pair_model"] == bridge
assert len(liquid["windows"]) == 9
for window in liquid["windows"]:
    run = root / "liquid" / window["label"]
    metadata = json.loads((run / "metadata.json").read_text())
    assert metadata["target_kedf"] == "lkt"
    assert metadata["phase"] == "liquid"
    assert metadata["csvr_tau"] == 5.0
    assert metadata["pair_model"] == bridge
    assert not list(run.glob("OUT.*"))
PY

bash remote_staging/run_kedf_ti_selected_extensions_node_local.sh \
  "$pilot" "$repository" "$bridge_dat" "$pilot/liquid"
touch "$pilot/liquid_pilot.completed"

mapfile -t replacements < <(
  env PYTHONPATH=. "$python" - "$old_merged" "$pilot/liquid" <<'PY'
import json
import sys
from pathlib import Path

old = json.loads(Path(sys.argv[1]).read_text())
for row in old["phase_results"]["solid"]["window_results"]:
    print(f"solid:{row['label']}={row['run']}")
liquid = Path(sys.argv[2])
manifest = json.loads((liquid / "manifest.json").read_text())
for window in manifest["windows"]:
    print(f"liquid:{window['label']}={liquid / window['label']}")
PY
)
base=$(
  "$python" - "$old_merged" <<'PY'
import json
import sys
print(json.loads(open(sys.argv[1]).read())["base"])
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
grep -q '"status": "pilot_grid_verified"' "$merged/merged_pilot_analysis.json"

if env PYTHONPATH=. "$python" scripts/design_kedf_ti_production.py \
  --method "lkt_bridge060=$merged/merged_pilot_analysis.json" \
  --out "$design" \
  --candidate-steps 2000 3000 4000 6000 9000 12000 18000 24000; then
  touch "$root/lkt_bridge060_formal_design.done"
else
  touch "$root/lkt_bridge060_formal_design.needs_refinement"
fi

find "$endpoints" "$pilot" "$merged" -type f \
  \( -name INPUT -o -name STRU -o -name KPT -o -name metadata.json \
     -o -name manifest.json -o -name endpoint_manifest.json \
     -o -name endpoint_validation.json -o -name phase_analysis.json \
     -o -name ti_analysis.json -o -name merged_pilot_analysis.json \) \
  -print0 | sort -z | xargs -0 sha256sum \
  >"$root/lkt_bridge060_abacus_pilot_SHA256SUMS"
touch "$root/lkt_bridge060_abacus_pilot_pipeline.done"
