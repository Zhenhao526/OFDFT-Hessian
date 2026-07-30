#!/usr/bin/env bash
set -euo pipefail

workspace=${WORKSPACE:-/home/shenwei01/WT_Al_melting_workspace_20260724}
repository=${REPO:-$workspace/repository}
run_root=${RUN_ROOT:-$workspace/runs/xwm_lkt_20260727}
root=$run_root/xwm/free_energy_T0900
confirmation=$run_root/xwm/zero_pressure_T0900_merged_v1
solid_json=$root/pair_reference_v1.json
solid_dat=$root/pair_reference_v1.dat
bridge_scan=$root/liquid_variance_bridge_alpha_scan_v1
bridge_json=$bridge_scan/alpha_0p900/model.json
bridge_dat=$root/liquid_variance_bridge_alpha090_v1.dat
bridge_sha=b207a394c9ba0ee652596156425f7ed4b93b4e75d3f4b8dcdcbd6871627cd89a
classical=$root/liquid_proxy14_to_bridge090_ti_lambda9_steps6000_v1
old_merged=$root/ti_pilot_merged_phase_specific_proxy14_v1/merged_pilot_analysis.json
endpoints=$root/endpoints_phase_specific_v3_bridge090
pilot=$root/ti_pilot_phase_specific_bridge090_lambda9_steps300_v1
merged=$root/ti_pilot_merged_phase_specific_bridge090_v1
design=$root/ti_formal_design_phase_specific_bridge090_v1.json
log=$root/xwm_bridge090_abacus_pilot_pipeline.log
python=${PYTHON:-$workspace/.venv-reference/bin/python}
done_file=$root/xwm_bridge090_abacus_pilot_pipeline.done
failed_file=$root/xwm_bridge090_abacus_pilot_pipeline.failed

exec > >(tee -a "$log") 2>&1

mark_failed() {
  local rc=$?
  trap - ERR
  printf '%s failed exit_code=%s\n' "$(date -Iseconds)" "$rc" \
    >"$failed_file"
  exit "$rc"
}
trap mark_failed ERR

command -v "$python" >/dev/null
[[ -f $confirmation/zero_pressure_confirmation_summary.json ]]
[[ -f $solid_json ]]
[[ -f $solid_dat ]]
[[ -f $bridge_json ]]
[[ -f $bridge_scan/bridge_alpha_scan_summary.json ]]
[[ -f $classical/overlap_refinement_summary.json ]]
[[ -f $classical/overlap_refinement.done ]]
[[ -f $old_merged ]]
[[ ! -e $pilot ]]
[[ ! -e $merged ]]
[[ ! -e $design ]]
[[ ! -e $done_file ]]
[[ ! -e $failed_file ]]
if pgrep -f '[a]bacus_pw' >/dev/null; then
  echo "refusing to share node01 with another ABACUS job" >&2
  exit 2
fi
printf '%s  %s\n' "$bridge_sha" "$bridge_json" | sha256sum -c -

cd "$repository"
env PYTHONPATH=. "$python" - \
  "$bridge_json" \
  "$bridge_scan/bridge_alpha_scan_summary.json" \
  "$classical/overlap_refinement_summary.json" \
  "$confirmation" \
  "$old_merged" <<'PY'
import json
import sys
from pathlib import Path

bridge = json.loads(Path(sys.argv[1]).read_text())
scan = json.loads(Path(sys.argv[2]).read_text())
classical = json.loads(Path(sys.argv[3]).read_text())
confirmation = json.loads(
    (Path(sys.argv[4]) / "zero_pressure_confirmation_summary.json").read_text()
)
merged = json.loads(Path(sys.argv[5]).read_text())
assert bridge["status"] == "verified"
assert bridge["reference_gate_passed"] is True
assert bridge["short_range_guard_passed"] is True
assert bridge["target_kedf"] == "xwm"
assert bridge["phase"] == "liquid"
assert bridge["alpha_correction"] == 0.9
assert scan["status"] == "verified"
assert scan["recommended_bridge"]["label"] == "alpha_0p900"
assert scan["recommended_bridge"]["eligible"] is True
assert classical["status"] == "verified"
assert classical["target_kedf"] == "xwm"
assert classical["checks"]["seventeen_windows_complete"] is True
assert classical["checks"]["all_liquid_verified"] is True
assert classical["checks"]["three_discard_reports_verified"] is True
assert classical["checks"]["discard_spread_le_2_mev_per_atom"] is True
assert confirmation["status"] == "all_confirmations_passed"
assert confirmation["target_kedf"] == "xwm"
assert confirmation["temperature_K"] == 900.0
assert merged["status"] == "pilot_grid_verified"
assert merged["phase_results"]["solid"]["status"] == "verified"
PY

env PYTHONPATH=. "$python" scripts/export_pair_reference_for_abacus.py \
  "$bridge_json" "$bridge_dat"
sha256sum "$bridge_json" "$bridge_dat" \
  >"$root/liquid_variance_bridge_alpha090_v1.sha256"

if [[ ! -f $endpoints/endpoint_pipeline.done ]]; then
  if [[ ! -e $endpoints ]]; then
    env PYTHONPATH=. "$python" scripts/prepare_ti_endpoints_from_confirmations.py \
      --confirmation-root "$confirmation" \
      --out "$endpoints" \
      --solid-pair-model "$solid_json" \
      --liquid-pair-model "$bridge_json" \
      --steps 10 \
      --config config/abacus_xwm_ti_cpu12.json
    bash remote_staging/run_kedf_ti_endpoints_node_local.sh \
      "$endpoints" "$repository" \
      "$solid_json" "$solid_dat" "$bridge_json" "$bridge_dat" "$python"
  else
    [[ -f $endpoints/endpoint_manifest.json ]]
    for phase in solid liquid; do
      pair_json=$solid_json
      [[ $phase == liquid ]] && pair_json=$bridge_json
      env PYTHONPATH=. "$python" scripts/validate_al108_ti_endpoints.py \
        "$endpoints/$phase" --pair-model "$pair_json" --expected-steps 10
      grep -q '"status": "endpoint_validation_passed"' \
        "$endpoints/$phase/endpoint_validation.json"
    done
    find "$endpoints" -type f \
      \( -name INPUT -o -name STRU -o -name KPT -o -name metadata.json \
         -o -name manifest.json -o -name endpoint_manifest.json \
         -o -name endpoint_validation.json \) \
      -print0 | sort -z | xargs -0 sha256sum >"$endpoints/SHA256SUMS"
    touch "$endpoints/endpoint_pipeline.done"
  fi
fi

env PYTHONPATH=. "$python" scripts/prepare_ti_pilot_from_endpoints.py \
  --endpoint-root "$endpoints" \
  --out "$pilot" \
  --solid-pair-model "$solid_json" \
  --liquid-pair-model "$bridge_json" \
  --steps 300 \
  --csvr-tau 5 \
  --config config/abacus_xwm_ti_cpu12.json

env PYTHONPATH=. "$python" - "$pilot" "$bridge_json" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
bridge = str(Path(sys.argv[2]).resolve())
manifest = json.loads((root / "pilot_manifest.json").read_text())
assert manifest["target_kedf"] == "xwm"
assert manifest["steps"] == 300
assert manifest["csvr_tau_fs"] == 5.0
liquid = json.loads((root / "liquid" / "manifest.json").read_text())
assert liquid["pair_model"] == bridge
assert len(liquid["windows"]) == 9
for window in liquid["windows"]:
    run = root / "liquid" / window["label"]
    metadata = json.loads((run / "metadata.json").read_text())
    assert metadata["target_kedf"] == "xwm"
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
grep -q '"status": "pilot_grid_verified"' \
  "$merged/merged_pilot_analysis.json"

if env PYTHONPATH=. "$python" scripts/design_kedf_ti_production.py \
  --method "xwm_bridge090=$merged/merged_pilot_analysis.json" \
  --out "$design" \
  --candidate-steps \
    2000 3000 4000 6000 9000 12000 18000 24000 36000 48000; then
  touch "$root/xwm_bridge090_formal_design.done"
else
  touch "$root/xwm_bridge090_formal_design.needs_refinement"
fi

find "$endpoints" "$pilot" "$merged" -type f \
  \( -name INPUT -o -name STRU -o -name KPT -o -name metadata.json \
     -o -name manifest.json -o -name endpoint_manifest.json \
     -o -name endpoint_validation.json -o -name phase_analysis.json \
     -o -name ti_analysis.json -o -name merged_pilot_analysis.json \) \
  -print0 | sort -z | xargs -0 sha256sum \
  >"$root/xwm_bridge090_abacus_pilot_SHA256SUMS"
touch "$done_file"
