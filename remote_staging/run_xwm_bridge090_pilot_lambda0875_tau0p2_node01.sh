#!/usr/bin/env bash
set -euo pipefail

workspace=${WORKSPACE:-/home/shenwei01/WT_Al_melting_workspace_20260724}
repository=${REPO:-$workspace/repository}
run_root=${RUN_ROOT:-$workspace/runs/xwm_lkt_20260727}
root=$run_root/xwm/free_energy_T0900
old_merged=$root/ti_pilot_merged_phase_specific_proxy14_v1/merged_pilot_analysis.json
original=$root/ti_pilot_phase_specific_bridge090_lambda9_steps300_v1/liquid
tau1=$root/ti_pilot_tempfix_phase_specific_bridge090_tau1_v2/liquid_fix/liquid
batch=$root/ti_pilot_tempfix_bridge090_lambda0875_steps600_tau0p2_v3
fix_root=$batch/liquid_fix
bridge_json=$root/liquid_variance_bridge_alpha_scan_v1/alpha_0p900/model.json
bridge_dat=$root/liquid_variance_bridge_alpha090_v1.dat
merged=$root/ti_pilot_merged_phase_specific_bridge090_verified_v4_tau0p2
design=$root/ti_formal_design_phase_specific_bridge090_v4_tau0p2.json
runner=$repository/remote_staging/run_kedf_ti_selected_extensions_node_local.sh
python=${PYTHON:-$workspace/.venv-reference/bin/python}
bridge_sha=b207a394c9ba0ee652596156425f7ed4b93b4e75d3f4b8dcdcbd6871627cd89a
label=lambda_0p875
log=$root/xwm_bridge090_pilot_lambda0875_tau0p2.log
done_file=$root/xwm_bridge090_pilot_lambda0875_tau0p2.done
failed_file=$root/xwm_bridge090_pilot_lambda0875_tau0p2.failed

exec > >(tee -a "$log") 2>&1

mark_failed() {
  local rc=$?
  trap - ERR
  printf '%s failed exit_code=%s\n' "$(date -Iseconds)" "$rc" >"$failed_file"
  exit "$rc"
}
trap mark_failed ERR

[[ -x $python ]]
[[ -x $runner ]]
[[ -f $old_merged ]]
[[ -f $original/manifest.json ]]
[[ -f $tau1/manifest.json ]]
[[ -f $bridge_json ]]
[[ -f $bridge_dat ]]
[[ ! -e $batch ]]
[[ ! -e $merged ]]
[[ ! -e $design ]]
[[ ! -e $done_file ]]
[[ ! -e $failed_file ]]
if pgrep -x abacus_pw_para >/dev/null; then
  echo "refusing to share node01 CPUs with another ABACUS CPU job" >&2
  exit 2
fi
printf '%s  %s\n' "$bridge_sha" "$bridge_json" | sha256sum -c -

cd "$repository"
source_run=$tau1/$label
env PYTHONPATH=. "$python" scripts/analyze_phase_run.py \
  "$source_run" --expected liquid --thermalized-initial \
  --out "$source_run/phase_analysis.json" >/dev/null
env PYTHONPATH=. "$python" - "$source_run/phase_analysis.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
assert report["status"] == "liquid_verified"
assert report["trajectory"]["minimum_nearest_neighbor_A"] > 2.0
PY

env PYTHONPATH=. "$python" scripts/prepare_ti_window_extensions.py \
  --phase-root "liquid=$tau1" \
  --out "$fix_root" \
  --steps 600 \
  --csvr-tau 0.2 \
  --seed 2026073003 \
  --ranks 36 \
  --phases liquid \
  --labels "$label" \
  --config config/abacus_xwm_ti_cpu12.json

env PYTHONPATH=. "$python" - \
  "$fix_root/liquid/manifest.json" "$bridge_json" "$label" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
bridge = Path(sys.argv[2]).resolve()
label = sys.argv[3]
manifest = json.loads(manifest_path.read_text())
assert manifest["target_kedf"] == "xwm"
assert manifest["phase"] == "liquid"
assert manifest["steps"] == 600
assert manifest["mpi_ranks"] == 36
assert [row["label"] for row in manifest["windows"]] == [label]
run = manifest_path.parent / label
metadata = json.loads((run / "metadata.json").read_text())
assert metadata["source_step"] == 295
assert metadata["source_velocities_discarded"] is False
assert metadata["csvr_tau"] == 0.2
assert metadata["target_kedf"] == "xwm"
assert metadata["phase"] == "liquid"
pair_model = Path(metadata["pair_model"]).resolve()
assert pair_model == bridge
assert hashlib.sha256(pair_model.read_bytes()).hexdigest() == hashlib.sha256(
    bridge.read_bytes()
).hexdigest()
assert not list(run.glob("OUT.*"))
PY

find "$batch" -type f \
  \( -name INPUT -o -name STRU -o -name KPT -o -name metadata.json \
     -o -name manifest.json \) \
  -print0 | sort -z | xargs -0 sha256sum >"$batch/PRELAUNCH_SHA256SUMS"

KEDF_TI_CPU_RANGES="38-73" \
  bash "$runner" "$fix_root" "$repository" "$bridge_dat" "$fix_root/liquid"

run=$fix_root/liquid/$label
env PYTHONPATH=. "$python" scripts/analyze_phase_run.py \
  "$run" --expected liquid --thermalized-initial \
  --out "$run/phase_analysis.json" >/dev/null
env PYTHONPATH=. "$python" - "$run/phase_analysis.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
assert report["status"] == "liquid_verified"
assert report["trajectory"]["minimum_nearest_neighbor_A"] > 2.0
PY

mapfile -t replacements < <(
  env PYTHONPATH=. "$python" - \
    "$old_merged" "$original" "$tau1" "$fix_root/liquid" <<'PY'
import json
import sys
from pathlib import Path

old = json.loads(Path(sys.argv[1]).read_text())
original = Path(sys.argv[2])
tau1 = Path(sys.argv[3])
correction = Path(sys.argv[4])
for row in old["phase_results"]["solid"]["window_results"]:
    print(f"solid:{row['label']}={row['run']}")
for row in json.loads((original / "manifest.json").read_text())["windows"]:
    label = row["label"]
    run = original / label
    if label == "lambda_0p250":
        run = tau1 / label
    elif label == "lambda_0p875":
        run = correction / label
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
grep -q '"status": "pilot_grid_verified"' \
  "$merged/merged_pilot_analysis.json"

env PYTHONPATH=. "$python" scripts/design_kedf_ti_production.py \
  --method "xwm_bridge090=$merged/merged_pilot_analysis.json" \
  --out "$design" \
  --candidate-steps \
    2000 3000 4000 6000 9000 12000 18000 24000 36000 48000

find "$original" "$tau1" "$batch" "$merged" -type f \
  \( -name INPUT -o -name STRU -o -name KPT -o -name metadata.json \
     -o -name manifest.json -o -name phase_analysis.json \
     -o -name ti_analysis.json -o -name merged_pilot_analysis.json \) \
  -print0 | sort -z | xargs -0 sha256sum \
  >"$root/xwm_bridge090_pilot_lambda0875_tau0p2_SHA256SUMS"
touch "$done_file"
