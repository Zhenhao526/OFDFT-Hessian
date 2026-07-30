#!/usr/bin/env bash
set -euo pipefail

workspace=/home/shenwei01/WT_Al_melting_workspace_20260724
repository=$workspace/repository
run_root_all=$workspace/runs/xwm_lkt_20260727
root=$run_root_all/lkt/free_energy_T0900
source_run=$run_root_all/lkt/volume_confirmation_T0900_steps600_r2_tau1_node01/liquid
old_merged=$root/ti_pilot_merged_phase_specific_proxy6_verified_v3/merged_pilot_analysis.json
original=$root/ti_pilot_phase_specific_bridge060_lambda9_steps300_v1/liquid
first_fix=$root/ti_pilot_tempfix_phase_specific_bridge060_tau1_v1/liquid_fix/liquid
batch=$root/ti_pilot_bridge060_lambda1_from_zero_pressure_steps1200_tau1_v5
prepared=$batch/liquid_fix
bridge_json=$root/liquid_variance_bridge_alpha060_v1.json
bridge_dat=$root/liquid_variance_bridge_alpha060_v1.dat
merged=$root/ti_pilot_merged_phase_specific_bridge060_verified_v4_zeropressure
design=$root/ti_formal_design_phase_specific_bridge060_v4_zeropressure.json
runner=$repository/remote_staging/run_kedf_ti_selected_extensions_node_local.sh
python=$workspace/.venv-reference/bin/python
bridge_sha=1fec6fe7a6a0d0104a0d76a3266419da83baa7b4a002c40b02fa7844ecc8ea6f
source_md_sha=1c86e048e7ec363cdc50a3c5197d397fc4c569c11d1d7f6481928a8011e20acd
label=lambda_1p000
marker=lkt_bridge060_pilot_lambda1_from_zero_pressure
log=$root/${marker}.log

exec > >(tee -a "$log") 2>&1

mark_failed() {
  local rc=$?
  trap - ERR
  printf '%s failed exit_code=%s\n' "$(date -Iseconds)" "$rc" \
    >"$root/${marker}.failed"
  exit "$rc"
}
trap mark_failed ERR

[[ -x $python ]]
[[ -x $runner ]]
[[ -f $source_run/metadata.json ]]
[[ -f $source_run/phase_analysis.json ]]
[[ -f $old_merged ]]
[[ -f $original/manifest.json ]]
[[ -f $first_fix/manifest.json ]]
[[ -f $bridge_json ]]
[[ -f $bridge_dat ]]
[[ ! -e $batch ]]
[[ ! -e $merged ]]
[[ ! -e $design ]]
[[ ! -e $root/${marker}.done ]]
if pgrep -f '[a]bacus_pw' >/dev/null; then
  echo "refusing to share node01 with another ABACUS job" >&2
  exit 2
fi
printf '%s  %s\n' "$bridge_sha" "$bridge_json" | sha256sum -c -
source_md=$(find "$source_run" -path '*/OUT.*/MD_dump' -type f)
[[ $(printf '%s\n' "$source_md" | wc -l) -eq 1 ]]
printf '%s  %s\n' "$source_md_sha" "$source_md" | sha256sum -c -

cd "$repository"
env PYTHONPATH=. "$python" - \
  "$source_run/phase_analysis.json" "$source_run/metadata.json" <<'PY'
import json
import sys

phase = json.load(open(sys.argv[1]))
metadata = json.load(open(sys.argv[2]))
assert phase["status"] == "liquid_verified"
assert phase["trajectory"]["minimum_nearest_neighbor_A"] > 2.0
assert phase["trajectory"]["diffusion_MSD_slope_A2_per_step"] > 0.001
assert metadata["target_kedf"] == "lkt"
assert metadata["phase"] == "liquid"
assert abs(float(metadata["volume_per_atom_A3"]) - 18.735) <= 1.0e-8
PY

env PYTHONPATH=. "$python" scripts/prepare_ti_window_extensions.py \
  --phase-root "liquid=$original" \
  --out "$prepared" \
  --steps 1200 \
  --csvr-tau 1 \
  --seed 2026073040 \
  --ranks 72 \
  --phases liquid \
  --labels "$label" \
  --pair-model-override "$bridge_json" \
  --source-run-override "liquid=$source_run" \
  --config config/abacus_lkt_ti_cpu12.json

env PYTHONPATH=. "$python" - \
  "$prepared/liquid/manifest.json" "$bridge_json" "$source_run" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
bridge = str(Path(sys.argv[2]).resolve())
source = str(Path(sys.argv[3]).resolve())
manifest = json.loads(manifest_path.read_text())
assert manifest["target_kedf"] == "lkt"
assert manifest["phase"] == "liquid"
assert manifest["steps"] == 1200
assert manifest["mpi_ranks"] == 72
assert manifest["pair_model"] == bridge
assert manifest["pair_model_override_at_lambda_one"] is True
assert manifest["source_run_override_at_lambda_one"] is True
assert manifest["source_run_override"] == source
assert [row["label"] for row in manifest["windows"]] == ["lambda_1p000"]
run = manifest_path.parent / "lambda_1p000"
metadata = json.loads((run / "metadata.json").read_text())
assert metadata["source_step"] == 595
assert metadata["source_velocities_discarded"] is False
assert metadata["csvr_tau"] == 1.0
assert metadata["lambda"] == 1.0
assert metadata["target_kedf"] == "lkt"
assert metadata["phase"] == "liquid"
assert metadata["pair_model"] == bridge
assert metadata["pair_model_override_at_lambda_one"] is True
assert metadata["source_run_override_at_lambda_one"] is True
assert not list(run.glob("OUT.*"))
PY

find "$batch" -type f \
  \( -name INPUT -o -name STRU -o -name KPT -o -name metadata.json \
     -o -name manifest.json \) \
  -print0 | sort -z | xargs -0 sha256sum >"$batch/PRELAUNCH_SHA256SUMS"

bash "$runner" "$prepared" "$repository" "$bridge_dat" "$prepared/liquid"

run=$prepared/liquid/$label
env PYTHONPATH=. "$python" scripts/analyze_phase_run.py \
  "$run" --expected liquid --thermalized-initial \
  --out "$run/phase_analysis.json" >/dev/null
env PYTHONPATH=. "$python" - \
  "$run/phase_analysis.json" "$prepared/liquid/ti_analysis.json" <<'PY'
import json
import sys

phase = json.load(open(sys.argv[1]))
assert phase["status"] == "liquid_verified", (
    phase["status"],
    phase.get("phase_gate"),
)
assert phase["trajectory"]["minimum_nearest_neighbor_A"] > 2.0
ti = json.load(open(sys.argv[2]))
row = next(row for row in ti["window_results"] if row["label"] == "lambda_1p000")
assert row["max_step"] == 1200
assert row["component_samples"] == 1200
assert abs(row["temperature_last_half_K"]["mean"] - 900.0) <= 20.0
PY

mapfile -t replacements < <(
  env PYTHONPATH=. "$python" - \
    "$old_merged" "$original" "$first_fix" "$prepared/liquid" <<'PY'
import json
import sys
from pathlib import Path

old = json.loads(Path(sys.argv[1]).read_text())
original = Path(sys.argv[2])
first_fix = Path(sys.argv[3])
zero_pressure = Path(sys.argv[4])
for row in old["phase_results"]["solid"]["window_results"]:
    print(f"solid:{row['label']}={row['run']}")
for row in json.loads((original / "manifest.json").read_text())["windows"]:
    label = row["label"]
    if label == "lambda_1p000":
        run = zero_pressure / label
    elif label in {"lambda_0p250", "lambda_0p875"}:
        run = first_fix / label
    else:
        run = original / label
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
  touch "$root/${marker}.verified"
else
  touch "$root/${marker}.needs_refinement"
  exit 3
fi

if env PYTHONPATH=. "$python" scripts/design_kedf_ti_production.py \
  --method "lkt_bridge060=$merged/merged_pilot_analysis.json" \
  --out "$design" \
  --candidate-steps 2000 3000 4000 6000 9000 12000 18000 24000; then
  touch "$root/lkt_bridge060_formal_design_v4_zeropressure.done"
else
  touch "$root/lkt_bridge060_formal_design_v4_zeropressure.needs_refinement"
  exit 4
fi

find "$source_run" "$original" "$first_fix" "$batch" "$merged" -type f \
  \( -name INPUT -o -name STRU -o -name KPT -o -name metadata.json \
     -o -name manifest.json -o -name phase_analysis.json \
     -o -name ti_analysis.json -o -name merged_pilot_analysis.json \) \
  -print0 | sort -z | xargs -0 sha256sum >"$root/${marker}_SHA256SUMS"
touch "$root/${marker}.done"
