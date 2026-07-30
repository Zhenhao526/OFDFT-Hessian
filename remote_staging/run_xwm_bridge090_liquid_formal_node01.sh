#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 STEPS_PER_WINDOW" >&2
  exit 2
fi

steps=$1
workspace=${WORKSPACE:-/home/shenwei01/WT_Al_melting_workspace_20260724}
repository=${REPO:-$workspace/repository}
run_root=${RUN_ROOT:-$workspace/runs/xwm_lkt_20260727}
root=$run_root/xwm/free_energy_T0900
merged=$root/ti_pilot_merged_phase_specific_bridge090_v1/merged_pilot_analysis.json
design=$root/ti_formal_design_phase_specific_bridge090_v1.json
solid_model=$root/pair_reference_v1.json
bridge_model=$root/liquid_variance_bridge_alpha_scan_v1/alpha_0p900/model.json
bridge_dat=$root/liquid_variance_bridge_alpha090_v1.dat
old_solid=$root/ti_formal_lambda9_steps3000_v1/solid
formal=$root/ti_formal_liquid_bridge090_steps${steps}_v1
python=${PYTHON:-$workspace/.venv-reference/bin/python}
log=$formal/liquid_formal_pipeline.log
done_file=$formal/liquid_formal_pipeline.done
failed_file=$formal/liquid_formal_pipeline.failed

mark_failed() {
  local rc=$?
  trap - ERR
  if [[ -d $formal ]]; then
    printf '%s exit_code=%s\n' "$(date -Iseconds)" "$rc" >"$failed_file"
  fi
  exit "$rc"
}
trap mark_failed ERR

[[ $steps =~ ^[0-9]+$ ]]
((steps >= 1000))
[[ -f $merged ]]
[[ -f $design ]]
[[ -f $solid_model ]]
[[ -f $bridge_model ]]
[[ -f $bridge_dat ]]
[[ -f $old_solid/discard_convergence_summary.json ]]
[[ ! -e $formal ]]
if pgrep -x abacus_pw_para >/dev/null; then
  echo "refusing to share node01 CPUs with another ABACUS job" >&2
  exit 2
fi

cd "$repository"
env PYTHONPATH=. "$python" - \
  "$merged" "$design" "$old_solid/discard_convergence_summary.json" \
  "$solid_model" "$bridge_model" "$steps" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

merged = json.loads(Path(sys.argv[1]).read_text())
design = json.loads(Path(sys.argv[2]).read_text())
solid = json.loads(Path(sys.argv[3]).read_text())
solid_model = Path(sys.argv[4]).resolve()
bridge_model = Path(sys.argv[5]).resolve()
steps = int(sys.argv[6])
assert merged["status"] == "pilot_grid_verified"
assert merged["phase_results"]["solid"]["status"] == "verified"
assert merged["phase_results"]["liquid"]["status"] == "verified"
assert solid["status"] == "verified"
assert design["status"] == "production_design_verified"
selected = design["methods"]["xwm_bridge090"]["selected_steps_per_window"]
assert int(selected) == steps
expected = {
    "solid": hashlib.sha256(solid_model.read_bytes()).hexdigest(),
    "liquid": hashlib.sha256(bridge_model.read_bytes()).hexdigest(),
}
for phase in ("solid", "liquid"):
    rows = merged["phase_results"][phase]["window_results"]
    assert len(rows) == 9
    for row in rows:
        metadata = json.loads((Path(row["run"]) / "metadata.json").read_text())
        assert metadata["pair_model_sha256"] == expected[phase]
PY

env PYTHONPATH=. "$python" scripts/prepare_kedf_ti_formal_from_merged.py \
  --merged "$merged" \
  --out "$formal" \
  --steps "$steps" \
  --csvr-tau 5 \
  --ranks 12 \
  --seed 2026073200 \
  --config config/abacus_xwm_ti_cpu12.json

env PYTHONPATH=. "$python" scripts/verify_kedf_ti_formal_preflight.py \
  "$formal" \
  --target-kedf xwm \
  --steps "$steps" \
  --ranks 12 \
  --source-step 295 \
  --minimum-nn 2.0 \
  --csvr-tau 5

bash remote_staging/run_kedf_ti_selected_extensions_node_local.sh \
  "$formal" "$repository" "$bridge_dat" "$formal/liquid"

reports=()
for suffix in d25 d50 d75; do
  case $suffix in
    d25) discard=0.25 ;;
    d50) discard=0.50 ;;
    d75) discard=0.75 ;;
  esac
  report=$formal/liquid/ti_prod_${suffix}.json
  env PYTHONPATH=. "$python" scripts/analyze_wt_pair_ti_production.py \
    "$formal/liquid" \
    --discard-fraction "$discard" \
    --temperature-tolerance 20 \
    --max-block-se 1 \
    --max-half-drift 2 \
    --max-quadrature-difference 2 \
    --minimum-overlap-ess 0.05 \
    --max-overlap-closure 2 \
    --out "$report" >>"$log" 2>&1
  reports+=("$report")
done
env PYTHONPATH=. "$python" scripts/summarize_wt_ti_convergence.py \
  "${reports[@]}" \
  --max-integral-discard-spread 1 \
  --max-window-discard-spread 1 \
  --max-window-block-se 1 \
  --max-window-half-drift 2 \
  --out "$formal/liquid/discard_convergence_summary.json" \
  >>"$log" 2>&1

status=$(
  "$python" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$formal/liquid/discard_convergence_summary.json"
)
if [[ $status != verified ]]; then
  touch "$formal/liquid_formal_pipeline.needs_extension"
  exit 1
fi

find "$formal" -type f \
  \( -name INPUT -o -name STRU -o -name KPT -o -name metadata.json \
     -o -name manifest.json -o -name formal_manifest.json \
     -o -name preflight.json -o -name phase_analysis.json \
     -o -name 'ti_prod_*.json' \
     -o -name discard_convergence_summary.json \) \
  -print0 | sort -z | xargs -0 sha256sum >"$formal/SHA256SUMS"
touch "$done_file"
