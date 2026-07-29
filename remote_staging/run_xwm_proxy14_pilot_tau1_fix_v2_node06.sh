#!/usr/bin/env bash
set -euo pipefail

workspace=/home/shenwei01/WT_Al_melting_workspace_20260724
repository=$workspace/repository
root=$workspace/runs/xwm_lkt_20260727/xwm/free_energy_T0900
parent=$root/ti_pilot_tempfix_phase_specific_proxy14_tau5_v1
batch=$root/ti_pilot_tempfix_phase_specific_proxy14_tau1_v2
solid_run=$batch/solid_fix
liquid_run=$batch/liquid_fix
solid_pair=$root/pair_reference_v1.dat
liquid_pair=$root/liquid_suf_pair_proxy_v1_sigma1p40_hardwall.dat
runner=$repository/remote_staging/run_kedf_ti_selected_extensions_node_local.sh

[[ ! -e "$batch" ]]
[[ -f "$parent/solid_fix/solid/manifest.json" ]]
[[ -f "$parent/liquid_fix/liquid/manifest.json" ]]
[[ -f "$solid_pair" ]]
[[ -f "$liquid_pair" ]]
[[ -x "$runner" ]]

cd "$repository"
env PYTHONPATH=. python3 scripts/prepare_ti_window_extensions.py \
  --phase-root "solid=$parent/solid_fix/solid" \
  --out "$solid_run" \
  --steps 300 \
  --csvr-tau 1 \
  --seed 2026073010 \
  --ranks 36 \
  --phases solid \
  --labels lambda_0p125 \
  --config config/abacus_xwm_ti_cpu12.json
env PYTHONPATH=. python3 scripts/prepare_ti_window_extensions.py \
  --phase-root "liquid=$parent/liquid_fix/liquid" \
  --out "$liquid_run" \
  --steps 300 \
  --csvr-tau 1 \
  --seed 2026073020 \
  --ranks 36 \
  --phases liquid \
  --labels lambda_0p250 \
  --config config/abacus_xwm_ti_cpu12.json

python3 - "$solid_run/solid" "$liquid_run/liquid" <<'PY'
import json
import sys
from pathlib import Path

expected = {"solid": "lambda_0p125", "liquid": "lambda_0p250"}
for argument in sys.argv[1:]:
    root = Path(argument)
    phase = root.name
    manifest = json.loads((root / "manifest.json").read_text())
    assert [window["label"] for window in manifest["windows"]] == [expected[phase]]
    assert manifest["target_kedf"] == "xwm"
    assert manifest["steps"] == 300
    assert manifest["mpi_ranks"] == 36
    metadata = json.loads(
        (root / expected[phase] / "metadata.json").read_text()
    )
    assert metadata["source_step"] == 295
    assert metadata["source_velocities_discarded"] is False
    assert metadata["csvr_tau"] == 1.0
    assert not list((root / expected[phase]).glob("OUT.*"))
PY

find "$batch" -type f \
  \( -name INPUT -o -name STRU -o -name KPT -o -name metadata.json \
     -o -name manifest.json \) \
  -print0 | sort -z | xargs -0 sha256sum >"$batch/PRELAUNCH_SHA256SUMS"

KEDF_TI_CPU_RANGES="0-35" \
  bash "$runner" "$solid_run" "$repository" "$solid_pair" "$solid_run/solid" &
solid_pid=$!
KEDF_TI_CPU_RANGES="38-73" \
  bash "$runner" "$liquid_run" "$repository" "$liquid_pair" "$liquid_run/liquid" &
liquid_pid=$!

status=0
wait "$solid_pid" || status=$?
wait "$liquid_pid" || status=$?
if [[ $status -ne 0 ]]; then
  touch "$batch/extension_pipeline.failed"
  exit "$status"
fi

touch "$batch/extension_pipeline.completed"
