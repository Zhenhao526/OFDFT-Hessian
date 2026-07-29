#!/usr/bin/env bash
set -euo pipefail

workspace=/home/shenwei01/WT_Al_melting_workspace_20260724
repository=$workspace/repository
root=$workspace/runs/xwm_lkt_20260727/xwm/free_energy_T0900
parent=$root/ti_pilot_phase_specific_proxy14_lambda9_steps300_v1
batch=$root/ti_pilot_tempfix_phase_specific_proxy14_tau5_v1
solid_run=$batch/solid_fix
liquid_run=$batch/liquid_fix
solid_pair=$root/pair_reference_v1.dat
liquid_pair=$root/liquid_suf_pair_proxy_v1_sigma1p40_hardwall.dat
runner=$repository/remote_staging/run_kedf_ti_selected_extensions_node_local.sh

[[ ! -e "$batch" ]]
[[ -f "$parent/solid/manifest.json" ]]
[[ -f "$parent/liquid/manifest.json" ]]
[[ -f "$solid_pair" ]]
[[ -f "$liquid_pair" ]]
[[ -x "$runner" ]]

cd "$repository"
env PYTHONPATH=. python3 scripts/prepare_ti_window_extensions.py \
  --phase-root "solid=$parent/solid" \
  --out "$solid_run" \
  --steps 300 \
  --csvr-tau 5 \
  --seed 2026072990 \
  --ranks 18 \
  --phases solid \
  --labels lambda_0p000 lambda_0p125 lambda_0p750 lambda_1p000 \
  --config config/abacus_xwm_ti_cpu12.json
env PYTHONPATH=. python3 scripts/prepare_ti_window_extensions.py \
  --phase-root "liquid=$parent/liquid" \
  --out "$liquid_run" \
  --steps 300 \
  --csvr-tau 5 \
  --seed 2026073000 \
  --ranks 24 \
  --phases liquid \
  --labels lambda_0p250 lambda_0p750 lambda_0p875 \
  --config config/abacus_xwm_ti_cpu12.json

python3 - "$solid_run/solid" "$liquid_run/liquid" <<'PY'
import json
import sys
from pathlib import Path

expected = {
    "solid": {
        "lambda_0p000",
        "lambda_0p125",
        "lambda_0p750",
        "lambda_1p000",
    },
    "liquid": {"lambda_0p250", "lambda_0p750", "lambda_0p875"},
}
expected_ranks = {"solid": 18, "liquid": 24}
for argument in sys.argv[1:]:
    root = Path(argument)
    phase = root.name
    manifest = json.loads((root / "manifest.json").read_text())
    labels = {window["label"] for window in manifest["windows"]}
    assert labels == expected[phase], (phase, labels)
    assert manifest["target_kedf"] == "xwm"
    assert manifest["steps"] == 300
    assert manifest["mpi_ranks"] == expected_ranks[phase]
    for label in labels:
        metadata = json.loads((root / label / "metadata.json").read_text())
        assert metadata["source_step"] == 295
        assert metadata["source_velocities_discarded"] is False
        assert metadata["csvr_tau"] == 5.0
        assert not list((root / label).glob("OUT.*"))
PY

find "$batch" -type f \
  \( -name INPUT -o -name STRU -o -name KPT -o -name metadata.json \
     -o -name manifest.json \) \
  -print0 | sort -z | xargs -0 sha256sum >"$batch/PRELAUNCH_SHA256SUMS"

KEDF_TI_CPU_RANGES="0-17 19-36 38-55 57-74" \
  bash "$runner" "$solid_run" "$repository" "$solid_pair" "$solid_run/solid"
KEDF_TI_CPU_RANGES="0-23 25-48 50-73" \
  bash "$runner" "$liquid_run" "$repository" "$liquid_pair" "$liquid_run/liquid"

touch "$batch/extension_pipeline.completed"
