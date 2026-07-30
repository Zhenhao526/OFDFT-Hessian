#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
  echo "usage: $0 TEMPERATURE_K ZERO_PRESSURE_JSON OUT REPOSITORY [STEPS] [CONFIG]" >&2
  exit 2
fi

temperature=$1
zero_pressure=$2
out=$3
repository=$4
steps=${5:-3000}
config=${6:-$repository/config/abacus_xwm_cpu36.json}
python=${PYTHON:-python3}
csvr_tau=${CSVR_TAU_FS:-5}
seed=${MD_SEED:-2026073000}
done_file=$out/enthalpy_pipeline.done
failed_file=$out/enthalpy_pipeline.failed

mark_failed() {
  local rc=$?
  trap - ERR
  if [[ -d $out ]]; then
    printf '%s exit_code=%s\n' "$(date -Iseconds)" "$rc" >"$failed_file"
  fi
  exit "$rc"
}
trap mark_failed ERR

[[ -f $zero_pressure ]]
[[ -f $config ]]
[[ ! -e $out ]]
cd "$repository"

mapfile -t phase_fields < <(
  env PYTHONPATH=. "$python" - "$zero_pressure" "$temperature" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
temperature = float(sys.argv[2])
document = json.loads(path.read_text())
assert document["status"] == "all_confirmations_passed"
assert document["target_kedf"] == "xwm"
assert math.isclose(float(document["temperature_K"]), temperature, abs_tol=1e-9)
phases = {row["phase"]: row for row in document["phase_results"]}
assert set(phases) == {"solid", "liquid"}
for phase in ("solid", "liquid"):
    row = phases[phase]
    assert row["status"] == "confirmation_passed"
    assert row["phase_status"] == f"{phase}_verified"
    assert float(row["nearest_neighbor_A"]) > 2.0
    assert abs(float(row["pressure_kbar"]["mean"])) <= 2.5
    assert Path(row["run"]).is_dir()
    print(f"{phase}\t{Path(row['run']).resolve()}\t{row['volume_per_atom_A3']}")
PY
)
[[ ${#phase_fields[@]} -eq 2 ]]

IFS=$'\t' read -r solid_phase solid_source solid_volume <<<"${phase_fields[0]}"
IFS=$'\t' read -r liquid_phase liquid_source liquid_volume <<<"${phase_fields[1]}"
[[ $solid_phase == solid ]]
[[ $liquid_phase == liquid ]]

env PYTHONPATH=. "$python" scripts/prepare_kedf_volume_confirmation.py prepare \
  --out "$out" \
  --solid-source "$solid_source" \
  --liquid-source "$liquid_source" \
  --solid-volume "$solid_volume" \
  --liquid-volume "$liquid_volume" \
  --config "$config" \
  --temperature "$temperature" \
  --steps "$steps" \
  --csvr-tau "$csvr_tau" \
  --seed "$seed" \
  --ranks 36

env PYTHONPATH=. "$python" - \
  "$out/confirmation_manifest.json" "$zero_pressure" "$steps" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
zero_path = Path(sys.argv[2]).resolve()
steps = int(sys.argv[3])
manifest = json.loads(manifest_path.read_text())
zero = json.loads(zero_path.read_text())
assert manifest["target_kedf"] == "xwm"
assert int(manifest["steps"]) == steps
assert manifest["stress_available"] is False
zero_phases = {row["phase"]: row for row in zero["phase_results"]}
for row in manifest["phases"]:
    phase = row["phase"]
    assert phase in zero_phases
    assert math.isclose(
        float(row["volume_per_atom_A3"]),
        float(zero_phases[phase]["volume_per_atom_A3"]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
provenance = {
    "schema": "xwm-enthalpy-input-provenance-v1",
    "target_kedf": "xwm",
    "temperature_K": manifest["temperature_K"],
    "steps": steps,
    "zero_pressure_summary": {
        "path": str(zero_path),
        "sha256": hashlib.sha256(zero_path.read_bytes()).hexdigest(),
    },
}
(manifest_path.parent / "zero_pressure_provenance.json").write_text(
    json.dumps(provenance, indent=2, sort_keys=True) + "\n"
)
PY

find "$out" -type f \
  \( -name INPUT -o -name STRU -o -name KPT -o -name metadata.json \
     -o -name confirmation_manifest.json \
     -o -name zero_pressure_provenance.json \) \
  -print0 | sort -z | xargs -0 sha256sum >"$out/INPUT_SHA256SUMS"

CPU_RANGES=${CPU_RANGES:-0-35,38-73} \
  bash remote_staging/run_kedf_enthalpy_pair_node_local.sh \
  "$out" "$repository"

status=$(
  "$python" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$out/confirmation_summary.json"
)
[[ $status == volume_confirmation_verified ]]
touch "$done_file"
