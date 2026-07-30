#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "usage: $0 SOURCE_ROOT OUT REPOSITORY [STEPS] [CONFIG]" >&2
  exit 2
fi

source_root=$1
out=$2
repository=$3
steps=${4:-3000}
config=${5:-$repository/config/abacus_xwm_cpu36.json}
python=${PYTHON:-python3}
csvr_tau=${CSVR_TAU_FS:-5}
seed=${MD_SEED:-2026073100}
done_file=$out/enthalpy_extension.done
failed_file=$out/enthalpy_extension.failed

mark_failed() {
  local rc=$?
  trap - ERR
  if [[ -d $out ]]; then
    printf '%s exit_code=%s\n' "$(date -Iseconds)" "$rc" >"$failed_file"
  fi
  exit "$rc"
}
trap mark_failed ERR

[[ -f $source_root/confirmation_manifest.json ]]
[[ -f $source_root/confirmation_summary.json ]]
[[ -f $config ]]
[[ ! -e $out ]]
cd "$repository"

env PYTHONPATH=. "$python" scripts/prepare_kedf_enthalpy_extension.py \
  --source "$source_root" \
  --out "$out" \
  --config "$config" \
  --steps "$steps" \
  --csvr-tau "$csvr_tau" \
  --seed "$seed" \
  --ranks 36

env PYTHONPATH=. "$python" - \
  "$out/confirmation_manifest.json" "$source_root" "$steps" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1]).resolve()
source_root = Path(sys.argv[2]).resolve()
steps = int(sys.argv[3])
manifest = json.loads(manifest_path.read_text())
assert manifest["schema"] == "kedf-volume-confirmation-v1"
assert manifest["extension_schema"] == "kedf-enthalpy-extension-v1"
assert manifest["target_kedf"] == "xwm"
assert int(manifest["steps"]) == steps
assert manifest["parent_confirmation"] == str(source_root)
assert manifest["source_velocities_discarded"] is False
assert all(
    row["source_velocities_discarded"] is False
    and int(row["source_step"]) >= 0
    for row in manifest["phases"]
)
sources = {}
for name in ("confirmation_manifest.json", "confirmation_summary.json"):
    path = source_root / name
    sources[name] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
provenance = {
    "schema": "xwm-enthalpy-extension-provenance-v1",
    "target_kedf": "xwm",
    "steps": steps,
    "parent_confirmation": str(source_root),
    "source_velocities_discarded": False,
    "sources": sources,
}
(manifest_path.parent / "extension_provenance.json").write_text(
    json.dumps(provenance, indent=2, sort_keys=True) + "\n"
)
PY

find "$out" -type f \
  \( -name INPUT -o -name STRU -o -name KPT -o -name metadata.json \
     -o -name confirmation_manifest.json \
     -o -name extension_provenance.json \) \
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
