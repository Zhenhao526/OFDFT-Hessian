#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 PREFILTER_ROOT REPOSITORY" >&2
  exit 2
fi

root=$1
repository=$2
manifest=$root/prefilter_manifest.json
[[ -f "$manifest" ]]
[[ -f "$root/INPUT_SHA256SUMS" ]]
(cd "$root" && sha256sum -c INPUT_SHA256SUMS)

mapfile -t points < <(
  python3 - "$manifest" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
if report.get("status") != "prepared":
    raise SystemExit("pressure prefilter is not prepared")
points = report.get("points", [])
if not points:
    raise SystemExit("pressure prefilter contains no points")
for point in points:
    print(point["run"])
PY
)
expected_jobs=$((${#points[@]} * 3))

mark_failed() {
  touch "$root/prefilter.failed"
}
trap mark_failed ERR

EXPECTED_JOBS="$expected_jobs" \
GPU_IDS="${GPU_IDS:-0,1,2,3}" CPU_IDS="${CPU_IDS:-36,37,74,75}" \
  bash "$repository/remote_staging/run_kedf_gpu_scf_batch_node_local.sh" \
  "$root"

cd "$repository"
for point in "${points[@]}"; do
  env PYTHONPATH=. python3 scripts/check_kedf_pressure_finite_difference.py \
    analyze "$point" >"$point/analysis.stdout"
done

python3 - "$manifest" "$root/prefilter_summary.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
manifest = json.loads(manifest_path.read_text())
results = []
for point in manifest["points"]:
    result_path = Path(point["run"]) / "pressure_fd_result.json"
    result = json.loads(result_path.read_text())
    results.append(
        {
            **point,
            "result": str(result_path.resolve()),
            "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            "finite_difference_static_pressure_kbar": result[
                "finite_difference_static_pressure_kbar"
            ],
            "ideal_ionic_pressure_kbar": result[
                "ideal_ionic_pressure_kbar"
            ],
            "estimated_total_pressure_kbar": result[
                "estimated_total_pressure_kbar"
            ],
            "analytic_static_pressure_kbar": result[
                "analytic_static_pressure_kbar"
            ],
        }
    )
summary = {
    **manifest,
    "schema": "kedf-pressure-prefilter-summary-v1",
    "status": "diagnostic_completed_requires_md_confirmation",
    "points": results,
}
output.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

find "$root" -type f \
  \( -name INPUT -o -name STRU -o -name KPT -o -name metadata.json \
     -o -name manifest.json -o -name pressure_fd_result.json \
     -o -name prefilter_manifest.json -o -name prefilter_summary.json \) \
  -print0 | sort -z | xargs -0 sha256sum >"$root/SHA256SUMS"
touch "$root/prefilter.done"
trap - ERR
