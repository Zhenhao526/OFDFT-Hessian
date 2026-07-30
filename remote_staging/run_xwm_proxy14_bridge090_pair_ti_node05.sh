#!/usr/bin/env bash
set -euo pipefail

workspace=${WORKSPACE:-/home/shenwei01/WT_Al_melting_workspace_20260724}
repository=${REPO:-$workspace/repository}
root=$workspace/runs/xwm_lkt_20260727/xwm/free_energy_T0900
reference=$root/liquid_suf_pair_proxy_v1_sigma1p40_hardwall.json
target=$root/liquid_variance_bridge_alpha_scan_v1/alpha_0p900/model.json
restart=$root/liquid_suf_pair_proxy_v1_sigma1p40_validation_steps20000/run/checkpoint.json
output=$root/liquid_proxy14_to_bridge090_ti_lambda9_steps6000_v1
runner=$repository/remote_staging/run_kedf_proxy_bridge_pair_ti_node_local.sh
steps=${STEPS:-6000}

[[ -x $runner ]]
[[ -f $root/liquid_variance_bridge_alpha_scan_v1/pipeline.done ]]
[[ -f $root/liquid_variance_bridge_alpha_scan_v1/bridge_alpha_scan_summary.json ]]

env PYTHONPATH="$repository" python3 - \
  "$root/liquid_variance_bridge_alpha_scan_v1/bridge_alpha_scan_summary.json" \
  "$target" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
target = Path(sys.argv[2])
recommended = summary["recommended_bridge"]
assert summary["status"] == "verified"
assert recommended["label"] == "alpha_0p900"
assert recommended["alpha_correction"] == 0.9
assert recommended["eligible"] is True
assert recommended["minimum_distance_angstrom"] > 2.0
assert recommended["checks"]["liquid_verified"] is True
assert recommended["model_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
PY

bash "$runner" \
  xwm \
  "$repository" \
  "$reference" \
  "$target" \
  "$restart" \
  "$output" \
  xwm_liquid_proxy14 \
  "$steps"
