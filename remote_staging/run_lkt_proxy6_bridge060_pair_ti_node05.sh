#!/usr/bin/env bash
set -euo pipefail

workspace=/home/shenwei01/WT_Al_melting_workspace_20260724
repository=$workspace/repository
root=$workspace/runs/xwm_lkt_20260727/lkt/free_energy_T0900
torch_runner=$repository/remote_staging/run_node05_cached_cuda_torch.sh
python=$workspace/.venv-reference-cuda/bin/python
reference=$root/liquid_suf_pair_proxy_v6_hardwall.json
target=$root/liquid_variance_bridge_alpha_scan_v5/alpha_0p600/model.json
restart=$root/liquid_suf_pair_proxy_v6_validation_steps20000/run/checkpoint.json
output=$root/liquid_proxy6_to_bridge060_ti_lambda9_steps6000_v1
reference_sha=6f06fc2f9fbcaacc8cda12048805e2107aa5eee1123c997b7ce9c1c96da4a010
target_sha=1fec6fe7a6a0d0104a0d76a3266419da83baa7b4a002c40b02fa7844ecc8ea6f
steps=${STEPS:-6000}
temperature=900
gamma_per_fs=0.2
lambdas=(0.000 0.125 0.250 0.375 0.500 0.625 0.750 0.875 1.000)
cpus=(36 37 74 75 76 77)
gpus=(0 1 2 3 4 5)

[[ -x $torch_runner ]]
[[ -x $python ]]
[[ -f $reference ]]
[[ -f $target ]]
[[ -f $restart ]]
[[ ! -e $output ]]
printf '%s  %s\n' "$reference_sha" "$reference" | sha256sum -c -
printf '%s  %s\n' "$target_sha" "$target" | sha256sum -c -

cd "$repository"
env PYTHONPATH=. "$torch_runner" - "$reference" "$target" "$restart" <<'PY'
import json
import sys
from pathlib import Path

from scripts.run_pair_pair_ti_md import load_pair_document

for path in map(Path, sys.argv[1:3]):
    load_pair_document(path, target_kedf="lkt", phase="liquid")
restart = json.loads(Path(sys.argv[3]).read_text())
assert restart["schema"] == "mpn-pair-reference-md-checkpoint-v1"
assert restart["positions_angstrom"]
assert restart["velocities_angstrom_per_fs"]
assert restart["lattice_angstrom"]
PY

mkdir -p "$output"
exec > >(tee -a "$output/pipeline.log") 2>&1

mark_failed() {
  local rc=$?
  trap - ERR
  printf '%s failed exit_code=%s\n' "$(date -Iseconds)" "$rc" \
    | tee "$output/pair_pair_ti.failed"
  exit "$rc"
}
trap mark_failed ERR

sha256sum "$reference" "$target" "$restart" \
  >"$output/SOURCE_SHA256"

for ((first = 0; first < ${#lambdas[@]}; first += ${#cpus[@]})); do
  pids=()
  labels=()
  for ((slot = 0; slot < ${#cpus[@]}; slot++)); do
    index=$((first + slot))
    ((index < ${#lambdas[@]})) || break
    coupling=${lambdas[$index]}
    label=$(printf 'lambda_%0.6f' "$coupling" | tr '.' 'p')
    point=$output/$label
    labels+=("$label")
    (
      taskset -c "${cpus[$slot]}" \
        env CUDA_VISIBLE_DEVICES="${gpus[$slot]}" PYTHONPATH="$repository" \
        "$torch_runner" "$repository/scripts/run_pair_pair_ti_md.py" \
        --reference-model "$reference" \
        --target-model "$target" \
        --restart "$restart" \
        --out "$point" \
        --lambda-value "$coupling" \
        --temperature "$temperature" \
        --target-kedf lkt \
        --phase liquid \
        --steps "$steps" \
        --gamma-per-fs "$gamma_per_fs" \
        --sample-every 10 \
        --seed "$((202608900 + index))" \
        --threads 1 \
        --device cuda \
        --minimum-distance 2.0 \
        --store-positions
    ) >"$output/$label.stdout" 2>&1 &
    pids+=("$!")
    printf '%s launched label=%s cpu=%s gpu=%s pid=%s\n' \
      "$(date -Iseconds)" "$label" "${cpus[$slot]}" "${gpus[$slot]}" "$!"
  done
  for index in "${!pids[@]}"; do
    wait "${pids[$index]}"
    printf '%s finished label=%s\n' "$(date -Iseconds)" "${labels[$index]}"
  done
done

for coupling in "${lambdas[@]}"; do
  label=$(printf 'lambda_%0.6f' "$coupling" | tr '.' 'p')
  env PYTHONPATH=. "$python" scripts/analyze_pair_reference_md.py \
    "$output/$label" \
    --expected liquid \
    --out "$output/$label/phase_analysis.json"
  grep -q '"status": "liquid_verified"' \
    "$output/$label/phase_analysis.json"
done

reports=()
for suffix in d25 d50 d75; do
  case $suffix in
    d25) discard=0.25 ;;
    d50) discard=0.50 ;;
    d75) discard=0.75 ;;
  esac
  report=$output/ti_${suffix}.json
  env PYTHONPATH=. "$python" scripts/analyze_suf_pair_ti.py \
    --root "$output" \
    --out "$report" \
    --discard-fraction "$discard" \
    --du-key du_target_minus_reference_ev_per_atom \
    --reference-label lkt_liquid_proxy6 \
    --minimum-distance 2.0 \
    --target-minimum-distance 2.0 \
    --target-lambda-threshold 0.0 \
    --temperature-tolerance 20 \
    --minimum-liquid-msd 1 \
    --max-block-se 2 \
    --max-half-drift 4 \
    --max-quadrature-difference 2 \
    --minimum-overlap-ess 0.05 \
    --max-overlap-closure 2
  reports+=("$report")
done

"$python" - "$output" "$steps" "$reference" "$target" "${reports[@]}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
steps = int(sys.argv[2])
reference = Path(sys.argv[3])
target = Path(sys.argv[4])
reports = [json.loads(Path(path).read_text()) for path in sys.argv[5:]]
integrals = [
    float(item["delta_f_pair_minus_reference_simpson_mev_per_atom"])
    for item in reports
]
spread = max(integrals) - min(integrals)
phase_reports = [
    json.loads(path.read_text())
    for path in sorted(root.glob("lambda_*/phase_analysis.json"))
]
checks = {
    "nine_windows_complete": len(list(root.glob("lambda_*/summary.json"))) == 9,
    "all_liquid_verified": (
        len(phase_reports) == 9
        and all(item.get("status") == "liquid_verified" for item in phase_reports)
    ),
    "three_discard_reports_verified": all(
        item.get("status") == "verified" for item in reports
    ),
    "discard_spread_le_2_mev_per_atom": spread <= 2.0,
}
payload = {
    "schema": "lkt-liquid-proxy-bridge-pair-ti-pilot-v1",
    "status": "pilot_grid_verified" if all(checks.values()) else "pilot_gate_failed",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "steps_per_window": steps,
    "reference_model": {
        "path": str(reference.resolve()),
        "sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
    },
    "target_model": {
        "path": str(target.resolve()),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    },
    "checks": checks,
    "discard_integrals_mev_per_atom": integrals,
    "discard_spread_mev_per_atom": spread,
    "maximum_block_standard_error_mev_per_atom": max(
        item["block_standard_error_mev_per_atom"] for item in reports
    ),
    "maximum_half_drift_mev_per_atom": max(
        item["half_drift_mev_per_atom"] for item in reports
    ),
    "maximum_quadrature_difference_mev_per_atom": max(
        item["quadrature_difference_mev_per_atom"] for item in reports
    ),
    "minimum_adjacent_effective_sample_fraction": min(
        item["minimum_adjacent_effective_sample_fraction"] for item in reports
    ),
    "maximum_adjacent_closure_mev_per_atom": max(
        item["maximum_adjacent_closure_mev_per_atom"] for item in reports
    ),
}
(root / "pilot_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
(root / "SHA256SUMS").write_text(
    "\n".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    )
    + "\n"
)
print(json.dumps(payload, indent=2, sort_keys=True))
if payload["status"] != "pilot_grid_verified":
    raise SystemExit("pair-pair TI pilot did not pass")
PY

printf '%s verified\n' "$(date -Iseconds)" \
  | tee "$output/pair_pair_ti.done"
