#!/usr/bin/env bash
set -euo pipefail

workspace=${WORKSPACE:-/home/shenwei01/WT_Al_melting_workspace_20260724}
repository=${REPO:-$workspace/repository}
root=$workspace/runs/xwm_lkt_20260727/xwm/free_energy_T0900/liquid_proxy14_to_bridge090_ti_lambda9_steps6000_v1
reference=$workspace/runs/xwm_lkt_20260727/xwm/free_energy_T0900/liquid_suf_pair_proxy_v1_sigma1p40_hardwall.json
target=$workspace/runs/xwm_lkt_20260727/xwm/free_energy_T0900/liquid_variance_bridge_alpha_scan_v1/alpha_0p900/model.json
restart=$workspace/runs/xwm_lkt_20260727/xwm/free_energy_T0900/liquid_suf_pair_proxy_v1_sigma1p40_validation_steps20000/run/checkpoint.json
torch_runner=${TORCH_RUNNER:-$repository/remote_staging/run_node05_cached_cuda_torch.sh}
python=${PYTHON:-$workspace/.venv-reference-cuda/bin/python}
steps=${STEPS:-6000}
temperature=900
gamma_per_fs=0.2
lambdas=(0.0625 0.1875 0.4375 0.8125)
cpus=(36 37 74 75)
gpus=(0 1 2 3)
done_file=$root/overlap_refinement.done
first_failed_file=$root/overlap_refinement.failed
failed_file=$first_failed_file
if [[ -e $failed_file ]]; then
  failed_file=$root/overlap_refinement_round2.failed
fi

[[ -x $torch_runner ]]
[[ -x $python ]]
[[ -f $root/pair_pair_ti_summary.json ]]
[[ -f $root/pair_pair_ti.failed ]]
[[ -f $reference ]]
[[ -f $target ]]
[[ -f $restart ]]
[[ ! -e $done_file ]]

cd "$repository"
env PYTHONPATH=. "$python" - \
  "$root/pair_pair_ti_summary.json" "$reference" "$target" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
reference = Path(sys.argv[2])
target = Path(sys.argv[3])
assert summary["target_kedf"] == "xwm"
assert summary["status"] == "gate_failed"
assert summary["checks"]["all_liquid_verified"] is True
assert summary["checks"]["discard_spread_le_2_mev_per_atom"] is True
assert summary["checks"]["three_discard_reports_verified"] is False
assert summary["minimum_adjacent_effective_sample_fraction"] < 0.05
assert summary["reference_model"]["sha256"] == hashlib.sha256(
    reference.read_bytes()
).hexdigest()
assert summary["target_model"]["sha256"] == hashlib.sha256(
    target.read_bytes()
).hexdigest()
PY

exec > >(tee -a "$root/overlap_refinement.log") 2>&1

mark_failed() {
  local rc=$?
  trap - ERR
  printf '%s failed exit_code=%s\n' "$(date -Iseconds)" "$rc" \
    | tee "$failed_file"
  exit "$rc"
}
trap mark_failed ERR

pids=()
labels=()
for index in "${!lambdas[@]}"; do
  coupling=${lambdas[$index]}
  label=$(printf 'lambda_%0.6f' "$coupling" | tr '.' 'p')
  if [[ -e $root/$label ]]; then
    [[ -f $first_failed_file ]]
    [[ -f $root/$label/summary.json ]]
    [[ -f $root/$label/phase_analysis.json ]]
    grep -q '"status": "liquid_verified"' \
      "$root/$label/phase_analysis.json"
    printf '%s retained completed label=%s\n' "$(date -Iseconds)" "$label"
    continue
  fi
  labels+=("$label")
  (
    taskset -c "${cpus[$index]}" \
      env CUDA_VISIBLE_DEVICES="${gpus[$index]}" PYTHONPATH="$repository" \
      "$torch_runner" "$repository/scripts/run_pair_pair_ti_md.py" \
      --reference-model "$reference" \
      --target-model "$target" \
      --restart "$restart" \
      --out "$root/$label" \
      --lambda-value "$coupling" \
      --temperature "$temperature" \
      --target-kedf xwm \
      --phase liquid \
      --steps "$steps" \
      --gamma-per-fs "$gamma_per_fs" \
      --sample-every 10 \
      --seed "$((202609100 + index))" \
      --threads 1 \
      --device cuda \
      --minimum-distance 2.0 \
      --store-positions
    env PYTHONPATH="$repository" "$python" \
      "$repository/scripts/analyze_pair_reference_md.py" \
      "$root/$label" \
      --expected liquid \
      --out "$root/$label/phase_analysis.json"
    grep -q '"status": "liquid_verified"' \
      "$root/$label/phase_analysis.json"
  ) >"$root/$label.stdout" 2>&1 &
  pids+=("$!")
  printf '%s launched label=%s cpu=%s gpu=%s pid=%s\n' \
    "$(date -Iseconds)" "$label" "${cpus[$index]}" "${gpus[$index]}" "$!"
done

for index in "${!pids[@]}"; do
  wait "${pids[$index]}"
  printf '%s finished label=%s\n' "$(date -Iseconds)" "${labels[$index]}"
done

reports=()
for suffix in d25 d50 d75; do
  case $suffix in
    d25) discard=0.25 ;;
    d50) discard=0.50 ;;
    d75) discard=0.75 ;;
  esac
  report=$root/ti_refined_${suffix}.json
  env PYTHONPATH=. "$python" scripts/analyze_suf_pair_ti.py \
    --root "$root" \
    --out "$report" \
    --discard-fraction "$discard" \
    --du-key du_target_minus_reference_ev_per_atom \
    --reference-label xwm_liquid_proxy14 \
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

"$python" - "$root" "$steps" "$reference" "$target" \
  "${reports[@]}" <<'PY'
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
    "thirteen_windows_complete": (
        len(list(root.glob("lambda_*/summary.json"))) == 13
    ),
    "all_liquid_verified": (
        len(phase_reports) == 13
        and all(item.get("status") == "liquid_verified" for item in phase_reports)
    ),
    "three_discard_reports_verified": all(
        item.get("status") == "verified" for item in reports
    ),
    "discard_spread_le_2_mev_per_atom": spread <= 2.0,
}
payload = {
    "schema": "xwm-liquid-proxy-bridge-overlap-refinement-v1",
    "status": "verified" if all(checks.values()) else "gate_failed",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "target_kedf": "xwm",
    "phase": "liquid",
    "base_steps_per_window": 6000,
    "refinement_steps_per_window": steps,
    "added_lambdas": [0.0625, 0.1875, 0.4375, 0.8125],
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
(root / "overlap_refinement_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
(root / "SHA256SUMS_refined").write_text(
    "\n".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS_refined"
    )
    + "\n"
)
print(json.dumps(payload, indent=2, sort_keys=True))
if payload["status"] != "verified":
    raise SystemExit("overlap refinement did not pass")
PY

printf '%s verified\n' "$(date -Iseconds)" | tee "$done_file"
