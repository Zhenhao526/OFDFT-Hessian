#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/home/shenwei01/WT_Al_melting_workspace_20260724/repository}
run_root=${RUN_ROOT:-/home/shenwei01/WT_Al_melting_workspace_20260724/runs/xwm_lkt_20260727}
torch_runner=${TORCH_RUNNER:-$repo/remote_staging/run_node05_cached_cuda_torch.sh}
model=${MODEL:-$run_root/lkt/free_energy_T0900/liquid_suf_pair_proxy_v6_hardwall.json}
validation=${VALIDATION:-$run_root/lkt/free_energy_T0900/liquid_suf_pair_proxy_v6_validation_steps20000}
output=${OUTPUT:-$run_root/lkt/free_energy_T0900/liquid_suf_proxy_ti_lambda9_steps3000_v1}
steps=${STEPS:-3000}
temperature=${TEMPERATURE:-900}
cpu_list=${CPU_LIST:-36,37,74,75}

IFS=, read -r -a cpus <<< "$cpu_list"
[[ ${#cpus[@]} -ge 1 ]] || {
  echo "CPU_LIST must contain at least one CPU" >&2
  exit 2
}
[[ -x $torch_runner && -f $model ]] || {
  echo "missing torch runner or proxy model" >&2
  exit 2
}
[[ -f $validation/validation_summary.json ]] || {
  echo "missing proxy dynamics validation" >&2
  exit 2
}
[[ -f $validation/run/checkpoint.json ]] || {
  echo "missing verified proxy checkpoint" >&2
  exit 2
}
[[ ! -e $output ]] || {
  echo "refusing existing output: $output" >&2
  exit 2
}

python3 - "$model" "$validation/validation_summary.json" <<'PY'
import json
import sys

model = json.load(open(sys.argv[1], encoding="utf-8"))
validation = json.load(open(sys.argv[2], encoding="utf-8"))
checks = {
    "model_static_gate": model.get("reference_gate_passed") is True,
    "model_is_lkt_liquid_suf_proxy": (
        model.get("target_kedf") == "lkt"
        and model.get("reference_phase") == "liquid"
        and model.get("reference_kind") == "suf_radial_proxy"
    ),
    "dynamics_verified": validation.get("status") == "verified",
}
print(json.dumps(checks, sort_keys=True))
if not all(checks.values()):
    raise SystemExit("sUF proxy preflight failed")
PY

mkdir -p "$output"
exec > >(tee -a "$output/pipeline.log") 2>&1

mark_failed() {
  local rc=$?
  trap - ERR
  printf '%s failed exit_code=%s\n' "$(date -Iseconds)" "$rc" \
    | tee "$output/pilot.failed"
  exit "$rc"
}
trap mark_failed ERR

sha256sum \
  "$model" \
  "$validation/validation_summary.json" \
  "$validation/run/checkpoint.json" \
  > "$output/SOURCE_SHA256"

lambdas=(0.000 0.125 0.250 0.375 0.500 0.625 0.750 0.875 1.000)
cd "$repo"
for ((first = 0; first < ${#lambdas[@]}; first += ${#cpus[@]})); do
  pids=()
  labels=()
  for ((slot = 0; slot < ${#cpus[@]}; slot++)); do
    index=$((first + slot))
    ((index < ${#lambdas[@]})) || break
    coupling=${lambdas[$index]}
    label=$(printf 'lambda_%0.3f' "$coupling" | tr '.' 'p')
    point=$output/$label
    labels+=("$label")
    (
      taskset -c "${cpus[$slot]}" env CUDA_VISIBLE_DEVICES= PYTHONPATH="$repo" \
        "$torch_runner" "$repo/scripts/run_suf_pair_ti_md.py" \
        --model "$model" \
        --restart "$validation/run/checkpoint.json" \
        --out "$point" \
        --lambda-value "$coupling" \
        --temperature "$temperature" \
        --suf-p 50 \
        --suf-sigma 1.28 \
        --suf-cutoff-sigma 5.0 \
        --steps "$steps" \
        --sample-every 10 \
        --seed "$((202608100 + index))" \
        --threads 1 \
        --device cpu
    ) >"$output/$label.stdout" 2>&1 &
    pids+=("$!")
    printf '%s launched label=%s cpu=%s pid=%s\n' \
      "$(date -Iseconds)" "$label" "${cpus[$slot]}" "$!"
  done
  for index in "${!pids[@]}"; do
    wait "${pids[$index]}"
    printf '%s finished label=%s\n' "$(date -Iseconds)" "${labels[$index]}"
  done
done

reports=()
for suffix in d25 d50 d75; do
  case $suffix in
    d25) discard=0.25 ;;
    d50) discard=0.50 ;;
    d75) discard=0.75 ;;
  esac
  report=$output/ti_pilot_${suffix}.json
  env PYTHONPATH=. python3 scripts/analyze_suf_pair_ti.py \
    --root "$output" \
    --out "$report" \
    --discard-fraction "$discard" \
    --temperature-tolerance 25 \
    --minimum-distance 1.5 \
    --target-minimum-distance 2.0 \
    --target-lambda-threshold 1.0 \
    --minimum-liquid-msd 1 \
    --max-block-se 2 \
    --max-half-drift 4 \
    --max-quadrature-difference 2 \
    --minimum-overlap-ess 0.05 \
    --max-overlap-closure 2
  reports+=("$report")
done

python3 - "$output" "${reports[@]}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
reports = [json.loads(Path(path).read_text()) for path in sys.argv[2:]]
integrals = [
    float(report["delta_f_pair_minus_suf_simpson_mev_per_atom"])
    for report in reports
]
spread = max(integrals) - min(integrals)
checks = {
    "three_discard_reports_verified": all(
        report.get("status") == "verified" for report in reports
    ),
    "integral_discard_spread_le_2_mev_per_atom": spread <= 2.0,
    "all_nine_windows_complete": len(
        list(root.glob("lambda_*/summary.json"))
    ) == 9,
}
payload = {
    "schema": "lkt-suf-proxy-ti-pilot-summary-v1",
    "status": "pilot_grid_verified" if all(checks.values()) else "pilot_gate_failed",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "checks": checks,
    "discard_integrals_mev_per_atom": integrals,
    "discard_spread_mev_per_atom": spread,
    "minimum_adjacent_effective_sample_fraction": min(
        report["minimum_adjacent_effective_sample_fraction"]
        for report in reports
    ),
    "maximum_adjacent_closure_mev_per_atom": max(
        report["maximum_adjacent_closure_mev_per_atom"]
        for report in reports
    ),
}
path = root / "pilot_summary.json"
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
(root / "SHA256SUMS").write_text(
    "\n".join(
        f"{hashlib.sha256(item.read_bytes()).hexdigest()}  {item}"
        for item in sorted(root.rglob("*"))
        if item.is_file() and item.name != "SHA256SUMS"
    )
    + "\n"
)
print(json.dumps(payload, indent=2, sort_keys=True))
if payload["status"] != "pilot_grid_verified":
    raise SystemExit("sUF-proxy TI pilot did not pass")
PY

printf '%s verified\n' "$(date -Iseconds)" | tee "$output/pilot.done"
