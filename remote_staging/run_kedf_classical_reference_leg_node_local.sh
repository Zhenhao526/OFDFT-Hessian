#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/home/shenwei01/WT_Al_melting_workspace_20260724/repository}
python=${PYTHON:-/home/shenwei01/WT_Al_melting_workspace_20260724/.venv-reference-cuda/bin/python}
model=${MODEL:?MODEL is required}
restart=${RESTART:?RESTART is required}
reference=${REFERENCE:?REFERENCE is required}
output=${OUTPUT:?OUTPUT is required}
leg_kind=${LEG_KIND:?LEG_KIND must be einstein or suf}
expected_phase=${EXPECTED_PHASE:?EXPECTED_PHASE must be solid or liquid}
steps=${STEPS:-6000}
grid_points=${GRID_POINTS:-17}
lambda_power=${LAMBDA_POWER:-1}
gamma_per_fs=${GAMMA_PER_FS:-0.1}
temperature=${TEMPERATURE:-900}
sample_every=${SAMPLE_EVERY:-10}
cuda_device=${CUDA_DEVICE:-5}
cpu=${CPU:-36}

[[ $leg_kind == einstein || $leg_kind == suf ]]
[[ $expected_phase == solid || $expected_phase == liquid ]]
[[ -x $python && -f $model && -f $restart && -f $reference ]]
[[ ! -e $output ]]

"$python" - "$model" "$restart" "$reference" "$grid_points" "$lambda_power" <<'PY'
import json
import sys

model = json.load(open(sys.argv[1], encoding="utf-8"))
restart = json.load(open(sys.argv[2], encoding="utf-8"))
reference = json.load(open(sys.argv[3], encoding="utf-8"))
grid_points = int(sys.argv[4])
lambda_power = float(sys.argv[5])
checks = {
    "model_static_gate": model.get("reference_gate_passed") is True,
    "restart_has_positions": bool(restart.get("positions_angstrom")),
    "restart_has_lattice": bool(restart.get("lattice_angstrom")),
    "odd_grid": grid_points >= 3 and grid_points % 2 == 1,
    "valid_lambda_power": lambda_power >= 1.0,
    "reference_is_structured": isinstance(reference, dict),
}
print(json.dumps(checks, sort_keys=True))
if not all(checks.values()):
    raise SystemExit("classical reference leg preflight failed")
PY

mkdir -p "$output"
exec > >(tee -a "$output/pipeline.log") 2>&1

mark_failed() {
  local rc=$?
  trap - ERR
  printf '%s failed exit_code=%s\n' "$(date -Iseconds)" "$rc" \
    | tee "$output/classical_leg.failed"
  exit "$rc"
}
trap mark_failed ERR

sha256sum "$model" "$restart" "$reference" >"$output/SOURCE_SHA256"

mapfile -t lambdas < <(
  python3 - "$grid_points" "$lambda_power" <<'PY'
import sys

points = int(sys.argv[1])
power = float(sys.argv[2])
for index in range(points):
    coordinate = index / (points - 1)
    print(f"{coordinate**power:.12f}")
PY
)

suf_p=
suf_sigma=
suf_cutoff=
if [[ $leg_kind == suf ]]; then
  read -r suf_p suf_sigma suf_cutoff < <(
    "$python" - "$reference" <<'PY'
import json
import sys

document = json.load(open(sys.argv[1], encoding="utf-8"))
print(
    int(document["p"]),
    float(document["selected_sigma_angstrom"]),
    float(document["cutoff_sigma"]),
)
PY
  )
fi

cd "$repo"
for index in "${!lambdas[@]}"; do
  coupling=${lambdas[$index]}
  label=$(printf 'lambda_%0.6f' "$coupling" | tr '.' 'p')
  point=$output/$label
  command=(
    "$python"
  )
  if [[ $leg_kind == einstein ]]; then
    command+=(
      scripts/run_einstein_pair_ti_md.py
      --model "$model"
      --restart "$restart"
      --einstein-reference "$reference"
    )
  else
    command+=(
      scripts/run_suf_pair_ti_md.py
      --model "$model"
      --restart "$restart"
      --suf-p "$suf_p"
      --suf-sigma "$suf_sigma"
      --suf-cutoff-sigma "$suf_cutoff"
    )
  fi
  command+=(
    --out "$point"
    --lambda-value "$coupling"
    --temperature "$temperature"
    --steps "$steps"
    --gamma-per-fs "$gamma_per_fs"
    --sample-every "$sample_every"
    --seed "$((202608500 + index))"
    --threads 1
    --device cuda
    --store-positions
  )
  printf '%s launched label=%s gpu=%s cpu=%s\n' \
    "$(date -Iseconds)" "$label" "$cuda_device" "$cpu"
  taskset -c "$cpu" env CUDA_VISIBLE_DEVICES="$cuda_device" PYTHONPATH="$repo" \
    "${command[@]}" >"$output/$label.stdout" 2>&1
  printf '%s finished label=%s\n' "$(date -Iseconds)" "$label"
done

target_label=$(printf 'lambda_%0.6f' 1.0 | tr '.' 'p')
env PYTHONPATH=. "$python" scripts/analyze_pair_reference_md.py \
  "$output/$target_label" \
  --expected "$expected_phase" \
  --out "$output/target_phase_analysis.json"

reports=()
for suffix in d25 d50 d75; do
  case $suffix in
    d25) discard=0.25 ;;
    d50) discard=0.50 ;;
    d75) discard=0.75 ;;
  esac
  report=$output/ti_${suffix}.json
  reference_label=sUF
  [[ $leg_kind == einstein ]] && reference_label=Einstein
  analysis_args=(
    scripts/analyze_suf_pair_ti.py
    --root "$output"
    --out "$report"
    --discard-fraction "$discard"
    --reference-label "$reference_label"
    --minimum-distance 1.5
    --target-minimum-distance 2.0
    --target-lambda-threshold 1.0
    --temperature-tolerance 25
    --max-block-se 1
    --max-half-drift 2
    --max-quadrature-difference 1
    --minimum-overlap-ess 0.05
    --max-overlap-closure 2
    --integration-coordinate-power "$lambda_power"
  )
  if [[ $leg_kind == einstein ]]; then
    analysis_args+=(
      --du-key du_pair_minus_harmonic_ev_per_atom
      --minimum-liquid-msd 0
    )
  else
    analysis_args+=(--minimum-liquid-msd 1)
  fi
  env PYTHONPATH=. "$python" "${analysis_args[@]}"
  reports+=("$report")
done

"$python" - "$output" "$grid_points" "$lambda_power" "$leg_kind" "$expected_phase" "${reports[@]}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
grid_points = int(sys.argv[2])
lambda_power = float(sys.argv[3])
leg_kind = sys.argv[4]
expected_phase = sys.argv[5]
reports = [json.load(open(path, encoding="utf-8")) for path in sys.argv[6:]]
phase = json.load(open(root / "target_phase_analysis.json", encoding="utf-8"))
integrals = [
    float(item["delta_f_pair_minus_reference_simpson_mev_per_atom"])
    for item in reports
]
spread = max(integrals) - min(integrals)
checks = {
    "all_requested_windows_complete": len(
        list(root.glob("lambda_*/summary.json"))
    ) == grid_points,
    "three_discard_reports_verified": all(
        item.get("status") == "verified" for item in reports
    ),
    "discard_spread_le_1_mev_per_atom": spread <= 1.0,
    "target_phase_verified": phase.get("status") == f"{expected_phase}_verified",
}
payload = {
    "schema": "kedf-classical-reference-leg-summary-v1",
    "status": "verified" if all(checks.values()) else "needs_extension",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "leg_kind": leg_kind,
    "expected_phase": expected_phase,
    "grid_points": grid_points,
    "lambda_equals_x_to_power": lambda_power,
    "checks": checks,
    "discard_integrals_mev_per_atom": integrals,
    "discard_spread_mev_per_atom": spread,
    "minimum_adjacent_effective_sample_fraction": min(
        item["minimum_adjacent_effective_sample_fraction"] for item in reports
    ),
    "maximum_adjacent_closure_mev_per_atom": max(
        item["maximum_adjacent_closure_mev_per_atom"] for item in reports
    ),
}
(root / "classical_leg_summary.json").write_text(
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
if payload["status"] != "verified":
    raise SystemExit("classical reference leg needs extension")
PY

printf '%s verified\n' "$(date -Iseconds)" | tee "$output/classical_leg.done"
