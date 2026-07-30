#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 METHOD REPOSITORY REFERENCE TARGET RESTART OUTPUT REFERENCE_LABEL STEPS" >&2
  exit 2
fi

method=$1
repository=$2
reference=$3
target=$4
restart=$5
output=$6
reference_label=$7
steps=$8
workspace=${WORKSPACE:-/home/shenwei01/WT_Al_melting_workspace_20260724}
torch_runner=${TORCH_RUNNER:-$repository/remote_staging/run_node05_cached_cuda_torch.sh}
python=${PYTHON:-$workspace/.venv-reference-cuda/bin/python}
temperature=${TEMPERATURE:-900}
gamma_per_fs=${GAMMA_PER_FS:-0.2}
seed_base=${SEED_BASE:-202609000}
lambdas=(0.000 0.125 0.250 0.375 0.500 0.625 0.750 0.875 1.000)
cpus=(36 37 74 75 76 77 78 79)
gpus=(0 1 2 3 4 5 6 7)

case $method in
  xwm|lkt) ;;
  *)
    echo "unsupported KEDF: $method" >&2
    exit 2
    ;;
esac
[[ $steps =~ ^[1-9][0-9]*$ ]]
[[ -x $torch_runner ]]
[[ -x $python ]]
[[ -f $reference ]]
[[ -f $target ]]
[[ -f $restart ]]
[[ ! -e $output ]]

cd "$repository"
env PYTHONPATH=. "$torch_runner" - \
  "$method" "$reference" "$target" "$restart" <<'PY'
import json
import sys
from pathlib import Path

from scripts.run_pair_pair_ti_md import load_pair_document

method = sys.argv[1]
reference = Path(sys.argv[2])
target = Path(sys.argv[3])
restart_path = Path(sys.argv[4])
for path in (reference, target):
    load_pair_document(path, target_kedf=method, phase="liquid")
restart = json.loads(restart_path.read_text())
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
        --target-kedf "$method" \
        --phase liquid \
        --steps "$steps" \
        --gamma-per-fs "$gamma_per_fs" \
        --sample-every 10 \
        --seed "$((seed_base + index))" \
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
    printf '%s finished label=%s\n' \
      "$(date -Iseconds)" "${labels[$index]}"
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
    --reference-label "$reference_label" \
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

"$python" - \
  "$method" "$output" "$steps" "$reference" "$target" \
  "${reports[@]}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

method = sys.argv[1]
root = Path(sys.argv[2])
steps = int(sys.argv[3])
reference = Path(sys.argv[4])
target = Path(sys.argv[5])
reports = [json.loads(Path(path).read_text()) for path in sys.argv[6:]]
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
    "schema": "kedf-liquid-proxy-bridge-pair-ti-v1",
    "status": "verified" if all(checks.values()) else "gate_failed",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "target_kedf": method,
    "phase": "liquid",
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
(root / "pair_pair_ti_summary.json").write_text(
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
    raise SystemExit("pair-pair TI did not pass")
PY

printf '%s verified\n' "$(date -Iseconds)" \
  | tee "$output/pair_pair_ti.done"
