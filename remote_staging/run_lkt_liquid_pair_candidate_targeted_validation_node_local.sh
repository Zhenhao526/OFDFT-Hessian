#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/home/shenwei01/WT_Al_melting_workspace_20260724/repository}
run_root=${RUN_ROOT:-/home/shenwei01/WT_Al_melting_workspace_20260724/runs/xwm_lkt_20260727}
torch_runner=${TORCH_RUNNER:-$repo/remote_staging/run_node05_cached_cuda_torch.sh}
source=${SOURCE:-$run_root/lkt/free_energy_T0900/liquid_pair_candidate_scan_v2_hardwall}
output=${OUTPUT:-$run_root/lkt/free_energy_T0900/liquid_pair_candidate_targeted_validation_v3}
steps=${STEPS:-30000}
labels=(liquid_s017_b33 liquid_s014_b45)
cpus=(36 37)
gpus=(0 1)

[[ -x $torch_runner ]]
[[ -f $source/candidate_summary.json ]]
[[ ! -e $output ]]

mkdir -p "$output"
log=$output/pipeline.log
done_file=$output/pipeline.done
failed_file=$output/pipeline.failed
exec > >(tee -a "$log") 2>&1

mark_failed() {
  local rc=$?
  trap - ERR
  if [[ ! -e $done_file ]]; then
    printf '%s failed exit_code=%s\n' "$(date -Iseconds)" "$rc" \
      | tee "$failed_file"
  fi
  exit "$rc"
}
trap mark_failed ERR

python3 - "$source" "${labels[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
labels = sys.argv[2:]
summary = json.loads((root / "candidate_summary.json").read_text())
rows = {row["label"]: row for row in summary["candidates"]}
for label in labels:
    row = rows[label]
    checks = row["checks"]
    assert checks["lkt_liquid_only_provenance"]
    assert checks["same_hard_wall_as_verified_proxy"]
    assert checks["static_reference_gate"]
    assert checks["stable"]
    assert checks["nearest_neighbor_gt_2_A"]
    assert checks["temperature_mean_within_25_K"]
    assert row["minimum_distance_angstrom"] > 2.25
    print(label, row["model_sha256"], row["minimum_distance_angstrom"])
PY

declare -a pids=()
for index in "${!labels[@]}"; do
  label=${labels[$index]}
  candidate=$source/$label
  run=$output/$label/liquid_md_steps${steps}
  mkdir -p "$output/$label"
  (
    taskset -c "${cpus[$index]}" \
      env CUDA_VISIBLE_DEVICES="${gpus[$index]}" PYTHONPATH="$repo" \
      "$torch_runner" "$repo/scripts/run_pair_reference_md.py" \
      --model "$candidate/model.json" \
      --out "$run" \
      --restart "$candidate/liquid_md_steps10000/checkpoint.json" \
      --temperature 900 \
      --steps "$steps" \
      --sample-every 10 \
      --seed "$((20260740 + index))" \
      --threads 1 \
      --device cuda \
      --store-positions
    env PYTHONPATH="$repo" python3 "$repo/scripts/analyze_pair_reference_md.py" \
      "$run" --expected liquid --out "$run/phase_analysis.json"
  ) >"$output/$label/run.stdout" 2>&1 &
  pids[$index]=$!
  printf '%s launched label=%s cpu=%s gpu=%s pid=%s\n' \
    "$(date -Iseconds)" "$label" "${cpus[$index]}" "${gpus[$index]}" "${pids[$index]}"
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failures=$((failures + 1))
  fi
done
[[ $failures -eq 0 ]] || {
  echo "$failures targeted validations failed to execute" >&2
  exit 2
}

python3 - "$source" "$output" "$steps" "${labels[@]}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

source = Path(sys.argv[1])
root = Path(sys.argv[2])
steps = int(sys.argv[3])
labels = sys.argv[4:]
source_summary = json.loads((source / "candidate_summary.json").read_text())
source_rows = {row["label"]: row for row in source_summary["candidates"]}
rows = []
for label in labels:
    run = root / label / f"liquid_md_steps{steps}"
    summary = json.loads((run / "summary.json").read_text())
    phase = json.loads((run / "phase_analysis.json").read_text())
    model_path = source / label / "model.json"
    checks = {
        "source_static_and_hardwall_gates": all(
            source_rows[label]["checks"][key]
            for key in (
                "lkt_liquid_only_provenance",
                "same_hard_wall_as_verified_proxy",
                "static_reference_gate",
            )
        ),
        "stable": summary.get("stable") is True,
        "nearest_neighbor_gt_2_A": (
            float(summary["minimum_distance_angstrom"]) > 2.0
        ),
        "temperature_mean_within_25_K": (
            abs(float(summary["temperature_mean_k"]) - 900.0) <= 25.0
        ),
        "liquid_verified": phase.get("status") == "liquid_verified",
    }
    rows.append(
        {
            "label": label,
            "eligible": all(checks.values()),
            "checks": checks,
            "model": str(model_path.resolve()),
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "validation": source_rows[label]["validation"],
            "temperature_mean_k": summary["temperature_mean_k"],
            "minimum_distance_angstrom": summary["minimum_distance_angstrom"],
            "msd_last_angstrom2": summary["msd_last_angstrom2"],
            "late_msd_slope_angstrom2_per_step": phase["MSD_A2"][
                "late_slope_per_step"
            ],
            "final_csp": phase["CSP_final"],
        }
    )
eligible = [row for row in rows if row["eligible"]]
eligible.sort(
    key=lambda row: (
        row["validation"]["force_rmse_ev_per_angstrom"],
        row["validation"]["energy_rmse_ev_per_atom"],
        -row["late_msd_slope_angstrom2_per_step"],
    )
)
payload = {
    "schema": "lkt-liquid-hardwall-pair-targeted-validation-v1",
    "status": "verified" if eligible else "no_verified_candidate",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "continuation_steps": steps,
    "source": str(source.resolve()),
    "candidates": rows,
    "recommended_candidate": eligible[0] if eligible else None,
}
(root / "targeted_validation_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
if not eligible:
    raise SystemExit("no targeted LKT liquid pair candidate passed")
PY

find "$output" -type f -print0 | sort -z | xargs -0 sha256sum \
  >"$output/SHA256SUMS"
printf '%s verified\n' "$(date -Iseconds)" | tee "$done_file"
