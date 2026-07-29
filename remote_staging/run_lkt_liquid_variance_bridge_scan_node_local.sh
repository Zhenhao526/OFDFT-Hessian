#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/home/shenwei01/WT_Al_melting_workspace_20260724/repository}
run_root=${RUN_ROOT:-/home/shenwei01/WT_Al_melting_workspace_20260724/runs/xwm_lkt_20260727}
torch_runner=${TORCH_RUNNER:-$repo/remote_staging/run_node05_cached_cuda_torch.sh}
root=$run_root/lkt/free_energy_T0900
source=${SOURCE:-$root/liquid_pair_candidate_scan_v2_hardwall/liquid_dataset}
base=${BASE:-$root/liquid_suf_pair_proxy_v6_hardwall.json}
restart=${RESTART:-$root/liquid_suf_pair_proxy_v6_validation_steps20000/run/checkpoint.json}
output=${OUTPUT:-$root/liquid_variance_bridge_scan_v1}
steps=${STEPS:-30000}
alphas=(0.75 0.85 0.90 0.95)
labels=(alpha_0p750 alpha_0p850 alpha_0p900 alpha_0p950)
cpus=(36 37 74 75)
gpus=(0 1 2 3)

[[ -x $torch_runner ]]
[[ -f $source/frames.jsonl ]]
[[ -f $source/manifest.json ]]
[[ -f $base ]]
[[ -f $restart ]]
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

fit=$output/lkt_liquid_target_fit_proxy_basis.json
taskset -c 38 env CUDA_VISIBLE_DEVICES= PYTHONPATH="$repo" \
  "$torch_runner" "$repo/scripts/fit_pair_reference.py" \
  --dataset "$source/frames.jsonl" \
  --out "$fit" \
  --max-per-phase 60 \
  --basis-count 77 \
  --basis-min 1.8 \
  --basis-max 6.3 \
  --sigma 0.1 \
  --force-scale 0.1 \
  --ridge 1e-7 \
  --core-amplitude 5000 \
  --core-cutoff 2.03 \
  --core-power 2 \
  --threads 1 \
  >"$output/fit.stdout"

python3 - "$base" "$fit" <<'PY'
import json
import sys

base = json.load(open(sys.argv[1]))
fit = json.load(open(sys.argv[2]))
assert base["target_kedf"] == fit["target_kedf"] == "lkt"
assert fit["reference_gate_passed"]
for key in (
    "centers_angstrom",
    "sigma_angstrom",
    "cutoff_angstrom",
    "repulsive_core",
):
    assert base["model"][key] == fit["model"][key], key
print("fit_validation", fit["validation"])
PY

declare -a pids=()
for index in "${!labels[@]}"; do
  label=${labels[$index]}
  alpha=${alphas[$index]}
  model=$output/$label/model.json
  run=$output/$label/liquid_md_steps${steps}
  mkdir -p "$output/$label"
  env PYTHONPATH="$repo" python3 "$repo/scripts/blend_pair_reference_models.py" \
    --base "$base" \
    --correction "$fit" \
    --out "$model" \
    --alpha "$alpha" \
    --target-kedf lkt \
    --phase liquid \
    >"$output/$label/blend.stdout"
  (
    taskset -c "${cpus[$index]}" \
      env CUDA_VISIBLE_DEVICES="${gpus[$index]}" PYTHONPATH="$repo" \
      "$torch_runner" "$repo/scripts/run_pair_reference_md.py" \
      --model "$model" \
      --out "$run" \
      --restart "$restart" \
      --temperature 900 \
      --steps "$steps" \
      --sample-every 10 \
      --seed "$((20260750 + index))" \
      --threads 1 \
      --device cuda \
      --store-positions
    env PYTHONPATH="$repo" python3 "$repo/scripts/analyze_pair_reference_md.py" \
      "$run" --expected liquid --out "$run/phase_analysis.json"
  ) >"$output/$label/run.stdout" 2>&1 &
  pids[$index]=$!
  printf '%s launched label=%s alpha=%s cpu=%s gpu=%s pid=%s\n' \
    "$(date -Iseconds)" "$label" "$alpha" "${cpus[$index]}" \
    "${gpus[$index]}" "${pids[$index]}"
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failures=$((failures + 1))
  fi
done
[[ $failures -eq 0 ]] || {
  echo "$failures bridge validations failed to execute" >&2
  exit 2
}

python3 - "$base" "$fit" "$restart" "$output" "$steps" "${labels[@]}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

base = Path(sys.argv[1])
fit = Path(sys.argv[2])
restart = Path(sys.argv[3])
root = Path(sys.argv[4])
steps = int(sys.argv[5])
labels = sys.argv[6:]
rows = []
for label in labels:
    model_path = root / label / "model.json"
    model = json.loads(model_path.read_text())
    run = root / label / f"liquid_md_steps{steps}"
    summary = json.loads((run / "summary.json").read_text())
    phase = json.loads((run / "phase_analysis.json").read_text())
    checks = {
        "blended_model_verified": (
            model.get("status") == "verified"
            and model.get("reference_gate_passed") is True
        ),
        "lkt_liquid_provenance": (
            model.get("target_kedf") == "lkt"
            and model.get("phase") == "liquid"
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
            "alpha_correction": model["alpha_correction"],
            "eligible": all(checks.values()),
            "checks": checks,
            "model": str(model_path.resolve()),
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "temperature_mean_k": summary["temperature_mean_k"],
            "minimum_distance_angstrom": summary["minimum_distance_angstrom"],
            "msd_last_angstrom2": summary["msd_last_angstrom2"],
            "late_msd_slope_angstrom2_per_step": phase["MSD_A2"][
                "late_slope_per_step"
            ],
            "final_csp": phase["CSP_final"],
        }
    )
eligible = sorted(
    (row for row in rows if row["eligible"]),
    key=lambda row: -row["alpha_correction"],
)
payload = {
    "schema": "lkt-liquid-variance-bridge-scan-v1",
    "status": "verified" if eligible else "no_verified_bridge",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "steps": steps,
    "base": {
        "path": str(base.resolve()),
        "sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
    },
    "target_fit": {
        "path": str(fit.resolve()),
        "sha256": hashlib.sha256(fit.read_bytes()).hexdigest(),
    },
    "restart": {
        "path": str(restart.resolve()),
        "sha256": hashlib.sha256(restart.read_bytes()).hexdigest(),
    },
    "candidates": rows,
    "recommended_bridge": eligible[0] if eligible else None,
}
(root / "bridge_scan_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
if not eligible:
    raise SystemExit("no LKT liquid variance bridge passed")
PY

find "$output" -type f -print0 | sort -z | xargs -0 sha256sum \
  >"$output/SHA256SUMS"
printf '%s verified\n' "$(date -Iseconds)" | tee "$done_file"
