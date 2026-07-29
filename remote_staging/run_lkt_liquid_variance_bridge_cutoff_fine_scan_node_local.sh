#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/home/shenwei01/WT_Al_melting_workspace_20260724/repository}
run_root=${RUN_ROOT:-/home/shenwei01/WT_Al_melting_workspace_20260724/runs/xwm_lkt_20260727}
torch_runner=${TORCH_RUNNER:-$repo/remote_staging/run_node05_cached_cuda_torch.sh}
root=$run_root/lkt/free_energy_T0900
source=${SOURCE:-$root/liquid_variance_bridge_scan_v2}
base=${BASE:-$root/liquid_suf_pair_proxy_v6_hardwall.json}
restart=${RESTART:-$root/liquid_suf_pair_proxy_v6_validation_steps20000/run/checkpoint.json}
output=${OUTPUT:-$root/liquid_variance_bridge_cutoff_fine_scan_v4}
steps=${STEPS:-50000}
alpha=0.75
seed=20260770
labels=(cutoff_2p0325 cutoff_2p0350 cutoff_2p0375 cutoff_2p0400)
cutoffs=(2.0325 2.0350 2.0375 2.0400)
cpus=(36 37 74 75)
gpus=(0 1 2 3)

fit=$source/lkt_liquid_target_fit_proxy_basis.json
[[ -x $torch_runner ]]
[[ -f $fit ]]
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

declare -a pids=()
for index in "${!labels[@]}"; do
  label=${labels[$index]}
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
    --core-amplitude 5000 \
    --core-cutoff "${cutoffs[$index]}" \
    --core-power 2 \
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
      --seed "$seed" \
      --threads 1 \
      --device cuda \
      --store-positions
    env PYTHONPATH="$repo" python3 "$repo/scripts/analyze_pair_reference_md.py" \
      "$run" --expected liquid --out "$run/phase_analysis.json"
  ) >"$output/$label/run.stdout" 2>&1 &
  pids[$index]=$!
  printf '%s launched label=%s cutoff=%s seed=%s cpu=%s gpu=%s pid=%s\n' \
    "$(date -Iseconds)" "$label" "${cutoffs[$index]}" "$seed" \
    "${cpus[$index]}" "${gpus[$index]}" "${pids[$index]}"
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failures=$((failures + 1))
  fi
done
[[ $failures -eq 0 ]] || {
  echo "$failures fine-scan validations failed to execute" >&2
  exit 2
}

python3 - "$base" "$fit" "$restart" "$output" "$steps" "$seed" "${labels[@]}" <<'PY'
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
seed = int(sys.argv[6])
labels = sys.argv[7:]
rows = []
for label in labels:
    model_path = root / label / "model.json"
    model = json.loads(model_path.read_text())
    run = root / label / f"liquid_md_steps{steps}"
    summary = json.loads((run / "summary.json").read_text())
    phase = json.loads((run / "phase_analysis.json").read_text())
    checks = {
        "bridge_and_core_override_recorded": (
            model.get("status") == "verified"
            and model.get("reference_gate_passed") is True
            and model.get("core_override") is not None
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
            "core_override": model["core_override"],
            "seed": seed,
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
    key=lambda row: row["core_override"]["cutoff_angstrom"],
)
payload = {
    "schema": "lkt-liquid-variance-bridge-cutoff-fine-scan-v1",
    "status": "verified" if eligible else "no_verified_bridge",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "steps": steps,
    "shared_seed": seed,
    "alpha_correction": 0.75,
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
(root / "bridge_cutoff_fine_scan_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
if not eligible:
    raise SystemExit("no fine-scan LKT liquid variance bridge passed")
PY

find "$output" -type f -print0 | sort -z | xargs -0 sha256sum \
  >"$output/SHA256SUMS"
printf '%s verified\n' "$(date -Iseconds)" | tee "$done_file"
