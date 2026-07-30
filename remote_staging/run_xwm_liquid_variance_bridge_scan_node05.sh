#!/usr/bin/env bash
set -euo pipefail

workspace=${WORKSPACE:-/home/shenwei01/WT_Al_melting_workspace_20260724}
repo=${REPO:-$workspace/repository}
run_root=${RUN_ROOT:-$workspace/runs/xwm_lkt_20260727}
torch_runner=${TORCH_RUNNER:-$repo/remote_staging/run_node05_cached_cuda_torch.sh}
root=$run_root/xwm/free_energy_T0900
source=${SOURCE:-$root/reference_dataset_v1}
base=${BASE:-$root/liquid_suf_pair_proxy_v1_sigma1p40_hardwall.json}
restart=${RESTART:-$root/liquid_suf_pair_proxy_v1_sigma1p40_validation_steps20000/run/checkpoint.json}
output=${OUTPUT:-$root/liquid_variance_bridge_alpha_scan_v1}
steps=${STEPS:-20000}
temperature=${TEMPERATURE:-900}
seed=${SEED:-20260840}

labels=(
  alpha_0p100 alpha_0p200 alpha_0p300 alpha_0p400
  alpha_0p500 alpha_0p600 alpha_0p750 alpha_0p900
)
alphas=(0.10 0.20 0.30 0.40 0.50 0.60 0.75 0.90)
cpus=(36 37 74 75 76 77 78 79)
gpus=(0 1 2 3 4 5 6 7)

source_frames=$source/frames.jsonl
source_manifest=$source/manifest.json
liquid_dataset=$output/liquid_dataset
fit=$output/xwm_liquid_target_fit_proxy_basis.json
summary=$output/bridge_alpha_scan_summary.json
done_file=$output/pipeline.done
failed_file=$output/pipeline.failed

[[ -x $torch_runner ]]
[[ -f $source_frames ]]
[[ -f $source_manifest ]]
[[ -f $base ]]
[[ -f $restart ]]
[[ ! -e $output ]]

mkdir -p "$output"
exec > >(tee -a "$output/pipeline.log") 2>&1

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

cd "$repo"
env PYTHONPATH=. python3 - "$source_manifest" "$base" "$restart" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
base = json.loads(Path(sys.argv[2]).read_text())
restart = json.loads(Path(sys.argv[3]).read_text())
assert manifest["target_kedf"] == "xwm"
assert int(manifest["phase_counts"]["liquid"]) >= 60
assert base["target_kedf"] == "xwm"
assert base["reference_gate_passed"] is True
assert base["short_range_guard_passed"] is True
assert base["reference_phase"] == "liquid"
assert restart["schema"] == "mpn-pair-reference-md-checkpoint-v1"
assert restart["positions_angstrom"]
assert restart["velocities_angstrom_per_fs"]
assert restart["lattice_angstrom"]
model = base["model"]
assert len(model["centers_angstrom"]) == 77
assert model["centers_angstrom"][0] == 1.8
assert model["centers_angstrom"][-1] == 6.3
assert model["sigma_angstrom"] == 0.1
assert model["cutoff_angstrom"] == 6.5
assert model["repulsive_core"] == {
    "amplitude_ev": 5000.0,
    "cutoff_angstrom": 2.03,
    "power": 2,
}
PY

env PYTHONPATH=. python3 scripts/filter_kedf_reference_dataset.py \
  --source-frames "$source_frames" \
  --source-manifest "$source_manifest" \
  --out "$liquid_dataset" \
  --target-kedf xwm \
  --phase liquid \
  >"$output/filter.stdout"

taskset -c 38 env CUDA_VISIBLE_DEVICES= PYTHONPATH="$repo" \
  "$torch_runner" "$repo/scripts/fit_pair_reference.py" \
  --dataset "$liquid_dataset/frames.jsonl" \
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

env PYTHONPATH=. python3 - "$base" "$fit" <<'PY'
import json
import sys
from pathlib import Path

base = json.loads(Path(sys.argv[1]).read_text())
fit = json.loads(Path(sys.argv[2]).read_text())
assert base["target_kedf"] == fit["target_kedf"] == "xwm"
assert fit["reference_gate_passed"] is True
assert fit["selected_frames"] == 60
base_model = base["model"]
fit_model = fit["model"]
assert len(base_model["centers_angstrom"]) == len(
    fit_model["centers_angstrom"]
)
assert max(
    abs(left - right)
    for left, right in zip(
        base_model["centers_angstrom"],
        fit_model["centers_angstrom"],
    )
) <= 1.0e-12
for key in ("sigma_angstrom", "cutoff_angstrom"):
    assert abs(base_model[key] - fit_model[key]) <= 1.0e-12, key
assert base_model["repulsive_core"] == fit_model["repulsive_core"]
PY

declare -a pids=()
for index in "${!labels[@]}"; do
  label=${labels[$index]}
  alpha=${alphas[$index]}
  model=$output/$label/model.json
  run=$output/$label/liquid_md_steps${steps}
  mkdir -p "$output/$label"
  env PYTHONPATH=. python3 scripts/blend_pair_reference_models.py \
    --base "$base" \
    --correction "$fit" \
    --out "$model" \
    --alpha "$alpha" \
    --target-kedf xwm \
    --phase liquid \
    --core-amplitude 5000 \
    --core-cutoff 2.035 \
    --core-power 2 \
    >"$output/$label/blend.stdout"
  (
    taskset -c "${cpus[$index]}" \
      env CUDA_VISIBLE_DEVICES="${gpus[$index]}" PYTHONPATH="$repo" \
      "$torch_runner" "$repo/scripts/run_pair_reference_md.py" \
      --model "$model" \
      --out "$run" \
      --restart "$restart" \
      --temperature "$temperature" \
      --steps "$steps" \
      --sample-every 10 \
      --seed "$seed" \
      --threads 1 \
      --device cuda \
      --store-positions
    env PYTHONPATH="$repo" python3 \
      "$repo/scripts/analyze_pair_reference_md.py" \
      "$run" --expected liquid --out "$run/phase_analysis.json"
  ) >"$output/$label/run.stdout" 2>&1 &
  pids[$index]=$!
  printf '%s launched label=%s alpha=%s seed=%s cpu=%s gpu=%s pid=%s\n' \
    "$(date -Iseconds)" "$label" "$alpha" "$seed" \
    "${cpus[$index]}" "${gpus[$index]}" "${pids[$index]}"
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failures=$((failures + 1))
  fi
done
[[ $failures -eq 0 ]] || {
  echo "$failures XWM bridge validation jobs failed" >&2
  exit 2
}

python3 - "$base" "$fit" "$restart" "$output" "$steps" "$seed" \
  "${labels[@]}" <<'PY'
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
    run_summary = json.loads((run / "summary.json").read_text())
    phase = json.loads((run / "phase_analysis.json").read_text())
    checks = {
        "bridge_and_core_override_recorded": (
            model.get("status") == "verified"
            and model.get("reference_gate_passed") is True
            and model.get("short_range_guard_passed") is True
            and model.get("core_override") is not None
        ),
        "xwm_liquid_provenance": (
            model.get("target_kedf") == "xwm"
            and model.get("phase") == "liquid"
        ),
        "stable": run_summary.get("stable") is True,
        "nearest_neighbor_gt_2_A": (
            float(run_summary["minimum_distance_angstrom"]) > 2.0
        ),
        "temperature_mean_within_25_K": (
            abs(float(run_summary["temperature_mean_k"]) - 900.0) <= 25.0
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
            "temperature_mean_k": run_summary["temperature_mean_k"],
            "minimum_distance_angstrom": run_summary[
                "minimum_distance_angstrom"
            ],
            "msd_last_angstrom2": run_summary["msd_last_angstrom2"],
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
    "schema": "xwm-liquid-variance-bridge-alpha-scan-v1",
    "status": "verified" if eligible else "no_verified_bridge",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "steps": steps,
    "temperature_k": 900.0,
    "shared_seed": seed,
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
(root / "bridge_alpha_scan_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
if not eligible:
    raise SystemExit("no XWM liquid variance bridge passed")
PY

find "$output" -type f -print0 | sort -z | xargs -0 sha256sum \
  >"$output/SHA256SUMS"
printf '%s verified summary=%s\n' "$(date -Iseconds)" "$summary" \
  | tee "$done_file"
