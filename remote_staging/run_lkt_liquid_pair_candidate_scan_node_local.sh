#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/home/shenwei01/WT_Al_melting_workspace_20260724/repository}
run_root=${RUN_ROOT:-/home/shenwei01/WT_Al_melting_workspace_20260724/runs/xwm_lkt_20260727}
torch_runner=${TORCH_RUNNER:-$repo/remote_staging/run_node05_cached_cuda_torch.sh}
output=${OUTPUT:-$run_root/lkt/free_energy_T0900/liquid_pair_candidate_scan_v1}
steps=${STEPS:-10000}
temperature=${TEMPERATURE:-900}
cpu_list=${CPU_LIST:-36,37,74,75}

IFS=, read -r -a cpus <<< "$cpu_list"
[[ ${#cpus[@]} -eq 4 ]] || {
  echo "CPU_LIST must contain exactly four CPUs" >&2
  exit 2
}
[[ -x $torch_runner ]] || {
  echo "missing torch runner: $torch_runner" >&2
  exit 2
}
[[ ! -e $output ]] || {
  echo "refusing existing output: $output" >&2
  exit 2
}

source_root=$run_root/lkt/free_energy_T0900/reference_dataset_v1
source_frames=$source_root/frames.jsonl
source_manifest=$source_root/manifest.json
[[ -f $source_frames && -f $source_manifest ]] || {
  echo "missing LKT reference dataset" >&2
  exit 2
}

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

cd "$repo"
env PYTHONPATH=. python3 scripts/filter_kedf_reference_dataset.py \
  --source-frames "$source_frames" \
  --source-manifest "$source_manifest" \
  --out "$output/liquid_dataset" \
  --target-kedf lkt \
  --phase liquid \
  > "$output/filter.stdout"

labels=(combined30 liquid30 liquid60 liquid60_force)
datasets=(
  "$source_frames"
  "$output/liquid_dataset/frames.jsonl"
  "$output/liquid_dataset/frames.jsonl"
  "$output/liquid_dataset/frames.jsonl"
)
max_per_phase=(30 30 60 60)
basis_counts=(17 17 17 25)
sigmas=(0.3 0.3 0.3 0.25)
force_scales=(0.2 0.2 0.2 0.1)
ridges=(1e-10 1e-10 1e-10 1e-9)

declare -a pids=()
for index in "${!labels[@]}"; do
  label=${labels[$index]}
  candidate=$output/$label
  mkdir -p "$candidate"
  (
    taskset -c "${cpus[$index]}" env CUDA_VISIBLE_DEVICES= PYTHONPATH="$repo" \
      "$torch_runner" "$repo/scripts/fit_pair_reference.py" \
      --dataset "${datasets[$index]}" \
      --out "$candidate/model.json" \
      --max-per-phase "${max_per_phase[$index]}" \
      --basis-count "${basis_counts[$index]}" \
      --sigma "${sigmas[$index]}" \
      --force-scale "${force_scales[$index]}" \
      --ridge "${ridges[$index]}" \
      --threads 1
    frame_index=$(
      python3 - "${datasets[$index]}" <<'PY'
import json
import sys

rows = [
    json.loads(line)
    for line in open(sys.argv[1], encoding="utf-8")
    if line.strip()
]
indices = [index for index, row in enumerate(rows) if row.get("phase") == "liquid"]
if not indices:
    raise SystemExit("candidate dataset has no liquid frame")
print(indices[-1])
PY
    )
    taskset -c "${cpus[$index]}" env CUDA_VISIBLE_DEVICES= PYTHONPATH="$repo" \
      "$torch_runner" "$repo/scripts/run_pair_reference_md.py" \
      --model "$candidate/model.json" \
      --out "$candidate/liquid_md_steps${steps}" \
      --dataset-frame "${datasets[$index]}" \
      --frame-index "$frame_index" \
      --temperature "$temperature" \
      --steps "$steps" \
      --sample-every 10 \
      --seed "$((20260730 + index))" \
      --threads 1 \
      --device cpu \
      --store-positions
    env PYTHONPATH="$repo" python3 "$repo/scripts/analyze_pair_reference_md.py" \
      "$candidate/liquid_md_steps${steps}" \
      --expected liquid \
      --out "$candidate/liquid_md_steps${steps}/phase_analysis.json"
  ) > "$candidate/run.stdout" 2>&1 &
  pids[$index]=$!
  printf '%s launched label=%s cpu=%s pid=%s\n' \
    "$(date -Iseconds)" "$label" "${cpus[$index]}" "${pids[$index]}"
done

failures=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    failures=$((failures + 1))
  fi
done
[[ $failures -eq 0 ]] || {
  echo "$failures candidate jobs failed to execute" >&2
  exit 2
}

python3 - "$output" "$steps" "${labels[@]}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
steps = int(sys.argv[2])
labels = sys.argv[3:]
candidates = []
for label in labels:
    model_path = root / label / "model.json"
    run = root / label / f"liquid_md_steps{steps}"
    model = json.loads(model_path.read_text())
    summary = json.loads((run / "summary.json").read_text())
    phase = json.loads((run / "phase_analysis.json").read_text())
    checks = {
        "static_reference_gate": model.get("reference_gate_passed") is True,
        "stable": summary.get("stable") is True,
        "nearest_neighbor_gt_2_A": (
            float(summary["minimum_distance_angstrom"]) > 2.0
        ),
        "temperature_mean_within_25_K": (
            abs(float(summary["temperature_mean_k"]) - 900.0) <= 25.0
        ),
        "liquid_verified": phase.get("status") == "liquid_verified",
    }
    candidates.append(
        {
            "label": label,
            "eligible": all(checks.values()),
            "checks": checks,
            "model": str(model_path.resolve()),
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "validation": model["validation"],
            "minimum_distance_angstrom": summary["minimum_distance_angstrom"],
            "temperature_mean_k": summary["temperature_mean_k"],
            "msd_last_angstrom2": summary["msd_last_angstrom2"],
            "late_msd_slope_angstrom2_per_step": phase["MSD_A2"][
                "late_slope_per_step"
            ],
            "final_csp": phase["CSP_final"],
        }
    )
eligible = [item for item in candidates if item["eligible"]]
eligible.sort(
    key=lambda item: (
        -item["late_msd_slope_angstrom2_per_step"],
        item["validation"]["force_rmse_ev_per_angstrom"],
    )
)
payload = {
    "schema": "lkt-liquid-pair-candidate-scan-v1",
    "status": "verified" if eligible else "no_verified_candidate",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "steps": steps,
    "candidates": candidates,
    "recommended_candidate": eligible[0] if eligible else None,
}
(root / "candidate_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
if not eligible:
    raise SystemExit("no LKT liquid pair candidate passed all gates")
PY

find "$output" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$output/SHA256SUMS"
printf '%s verified\n' "$(date -Iseconds)" | tee "$done_file"
