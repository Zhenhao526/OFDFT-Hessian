#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/home/shenwei01/WT_Al_melting_workspace_20260724/repository}
run_root=${RUN_ROOT:-/home/shenwei01/WT_Al_melting_workspace_20260724/runs/xwm_lkt_20260727}
torch_runner=${TORCH_RUNNER:-$repo/remote_staging/run_node05_cached_cuda_torch.sh}
steps=${STEPS:-5000}
temperature=${TEMPERATURE:-900}
cpu_list=${CPU_LIST:-36,37,74,75}
orchestration_label=${ORCHESTRATION_LABEL:-classical_reference_preequil_T0900_node05_v2}
orchestration_root=$run_root/$orchestration_label
log=$orchestration_root/pipeline.log
done_file=$orchestration_root/pipeline.done
failed_file=$orchestration_root/pipeline.failed

IFS=, read -r -a cpus <<< "$cpu_list"
methods=(xwm xwm lkt lkt)
phases=(solid liquid solid liquid)

[[ ${#cpus[@]} -eq 4 ]] || {
  echo "CPU_LIST must contain exactly four CPUs" >&2
  exit 2
}
[[ -x $torch_runner ]] || {
  echo "missing torch runner: $torch_runner" >&2
  exit 2
}
[[ ! -e $done_file && ! -e $failed_file ]] || {
  echo "refusing existing completion marker under $orchestration_root" >&2
  exit 2
}

mkdir -p "$orchestration_root"
printf '%s preflight_started steps=%s cpus=%s\n' \
  "$(date -Iseconds)" "$steps" "$cpu_list" > "$log"
exec > >(tee -a "$log") 2>&1

mark_failed() {
  local rc=$?
  trap - ERR
  if [[ ! -e $done_file ]]; then
    printf '%s failed exit_code=%s\n' "$(date -Iseconds)" "$rc" \
      | tee -a "$log" "$failed_file"
  fi
  exit "$rc"
}
trap mark_failed ERR

declare -a frame_indices=()
for index in "${!methods[@]}"; do
  method=${methods[$index]}
  phase=${phases[$index]}
  fe_root=$run_root/$method/free_energy_T0900
  dataset=$fe_root/reference_dataset_v1/frames.jsonl
  dataset_manifest=$fe_root/reference_dataset_v1/manifest.json
  pair_model=$fe_root/pair_reference_v1.json
  [[ -f $dataset && -f $dataset_manifest && -f $pair_model ]] || {
    echo "missing reference inputs for $method" >&2
    exit 2
  }
  frame_indices[$index]=$(
    python3 - "$dataset" "$dataset_manifest" "$pair_model" "$method" "$phase" <<'PY'
import json
import sys

dataset_path, manifest_path, model_path, expected_method, phase = sys.argv[1:]
model = json.load(open(model_path, encoding="utf-8"))
if not model.get("reference_gate_passed"):
    raise SystemExit("pair reference has not passed its gate")
if model.get("target_kedf") != expected_method:
    raise SystemExit("pair-reference target_kedf mismatch")
manifest = json.load(open(manifest_path, encoding="utf-8"))
if manifest.get("target_kedf") != expected_method:
    raise SystemExit("dataset-manifest target_kedf mismatch")
indices = []
with open(dataset_path, encoding="utf-8") as stream:
    for index, line in enumerate(stream):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("phase") == phase:
            indices.append(index)
if not indices:
    raise SystemExit(f"dataset has no {phase} frames")
print(indices[-1])
PY
  )
done

python3 - "$orchestration_root/preflight.json" "$run_root" "$steps" \
  "$temperature" "$cpu_list" "${frame_indices[@]}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])
root = Path(sys.argv[2])
steps = int(sys.argv[3])
temperature = float(sys.argv[4])
cpus = [int(value) for value in sys.argv[5].split(",")]
frame_indices = [int(value) for value in sys.argv[6:]]
methods = ("xwm", "xwm", "lkt", "lkt")
phases = ("solid", "liquid", "solid", "liquid")
jobs = []
for method, phase, cpu, frame_index in zip(
    methods, phases, cpus, frame_indices
):
    fe_root = root / method / "free_energy_T0900"
    inputs = {
        "pair_model": fe_root / "pair_reference_v1.json",
        "dataset": fe_root / "reference_dataset_v1" / "frames.jsonl",
    }
    jobs.append(
        {
            "method": method,
            "phase": phase,
            "cpu": cpu,
            "frame_index": frame_index,
            "inputs": {
                label: {
                    "path": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for label, path in inputs.items()
            },
        }
    )
payload = {
    "schema": "kedf-classical-reference-preequil-preflight-v1",
    "status": "verified",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "temperature_k": temperature,
    "steps": steps,
    "jobs": jobs,
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

printf '%s starting steps=%s cpus=%s\n' \
  "$(date -Iseconds)" "$steps" "$cpu_list"

declare -a pids=()
declare -a outputs=()
for index in "${!methods[@]}"; do
  method=${methods[$index]}
  phase=${phases[$index]}
  cpu=${cpus[$index]}
  frame_index=${frame_indices[$index]}
  fe_root=$run_root/$method/free_energy_T0900
  out=$fe_root/classical_reference_T0900_v1/pair_preequil_steps${steps}/$phase
  [[ ! -e $out ]] || {
    echo "refusing existing output: $out" >&2
    exit 2
  }
  mkdir -p "$(dirname "$out")"
  outputs[$index]=$out
  (
    cd "$repo"
    exec taskset -c "$cpu" env CUDA_VISIBLE_DEVICES= "$torch_runner" \
      scripts/run_pair_reference_md.py \
      --model "$fe_root/pair_reference_v1.json" \
      --out "$out" \
      --dataset-frame "$fe_root/reference_dataset_v1/frames.jsonl" \
      --frame-index "$frame_index" \
      --temperature "$temperature" \
      --steps "$steps" \
      --sample-every 10 \
      --threads 1 \
      --device cpu \
      --store-positions
  ) > "$orchestration_root/${method}_${phase}.stdout" 2>&1 &
  pids[$index]=$!
  printf '%s launched method=%s phase=%s cpu=%s pid=%s frame_index=%s\n' \
    "$(date -Iseconds)" "$method" "$phase" "$cpu" "${pids[$index]}" \
    "$frame_index" | tee -a "$log"
done

failures=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    failures=$((failures + 1))
  fi
done
[[ $failures -eq 0 ]] || {
  echo "$failures pair-reference pre-equilibration jobs failed" >&2
  exit 2
}

python3 - "$temperature" "$steps" "${outputs[@]}" <<'PY'
import json
import sys
from pathlib import Path

temperature = float(sys.argv[1])
steps = int(sys.argv[2])
phases = ("solid", "liquid", "solid", "liquid")
for phase, root_value in zip(phases, sys.argv[3:]):
    root = Path(root_value)
    summary = json.loads((root / "summary.json").read_text())
    checks = {
        "natoms": int(summary["natoms"]) == 108,
        "steps": int(summary["steps"]) == steps,
        "stable": summary.get("stable") is True,
        "minimum_distance": float(summary["minimum_distance_angstrom"]) >= 2.0,
        "temperature": abs(float(summary["temperature_mean_k"]) - temperature)
        <= 25.0,
        "phase_motion": (
            float(summary["msd_last_angstrom2"]) < 1.0
            if phase == "solid"
            else float(summary["msd_last_angstrom2"]) >= 1.0
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit(f"{root}: failed gates {failed}")
PY

declare -a einstein_pids=()
for method_index in 0 2; do
  method=${methods[$method_index]}
  cpu=${cpus[$method_index]}
  solid_root=${outputs[$method_index]}
  output=$run_root/$method/free_energy_T0900/classical_reference_T0900_v1/einstein_reference.json
  [[ ! -e $output ]] || {
    echo "refusing existing Einstein reference: $output" >&2
    exit 2
  }
  (
    cd "$repo"
    exec taskset -c "$cpu" env CUDA_VISIBLE_DEVICES= "$torch_runner" \
      scripts/build_einstein_reference.py \
      --trajectory "$solid_root/trajectory.jsonl" \
      --checkpoint "$solid_root/checkpoint.json" \
      --out "$output" \
      --temperature "$temperature" \
      --discard-fraction 0.25
  ) > "$orchestration_root/${method}_einstein.stdout" 2>&1 &
  einstein_pids+=("$!")
done
for pid in "${einstein_pids[@]}"; do
  wait "$pid"
done

find "$orchestration_root" \
  "$run_root/xwm/free_energy_T0900/classical_reference_T0900_v1" \
  "$run_root/lkt/free_energy_T0900/classical_reference_T0900_v1" \
  -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$orchestration_root/SHA256SUMS"
printf '%s verified\n' "$(date -Iseconds)" | tee -a "$log" "$done_file"
