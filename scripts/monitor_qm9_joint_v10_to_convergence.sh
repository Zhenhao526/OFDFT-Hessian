#!/usr/bin/env bash
set -Eeuo pipefail

root=${QM9_NODE02_ROOT:-/home/shenwei01/xzh_node02_20260724}
repo=${QM9_NODE02_REPO:-${root}/work/structures25_autodiff_v8_20260805}
family=${root}/runs/qm9_graphformer_egfh_torch_autograd_joint_egfh_v10
launcher=${repo}/scripts/launch_qm9_graphformer_implicit_relaxed_hvp_pilot_node02.sh
activation=${repo}/scripts/activate_qm9_node_local.sh
python_bin=${repo}/.venv/bin/python
protocol=${repo}/configs/audit/qm9_graphformer_egfh_torch_autograd_joint_egfh_v10_sparse_eval100_ckpt50.yaml
calibration=${family}/node05_joint_v10_sparse_eval100_ckpt50_calibration_20260811a/calibration.json
source_checkpoint=${QM9_CONVERGENCE_SOURCE_CHECKPOINT:-${family}/node05_joint_v10_sparse_eval100_ckpt50_resume_s80to300_20260811a/checkpoints/step_0000300.ckpt}
source_checkpoint=$(realpath "${source_checkpoint}")
baseline_hessian=${family}/node05_joint_v10_sparse_eval100_ckpt50_resume_s80to300_20260811a/full_hessian_metrics.csv
monitor_dir=${family}/convergence_monitor_20260812a
status_file=${monitor_dir}/status.json
segments_file=${monitor_dir}/segments.json
history_file=${monitor_dir}/convergence_history.json
log_file=${monitor_dir}/orchestrator.log
gpu=${QM9_IMPLICIT_HVP_GPU:-3}
current_step=${QM9_CONVERGENCE_CURRENT_STEP:-300}

mkdir -p "${monitor_dir}"
exec 9>"${monitor_dir}/orchestrator.lock"
if ! flock -n 9; then
  echo "another convergence orchestrator holds ${monitor_dir}/orchestrator.lock" >&2
  exit 1
fi
exec >>"${log_file}" 2>&1

write_status() {
  local state=$1 step=$2 detail=$3
  "${python_bin}" - "${status_file}" "${state}" "${step}" "${detail}" <<'PY'
import json
from datetime import datetime
from pathlib import Path
import sys

path, state, step, detail = sys.argv[1:]
payload = {
    "updated_at": datetime.now().astimezone().isoformat(),
    "state": state,
    "latest_completed_step": int(step),
    "target": "audited_convergence",
    "chunk_steps": 220,
    "checkpoint_interval": 50,
    "full_hessian_interval": 100,
    "minimum_consecutive_plateau_intervals": 3,
    "relative_frobenius_plateau_per_100_steps": 0.01,
    "hessian_mae_plateau_per_100_steps": 0.01,
    "force_mae_plateau_per_100_steps": 0.02,
    "terminal_total_loss_trend_per_100_steps": 0.02,
    "terminal_force_loss_trend_per_100_steps": 0.02,
    "energy_regression_ratio_max": 10.0,
    "detail": detail,
}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

if [[ ! -s "${segments_file}" ]]; then
  printf '[]\n' >"${segments_file}"
fi

validate_and_assess() {
  local output=$1 source_step=$2 target_step=$3
  env PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${python_bin}" - "${output}" "${source_step}" "${target_step}" \
    "${baseline_hessian}" "${segments_file}" "${history_file}" <<'PY'
import csv
import json
import math
from pathlib import Path
import sys
import torch

import mldft.utils.local_frames  # noqa: F401

output = Path(sys.argv[1])
source_step = int(sys.argv[2])
target_step = int(sys.argv[3])
baseline_hessian = Path(sys.argv[4])
segments_file = Path(sys.argv[5])
history_file = Path(sys.argv[6])

summary = json.loads((output / "summary.json").read_text())
if int(summary.get("final_step", -1)) != target_step:
    raise SystemExit("unexpected segment final step")
if summary.get("validation_accessed") is not False:
    raise SystemExit("Validation was accessed")
if summary.get("test100_accessed") is not False:
    raise SystemExit("Test100 was accessed")
checkpoint = output / "checkpoints" / f"step_{target_step:07d}.ckpt"
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
state = payload.get("complete_total_capacity")
if not isinstance(state, dict) or int(state.get("step", -1)) != target_step:
    raise SystemExit("terminal checkpoint is not restorable at expected step")
if state.get("alternating_hvp_updates") is not False:
    raise SystemExit("terminal checkpoint is not joint E/G/F/H")
if state.get("validation_accessed") is not False or state.get("test100_accessed") is not False:
    raise SystemExit("terminal checkpoint records forbidden evaluation access")

with (output / "full_hessian_metrics.csv").open(newline="") as handle:
    terminal_rows = list(csv.DictReader(handle))
if not any(int(row["step"]) == target_step for row in terminal_rows):
    raise SystemExit("terminal complete Hessian is missing")
with (output / "training_curve.csv").open(newline="") as handle:
    training_rows = list(csv.DictReader(handle))
if not training_rows or int(training_rows[-1]["step"]) != target_step:
    raise SystemExit("terminal training curve is incomplete")

segments = json.loads(segments_file.read_text())
segments.append({"path": output.as_posix(), "source_step": source_step, "target_step": target_step})
segments_file.write_text(json.dumps(segments, indent=2, sort_keys=True) + "\n")

with baseline_hessian.open(newline="") as handle:
    base_rows = list(csv.DictReader(handle))
points = [dict(row) for row in base_rows if int(row["step"]) == 300]
terminal_by_step = {}
for segment in segments:
    path = Path(segment["path"]) / "full_hessian_metrics.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        step = int(row["step"])
        terminal_by_step[step] = dict(row)
        if (step - int(segment["source_step"])) % 100 == 0:
            points.append(dict(row))
points.sort(key=lambda row: int(row["step"]))

metric_thresholds = {
    "relative_frobenius": 0.01,
    "mae": 0.01,
    "complete_total_force_mae_hartree_per_bohr": 0.02,
}
intervals = []
for previous, current in zip(points, points[1:]):
    gap = int(current["step"]) - int(previous["step"])
    record = {"from_step": int(previous["step"]), "to_step": int(current["step"]), "step_gap": gap}
    passed = True
    for metric, threshold in metric_thresholds.items():
        before = float(previous[metric])
        after = float(current[metric])
        normalized = abs(after / before - 1.0) * 100.0 / gap
        record[f"{metric}_absolute_fraction_per_100_steps"] = normalized
        record[f"{metric}_threshold"] = threshold
        passed = passed and normalized <= threshold
    record["plateau_passed"] = passed
    intervals.append(record)

latest_cadence = points[-1]
terminal = terminal_by_step[target_step]
root_initial = state.get("root_initial_full_hessian_metrics")
if not isinstance(root_initial, list) or not root_initial:
    raise SystemExit("terminal checkpoint lacks root initial metrics")
root_energy = float(root_initial[0]["total_energy_abs_error_hartree"])
energy_ratio = float(terminal["total_energy_abs_error_hartree"]) / root_energy
terminal_consistent = all(
    abs(float(terminal[metric]) / float(latest_cadence[metric]) - 1.0) <= threshold
    for metric, threshold in metric_thresholds.items()
)
tail = training_rows[-50:]
def normalized_linear_trend(rows, metric):
    pairs = [(float(row["step"]), float(row[metric])) for row in rows]
    x_mean = sum(x for x, _ in pairs) / len(pairs)
    y_mean = sum(y for _, y in pairs) / len(pairs)
    denominator = sum((x - x_mean) ** 2 for x, _ in pairs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in pairs) / denominator
    return abs(slope) * 100.0 / max(abs(y_mean), 1.0e-15)
total_loss_trend = normalized_linear_trend(tail, "total_loss")
force_loss_trend = normalized_linear_trend(tail, "loss/force")
training_trend_passed = total_loss_trend <= 0.02 and force_loss_trend <= 0.02
last_three = intervals[-3:]
converged = (
    len(last_three) == 3
    and all(item["plateau_passed"] for item in last_three)
    and energy_ratio <= 10.0
    and terminal_consistent
    and training_trend_passed
    and bool(summary.get("density_cost_gate_passed"))
    and bool(summary.get("training_density_stationarity_gate_passed"))
)
history = {
    "schema_version": 1,
    "criterion": {
        "minimum_consecutive_intervals": 3,
        "relative_frobenius_absolute_fraction_per_100_steps_max": 0.01,
        "hessian_mae_absolute_fraction_per_100_steps_max": 0.01,
        "force_mae_absolute_fraction_per_100_steps_max": 0.02,
        "terminal_energy_error_over_root_max": 10.0,
        "terminal_must_match_latest_cadence_thresholds": True,
        "terminal_total_loss_absolute_trend_per_100_steps_max": 0.02,
        "terminal_force_loss_absolute_trend_per_100_steps_max": 0.02,
        "density_gates_required": True,
    },
    "cadence_points": [
        {
            key: row[key]
            for key in (
                "step", "relative_frobenius", "mae",
                "complete_total_force_mae_hartree_per_bohr",
                "total_energy_abs_error_hartree",
            )
        }
        for row in points
    ],
    "intervals": intervals,
    "terminal_step": target_step,
    "terminal_energy_error_over_root": energy_ratio,
    "terminal_consistent_with_latest_cadence": terminal_consistent,
    "terminal_total_loss_absolute_trend_per_100_steps": total_loss_trend,
    "terminal_force_loss_absolute_trend_per_100_steps": force_loss_trend,
    "terminal_training_trend_passed": training_trend_passed,
    "converged": converged,
}
history_file.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
print(json.dumps(history, sort_keys=True))
raise SystemExit(0 if converged else 3)
PY
}

main() {
  local source_sha calibration_sha output target_step verdict
  calibration_sha=$(sha256sum "${calibration}" | awk '{print $1}')
  write_status preparing "${current_step}" "preparing same-protocol joint E/G/F/H convergence continuation"

  while true; do
    target_step=$((current_step + 220))
    output=${family}/node05_joint_v10_convergence_resume_s${current_step}to${target_step}_20260812a
    if [[ -e "${output}" || -e "${output}.pipeline.log" ]]; then
      write_status failed "${current_step}" "refusing to reuse convergence segment output ${output}"
      return 1
    fi
    source_sha=$(sha256sum "${source_checkpoint}" | awk '{print $1}')
    write_status training "${current_step}" "training segment ${current_step}->${target_step}; convergence not yet established"
    env \
      QM9_NODE02_REPO="${repo}" \
      QM9_CODE_ROOT="${repo}" \
      QM9_NODE_ACTIVATION_SCRIPT="${activation}" \
      QM9_IMPLICIT_HVP_PROTOCOL="${protocol}" \
      QM9_IMPLICIT_HVP_PHASE=torch_v10_joint_resume \
      QM9_IMPLICIT_HVP_GPU="${gpu}" \
      QM9_IMPLICIT_HVP_CALIBRATION="${calibration}" \
      QM9_IMPLICIT_HVP_CALIBRATION_SHA256="${calibration_sha}" \
      QM9_IMPLICIT_HVP_RESUME_CHECKPOINT="${source_checkpoint}" \
      QM9_IMPLICIT_HVP_RESUME_CHECKPOINT_SHA256="${source_sha}" \
      QM9_IMPLICIT_HVP_RESUME_ADDITIONAL_STEPS=220 \
      QM9_IMPLICIT_HVP_CHECKPOINT_INTERVAL=50 \
      QM9_IMPLICIT_HVP_EVAL_INTERVAL=100 \
      QM9_IMPLICIT_HVP_OUTPUT="${output}" \
      bash "${launcher}" >"${output}.pipeline.log" 2>&1

    current_step=${target_step}
    # Run the convergence assessor as a conditional command so its intentional
    # exit code 3 (valid segment, not converged yet) is exempt from both
    # `errexit` and the inherited ERR trap.
    if validate_and_assess "${output}" "$((target_step - 220))" "${target_step}"; then
      verdict=0
    else
      verdict=$?
    fi
    if [[ ${verdict} -eq 0 ]]; then
      write_status complete "${current_step}" "audited convergence criterion satisfied; terminal Hessian/checkpoint validated"
      return 0
    elif [[ ${verdict} -ne 3 ]]; then
      write_status failed "${current_step}" "segment validation or convergence audit failed"
      return "${verdict}"
    fi
    source_checkpoint=${output}/checkpoints/step_$(printf '%07d' "${target_step}").ckpt
    write_status continuing "${current_step}" "segment validated but convergence criterion not yet satisfied"
  done
}

handle_error() {
  local exit_code=$? line=${BASH_LINENO[0]:-unknown}
  trap - ERR
  write_status failed "${current_step}" "convergence orchestrator failed at line ${line} with exit code ${exit_code}" || true
  exit "${exit_code}"
}

trap handle_error ERR
main "$@"
