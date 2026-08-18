#!/usr/bin/env bash
set -Eeuo pipefail

root=${QM9_NODE02_ROOT:-/home/shenwei01/xzh_node02_20260724}
repo=${QM9_NODE02_REPO:-${root}/work/structures25_autodiff_v8_20260805}
family=${root}/runs/qm9_graphformer_egfh_torch_autograd_joint_egfh_v10
calibration=${family}/node01_joint_v10_calibration_20260810a/calibration.json
calibration_sha=5410ca305d62b62fb204dc61d8599aee9bf54dd5dfe77410f2e47c8d2a45e13a
launcher=${repo}/scripts/launch_qm9_graphformer_implicit_relaxed_hvp_pilot_node02.sh
activation=${repo}/scripts/activate_qm9_node_local.sh
python_bin=${repo}/.venv/bin/python
monitor_dir=${family}/step300_monitor_20260810a
status_file=${monitor_dir}/status.json
log_file=${monitor_dir}/orchestrator.log
initial_output=${family}/node01_joint_v10_resume_s10to50_20260810a
target_step=300
gpu=${QM9_IMPLICIT_HVP_GPU:-0}
execution_label=${QM9_EXECUTION_LABEL:-node01}
current_step=10

mkdir -p "${monitor_dir}"
exec >>"${log_file}" 2>&1

write_status() {
  local state=$1
  local step=$2
  local detail=$3
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
    "target_step": 300,
    "detail": detail,
}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

validate_completed_run() {
  local output=$1
  local expected_step=$2
  env PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${python_bin}" - "${output}" "${expected_step}" <<'PY'
import csv
import json
from pathlib import Path
import sys
import torch

# This pinned PyTorch build is not re-entrant. Importing local_frames for the
# first time during outer checkpoint unpickling would recursively load Jd.pt
# and clear torch.load's thread-local map_location state.
import mldft.utils.local_frames  # noqa: F401

output = Path(sys.argv[1])
expected_step = int(sys.argv[2])
summary = json.loads((output / "summary.json").read_text())
if int(summary.get("final_step", -1)) != expected_step:
    raise SystemExit(
        f"unexpected final step {summary.get('final_step')} != {expected_step}"
    )
checkpoint = output / "checkpoints" / f"step_{expected_step:07d}.ckpt"
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
state = payload.get("complete_total_capacity")
if not isinstance(state, dict) or int(state.get("step", -1)) != expected_step:
    raise SystemExit("checkpoint capacity step mismatch")
if state.get("protocol_id") != "qm9_graphformer_egfh_torch_autograd_joint_egfh_v10":
    raise SystemExit("checkpoint protocol mismatch")
if state.get("alternating_hvp_updates") is not False:
    raise SystemExit("checkpoint is not the shared-optimizer joint protocol")
rows = list(csv.DictReader((output / "full_hessian_metrics.csv").open()))
if not any(int(row["step"]) == expected_step for row in rows):
    raise SystemExit("final complete-Hessian row is missing")
PY
}

wait_for_initial_step50() {
  while true; do
    if [[ -s "${initial_output}/summary.json" ]]; then
      validate_completed_run "${initial_output}" 50
      write_status ready 50 "step-50 continuation completed and validated"
      return
    fi
    if ! pgrep -f "qm9_complete_total_capacity_train.py.*--output-dir ${initial_output}" >/dev/null; then
      write_status failed 10 "step-10-to-50 process exited without a valid summary"
      return 1
    fi
    write_status training 10 "waiting for the active step-10-to-50 continuation"
    sleep 30
  done
}

run_full_chunk() {
  local start_step=$1
  local end_step=$2
  local source_checkpoint=$3
  local output=${family}/${execution_label}_joint_v10_resume_s${start_step}to${end_step}_20260810a
  local source_sha
  current_step=${start_step}
  source_sha=$(sha256sum "${source_checkpoint}" | awk '{print $1}')

  if [[ -s "${output}/summary.json" ]]; then
    validate_completed_run "${output}" "${end_step}"
    write_status ready "${end_step}" "existing continuation chunk validated"
    printf '%s\n' "${output}/checkpoints/step_$(printf '%07d' "${end_step}").ckpt"
    return
  fi
  if [[ -e "${output}" || -e "${output}.pipeline.log" ]]; then
    write_status failed "${start_step}" "refusing to reuse an incomplete continuation output"
    return 1
  fi

  write_status training "${start_step}" "launching cumulative steps ${start_step} to ${end_step}"
  env \
    QM9_NODE02_REPO="${repo}" \
    QM9_CODE_ROOT="${repo}" \
    QM9_NODE_ACTIVATION_SCRIPT="${activation}" \
    QM9_IMPLICIT_HVP_PHASE=torch_v10_joint_resume \
    QM9_IMPLICIT_HVP_GPU="${gpu}" \
    QM9_IMPLICIT_HVP_CALIBRATION="${calibration}" \
    QM9_IMPLICIT_HVP_CALIBRATION_SHA256="${calibration_sha}" \
    QM9_IMPLICIT_HVP_RESUME_CHECKPOINT="${source_checkpoint}" \
    QM9_IMPLICIT_HVP_RESUME_CHECKPOINT_SHA256="${source_sha}" \
    QM9_IMPLICIT_HVP_OUTPUT="${output}" \
    bash "${launcher}" >"${output}.pipeline.log" 2>&1

  validate_completed_run "${output}" "${end_step}"
  write_status ready "${end_step}" "continuation chunk completed and validated"
  printf '%s\n' "${output}/checkpoints/step_$(printf '%07d' "${end_step}").ckpt"
}

run_final_partial_chunk() {
  local source_checkpoint=$1
  local output=${family}/${execution_label}_joint_v10_resume_s290to330_stop300_20260810a
  local checkpoint300=${output}/checkpoints/step_0000300.ckpt
  local metrics=${output}/full_hessian_metrics.csv
  local source_sha
  current_step=290
  source_sha=$(sha256sum "${source_checkpoint}" | awk '{print $1}')

  if [[ -e "${output}" || -e "${output}.pipeline.log" ]]; then
    write_status failed 290 "refusing to reuse the final partial-chunk output"
    return 1
  fi

  write_status training 290 "launching final registered chunk; stop after step-300 Hessian/checkpoint"
  setsid env \
    QM9_NODE02_REPO="${repo}" \
    QM9_CODE_ROOT="${repo}" \
    QM9_NODE_ACTIVATION_SCRIPT="${activation}" \
    QM9_IMPLICIT_HVP_PHASE=torch_v10_joint_resume \
    QM9_IMPLICIT_HVP_GPU="${gpu}" \
    QM9_IMPLICIT_HVP_CALIBRATION="${calibration}" \
    QM9_IMPLICIT_HVP_CALIBRATION_SHA256="${calibration_sha}" \
    QM9_IMPLICIT_HVP_RESUME_CHECKPOINT="${source_checkpoint}" \
    QM9_IMPLICIT_HVP_RESUME_CHECKPOINT_SHA256="${source_sha}" \
    QM9_IMPLICIT_HVP_OUTPUT="${output}" \
    bash "${launcher}" >"${output}.pipeline.log" 2>&1 < /dev/null &
  local launcher_pid=$!

  while true; do
    if [[ -s "${checkpoint300}" && -s "${metrics}" ]]; then
      if env PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${python_bin}" - "${checkpoint300}" "${metrics}" <<'PY'
import csv
from pathlib import Path
import sys
import torch

import mldft.utils.local_frames  # noqa: F401

checkpoint, metrics = map(Path, sys.argv[1:])
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
state = payload.get("complete_total_capacity")
if not isinstance(state, dict) or int(state.get("step", -1)) != 300:
    raise SystemExit(1)
rows = list(csv.DictReader(metrics.open()))
if not any(int(row["step"]) == 300 for row in rows):
    raise SystemExit(1)
PY
      then
        kill -TERM -- "-${launcher_pid}" 2>/dev/null || true
        wait "${launcher_pid}" || true
        write_status complete 300 "step-300 complete Hessian and restorable checkpoint validated"
        return
      fi
    fi
    if ! kill -0 "${launcher_pid}" 2>/dev/null; then
      wait "${launcher_pid}" || true
      write_status failed 290 "final chunk exited before step-300 artifacts were complete"
      return 1
    fi
    sleep 5
  done
}

main() {
  write_status monitoring 10 "step-300 monitor initialized"
  wait_for_initial_step50

  local checkpoint=${initial_output}/checkpoints/step_0000050.ckpt
  local start_step=50
  local end_step
  for end_step in 90 130 170 210 250 290; do
    checkpoint=$(run_full_chunk "${start_step}" "${end_step}" "${checkpoint}" | tail -n 1)
    start_step=${end_step}
  done
  run_final_partial_chunk "${checkpoint}"
}

handle_error() {
  local exit_code=$?
  local line=${BASH_LINENO[0]:-unknown}
  trap - ERR
  write_status failed "${current_step}" \
    "orchestrator failed at line ${line} with exit code ${exit_code}" || true
  exit "${exit_code}"
}

trap handle_error ERR
main "$@"
