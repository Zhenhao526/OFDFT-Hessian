#!/usr/bin/env bash
set -Eeuo pipefail

root=${QM9_NODE02_ROOT:-/home/shenwei01/xzh_node02_20260724}
repo=${QM9_NODE02_REPO:-${root}/work/structures25_autodiff_v8_20260805}
family=${root}/runs/qm9_graphformer_egfh_torch_autograd_joint_egfh_v10
launcher=${repo}/scripts/launch_qm9_graphformer_implicit_relaxed_hvp_pilot_node02.sh
activation=${repo}/scripts/activate_qm9_node_local.sh
python_bin=${repo}/.venv/bin/python
protocol=${repo}/configs/audit/qm9_graphformer_egfh_torch_autograd_joint_egfh_v10_sparse_eval50.yaml
source_checkpoint=${family}/node05_joint_v10_resume_s50to90_20260810a/checkpoints/step_0000080.ckpt
migrated_checkpoint=${family}/node05_joint_v10_sparse_eval50_migration_20260811a/step_0000080.ckpt
calibration_dir=${family}/node05_joint_v10_sparse_eval50_calibration_20260811a
calibration=${calibration_dir}/calibration.json
output=${family}/node05_joint_v10_sparse_eval50_resume_s80to300_20260811a
monitor_dir=${family}/step300_sparse_monitor_20260811a
status_file=${monitor_dir}/status.json
log_file=${monitor_dir}/orchestrator.log
gpu=${QM9_IMPLICIT_HVP_GPU:-1}
current_step=80

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
    "full_hessian_interval": 50,
    "checkpoint_interval": 5,
    "detail": detail,
}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

validate_final() {
  env PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${python_bin}" - "${output}" <<'PY'
import csv
import json
from pathlib import Path
import sys
import torch

import mldft.utils.local_frames  # noqa: F401

output = Path(sys.argv[1])
summary = json.loads((output / "summary.json").read_text())
if int(summary.get("final_step", -1)) != 300:
    raise SystemExit("unexpected final step")
checkpoint = output / "checkpoints/step_0000300.ckpt"
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
state = payload.get("complete_total_capacity")
if not isinstance(state, dict) or int(state.get("step", -1)) != 300:
    raise SystemExit("checkpoint step mismatch")
if state.get("alternating_hvp_updates") is not False:
    raise SystemExit("checkpoint is not joint E/G/F/H")
if state.get("validation_accessed") is not False:
    raise SystemExit("Validation was accessed")
if state.get("test100_accessed") is not False:
    raise SystemExit("Test100 was accessed")
rows = list(csv.DictReader((output / "full_hessian_metrics.csv").open()))
if not any(int(row["step"]) == 300 for row in rows):
    raise SystemExit("step-300 complete Hessian row is missing")
PY
}

main() {
  local protocol_sha source_sha migrated_sha calibration_sha
  protocol_sha=$(sha256sum "${protocol}" | awk '{print $1}')
  write_status preparing 80 "migrating step-80 checkpoint to sparse-evaluation schedule"

  if [[ ! -s "${migrated_checkpoint}" ]]; then
    env PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${python_bin}" "${repo}/scripts/migrate_qm9_v10_schedule_checkpoint.py" \
      --source "${source_checkpoint}" \
      --destination "${migrated_checkpoint}" \
      --protocol "${protocol}" \
      --expected-step 80
  fi
  migrated_sha=$(sha256sum "${migrated_checkpoint}" | awk '{print $1}')

  write_status calibrating 80 "running zero-LR calibration for the sparse-evaluation protocol SHA"
  if [[ ! -s "${calibration}" ]]; then
    env \
      QM9_NODE02_REPO="${repo}" \
      QM9_CODE_ROOT="${repo}" \
      QM9_NODE_ACTIVATION_SCRIPT="${activation}" \
      QM9_IMPLICIT_HVP_PROTOCOL="${protocol}" \
      QM9_IMPLICIT_HVP_PHASE=torch_v10_joint_calibration \
      QM9_IMPLICIT_HVP_GPU="${gpu}" \
      QM9_IMPLICIT_HVP_OUTPUT="${calibration_dir}" \
      bash "${launcher}" >"${calibration_dir}.pipeline.log" 2>&1
  fi
  calibration_sha=$(sha256sum "${calibration}" | awk '{print $1}')

  "${python_bin}" - "${calibration}" "${protocol_sha}" <<'PY'
import json
import math
import sys

payload = json.load(open(sys.argv[1]))
if payload.get("protocol_sha256") != sys.argv[2]:
    raise SystemExit("calibration protocol SHA mismatch")
if not math.isclose(float(payload["formal_lambda_h"]), 0.2647199267052535,
                    rel_tol=0.0, abs_tol=1.0e-12):
    raise SystemExit("schedule-only calibration changed lambda_H")
PY

  if [[ -e "${output}" || -e "${output}.pipeline.log" ]]; then
    write_status failed 80 "refusing to reuse an incomplete sparse-evaluation output"
    return 1
  fi

  write_status training 80 "joint E/G/F/H training to step 300; full Hessian every 50 steps"
  env \
    QM9_NODE02_REPO="${repo}" \
    QM9_CODE_ROOT="${repo}" \
    QM9_NODE_ACTIVATION_SCRIPT="${activation}" \
    QM9_IMPLICIT_HVP_PROTOCOL="${protocol}" \
    QM9_IMPLICIT_HVP_PHASE=torch_v10_joint_resume \
    QM9_IMPLICIT_HVP_GPU="${gpu}" \
    QM9_IMPLICIT_HVP_CALIBRATION="${calibration}" \
    QM9_IMPLICIT_HVP_CALIBRATION_SHA256="${calibration_sha}" \
    QM9_IMPLICIT_HVP_RESUME_CHECKPOINT="${migrated_checkpoint}" \
    QM9_IMPLICIT_HVP_RESUME_CHECKPOINT_SHA256="${migrated_sha}" \
    QM9_IMPLICIT_HVP_RESUME_ADDITIONAL_STEPS=220 \
    QM9_IMPLICIT_HVP_EVAL_INTERVAL=50 \
    QM9_IMPLICIT_HVP_OUTPUT="${output}" \
    bash "${launcher}" >"${output}.pipeline.log" 2>&1

  current_step=300
  validate_final
  write_status complete 300 "step-300 Hessian/checkpoint validated; sparse schedule finished"
}

handle_error() {
  local exit_code=$?
  local line=${BASH_LINENO[0]:-unknown}
  trap - ERR
  write_status failed "${current_step}" \
    "sparse-evaluation orchestrator failed at line ${line} with exit code ${exit_code}" || true
  exit "${exit_code}"
}

trap handle_error ERR
main "$@"
