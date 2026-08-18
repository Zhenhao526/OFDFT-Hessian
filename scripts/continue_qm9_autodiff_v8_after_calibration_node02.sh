#!/usr/bin/env bash
set -euo pipefail

root=${QM9_NODE02_ROOT:-/home/shenwei01/xzh_node02_20260724}
repo=${QM9_NODE02_REPO:-${root}/work/structures25_autodiff_v8_20260805}
calibration_launcher_pid=${QM9_AUTODIFF_V8_CALIBRATION_LAUNCHER_PID:?Set the active calibration launcher PID}
calibration_dir=${QM9_AUTODIFF_V8_CALIBRATION_DIR:?Set the active calibration directory}
formal_output=${QM9_AUTODIFF_V8_FORMAL_OUTPUT:-${root}/runs/qm9_graphformer_egfh_autodiff_relaxed_hvp_v8/formal_0003374_s10_20260805_a}
gpu=${QM9_IMPLICIT_HVP_GPU:-0}

if [[ -e "${formal_output}" ]]; then
  echo "Refusing to reuse formal output: ${formal_output}" >&2
  exit 2
fi
if [[ ! -r "/proc/${calibration_launcher_pid}/environ" ]] || \
   ! tr '\0' '\n' < "/proc/${calibration_launcher_pid}/environ" | \
     grep -Fxq "QM9_IMPLICIT_HVP_OUTPUT=${calibration_dir}"; then
  echo "Calibration launcher PID does not own the requested calibration" >&2
  exit 3
fi

while kill -0 "${calibration_launcher_pid}" 2>/dev/null; do
  sleep 30
done

calibration=${calibration_dir}/calibration.json
summary=${calibration_dir}/summary.json
curve=${calibration_dir}/training_curve.csv
if [[ ! -s "${calibration}" || ! -s "${summary}" || ! -s "${curve}" ]]; then
  echo "Calibration did not produce a complete, nonempty artifact set" >&2
  exit 4
fi

calibration_sha256=$(sha256sum "${calibration}" | awk '{print $1}')
export QM9_NODE02_REPO="${repo}"
export QM9_IMPLICIT_HVP_PHASE=autodiff_v8_formal
export QM9_IMPLICIT_HVP_GPU="${gpu}"
export QM9_IMPLICIT_HVP_OUTPUT="${formal_output}"
export QM9_IMPLICIT_HVP_CALIBRATION="${calibration}"
export QM9_IMPLICIT_HVP_CALIBRATION_SHA256="${calibration_sha256}"

exec bash "${repo}/scripts/launch_qm9_graphformer_implicit_relaxed_hvp_pilot_node02.sh"
