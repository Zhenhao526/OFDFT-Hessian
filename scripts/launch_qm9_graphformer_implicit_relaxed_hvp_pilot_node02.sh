#!/usr/bin/env bash
set -euo pipefail

root=${QM9_NODE02_ROOT:-/home/shenwei01/xzh_node02_20260724}
repo=${QM9_NODE02_REPO:-${root}/work/structures25}
phase=${QM9_IMPLICIT_HVP_PHASE:-smoke}
default_protocol=${repo}/configs/audit/qm9_graphformer_egfh_implicit_relaxed_hvp_pilot_v1.yaml
if [[ "${phase}" == "trust_screen" || "${phase}" == "trust_formal" || "${phase}" == "trust_resume_formal" ]]; then
  default_protocol=${repo}/configs/audit/qm9_graphformer_egfh_implicit_relaxed_hvp_pilot_v3.yaml
fi
if [[ "${phase}" == "canonical_v4_smoke" || "${phase}" == "canonical_v4_calibration" || "${phase}" == "canonical_v4_formal" || "${phase}" == "canonical_v4_resume" ]]; then
  default_protocol=${repo}/configs/audit/qm9_graphformer_egfh_implicit_relaxed_hvp_pilot_v4.yaml
fi
if [[ "${phase}" == "canonical_v5_calibration" || "${phase}" == "canonical_v5_formal" ]]; then
  default_protocol=${repo}/configs/audit/qm9_graphformer_egfh_implicit_relaxed_hvp_small_multidirection_v5.yaml
fi
if [[ "${phase}" == "autodiff_v8_calibration" || "${phase}" == "autodiff_v8_formal" ]]; then
  default_protocol=${repo}/configs/audit/qm9_graphformer_egfh_autodiff_relaxed_hvp_v8.yaml
fi
if [[ "${phase}" == "torch_v9_calibration" || "${phase}" == "torch_v9_formal" ]]; then
  default_protocol=${repo}/configs/audit/qm9_graphformer_egfh_torch_autograd_relaxed_hvp_v9.yaml
fi
if [[ "${phase}" == torch_v10_joint_* ]]; then
  default_protocol=${repo}/configs/audit/qm9_graphformer_egfh_torch_autograd_joint_egfh_v10.yaml
fi
protocol=${QM9_IMPLICIT_HVP_PROTOCOL:-${default_protocol}}
expected_canonical_protocol_id=
if [[ "${phase}" == canonical_v4_* ]]; then
  expected_canonical_protocol_id=qm9_graphformer_egfh_implicit_relaxed_hvp_pilot_v4
elif [[ "${phase}" == canonical_v5_* ]]; then
  expected_canonical_protocol_id=qm9_graphformer_egfh_implicit_relaxed_hvp_small_multidirection_v5
elif [[ "${phase}" == autodiff_v8_* ]]; then
  expected_canonical_protocol_id=qm9_graphformer_egfh_autodiff_relaxed_hvp_v8
elif [[ "${phase}" == torch_v9_* ]]; then
  expected_canonical_protocol_id=qm9_graphformer_egfh_torch_autograd_relaxed_hvp_v9
elif [[ "${phase}" == torch_v10_joint_* ]]; then
  expected_canonical_protocol_id=qm9_graphformer_egfh_torch_autograd_joint_egfh_v10
fi
if [[ -n "${expected_canonical_protocol_id}" ]]; then
  protocol_id=$(awk '$1 == "protocol_id:" {print $2; exit}' "${protocol}")
  physical_definition_id=$(awk '$1 == "physical_definition_id:" {print $2; exit}' "${protocol}")
  if [[ "${protocol_id}" != "${expected_canonical_protocol_id}" || "${physical_definition_id}" != "qm9_complete_total_relaxed_egfh_v1" ]]; then
    echo "Canonical phase requires its registered protocol and physical definition." >&2
    exit 3
  fi
fi
manifest=${QM9_IMPLICIT_HVP_MANIFEST:-${root}/artifacts/graphformer_hybrid_relaxed_hvp_rebuild_v1/manifests/train20_parent_manifest.json}
direction_manifest=${QM9_IMPLICIT_HVP_DIRECTION_MANIFEST:-${root}/artifacts/graphformer_hybrid_relaxed_hvp_rebuild_v1/directions/train20/manifest.json}
manifest_sha256=4d2c8841d550951eb518021009e12c2b6a29862f28afb0d07d32fd4da8484d01
direction_manifest_sha256=1ea9fe04a971b5998620d49f5e68567938ab4f505f17636fb2065085b8c73804
molecule_id=0016298
directions_per_step=1
integral_cache_entries=48
direction_strategy_args=()
if [[ "${phase}" == canonical_v5_* || "${phase}" == autodiff_v8_* || "${phase}" == torch_v9_* || "${phase}" == torch_v10_joint_* ]]; then
  manifest=${root}/artifacts/qm9_graphformer_small_molecule_assets_v1/0003374/parent_manifest.json
  direction_manifest=${root}/artifacts/qm9_graphformer_small_molecule_assets_v1/0003374/directions/manifest.json
  manifest_sha256=5eea79a72d900a99371b9bd9d334e4549f69b6035e1680244e980c650aeba0ef
  direction_manifest_sha256=12853f299f3b0f7292ccf4e86ddc167e503df73416c5137e8078de3f4f5feafe
  molecule_id=0003374
  directions_per_step=1
  integral_cache_entries=4
  if [[ "${phase}" == canonical_v5_* ]]; then
    directions_per_step=3
    direction_strategy_args=(
      --direction-strategy cyclic_orthogonal_internal_basis_blocks
    )
  fi
fi
article_checkpoint=${root}/runtime_parent/_runtime/models/train/runs/trained-on-qm9/checkpoints/last.ckpt
article_run_dir=${root}/models/train/runs/qm9_graphformer_egfh10_force_secant_article_warmstart_s100_v1
run_spec="released_article_qm9_scalar_graphformer=${article_run_dir}=${article_checkpoint}"
source_checkpoint_sha=9759da26660c619de9c3bbf4c2dc164343ee90e08e22b3fcdcc9682dacb9bd09
output_family=qm9_graphformer_egfh_implicit_relaxed_hvp_pilot_v1
if [[ "${phase}" == "trust_screen" || "${phase}" == "trust_formal" || "${phase}" == "trust_resume_formal" ]]; then
  output_family=qm9_graphformer_egfh_implicit_relaxed_hvp_pilot_v3
fi
if [[ "${phase}" == "canonical_v4_smoke" || "${phase}" == "canonical_v4_calibration" || "${phase}" == "canonical_v4_formal" || "${phase}" == "canonical_v4_resume" ]]; then
  output_family=qm9_graphformer_egfh_implicit_relaxed_hvp_pilot_v4
fi
if [[ "${phase}" == canonical_v5_* ]]; then
  output_family=qm9_graphformer_egfh_implicit_relaxed_hvp_small_multidirection_v5
fi
if [[ "${phase}" == autodiff_v8_* ]]; then
  output_family=qm9_graphformer_egfh_autodiff_relaxed_hvp_v8
fi
if [[ "${phase}" == torch_v9_* ]]; then
  output_family=qm9_graphformer_egfh_torch_autograd_relaxed_hvp_v9
fi
if [[ "${phase}" == torch_v10_joint_* ]]; then
  output_family=qm9_graphformer_egfh_torch_autograd_joint_egfh_v10
fi
output=${QM9_IMPLICIT_HVP_OUTPUT:-${root}/runs/${output_family}/${phase}_$(date +%Y%m%d_%H%M%S)}
gradient_weight_args=(--lambda-rho 0.8)
if [[ -n "${expected_canonical_protocol_id}" ]]; then
  gradient_weight_args=(
    --lambda-g 0.8
    --center-electron-number-residual-max 1e-10
    --analytic-response-constraint-tolerance 1e-10
  )
fi

if [[ -e "${output}" ]]; then
  echo "Refusing to reuse output directory: ${output}" >&2
  exit 2
fi
mkdir -p "$(dirname "${output}")"

check_sha256() {
  local path=$1
  local expected=$2
  local actual
  actual=$(sha256sum "${path}" | awk '{print $1}')
  if [[ "${actual}" != "${expected}" ]]; then
    echo "SHA256 mismatch for ${path}: ${actual} != ${expected}" >&2
    exit 3
  fi
}

validate_canonical_calibration() {
  local artifact=$1
  local expected_protocol_sha
  expected_protocol_sha=$(sha256sum "${protocol}" | awk '{print $1}')
  python3 - "${artifact}" "${expected_protocol_sha}" "${expected_canonical_protocol_id}" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import sys

artifact = Path(sys.argv[1]).resolve()
expected_protocol_sha = sys.argv[2]
expected_protocol_id = sys.argv[3]
payload = json.loads(artifact.read_text())
if payload.get("protocol_id") != expected_protocol_id:
    raise SystemExit("canonical formal requires a same-protocol calibration artifact")
if payload.get("physical_definition_id") != "qm9_complete_total_relaxed_egfh_v1":
    raise SystemExit("canonical v4 calibration has the wrong physical definition")
if payload.get("protocol_sha256") != expected_protocol_sha:
    raise SystemExit("canonical v4 calibration protocol SHA256 does not match runtime protocol")
training_curve = payload.get("training_curve")
training_curve_sha = payload.get("training_curve_sha256")
if not training_curve or not training_curve_sha:
    raise SystemExit("canonical v4 calibration lacks a bound training curve")
training_curve_path = Path(training_curve)
if not training_curve_path.is_file():
    raise SystemExit(f"canonical v4 calibration training curve is missing: {training_curve_path}")
actual_training_curve_sha = hashlib.sha256(training_curve_path.read_bytes()).hexdigest()
if actual_training_curve_sha != training_curve_sha:
    raise SystemExit("canonical v4 calibration training-curve SHA256 mismatch")
training_summary = payload.get("training_summary")
training_summary_sha = payload.get("training_summary_sha256")
if not training_summary or not training_summary_sha:
    raise SystemExit("canonical v4 calibration lacks a bound training summary")
training_summary_path = Path(training_summary)
if not training_summary_path.is_file():
    raise SystemExit(f"canonical v4 calibration summary is missing: {training_summary_path}")
actual_training_summary_sha = hashlib.sha256(training_summary_path.read_bytes()).hexdigest()
if actual_training_summary_sha != training_summary_sha:
    raise SystemExit("canonical v4 calibration summary SHA256 mismatch")
summary = json.loads(training_summary_path.read_text())
if summary.get("physical_definition_id") != payload.get("physical_definition_id"):
    raise SystemExit("canonical v4 calibration summary physical definition mismatch")
if summary.get("protocol_id") != payload.get("protocol_id") or summary.get("protocol_sha256") != expected_protocol_sha:
    raise SystemExit("canonical v4 calibration summary protocol mismatch")
if not isinstance(payload.get("code_provenance"), dict) or summary.get("code_provenance") != payload.get("code_provenance"):
    raise SystemExit("canonical v4 calibration code provenance mismatch")
formal_lambda_h = float(payload.get("formal_lambda_h", math.nan))
if not math.isfinite(formal_lambda_h) or formal_lambda_h <= 0.0:
    raise SystemExit("canonical v4 calibration formal_lambda_h must be finite and positive")
PY
}

read_canonical_lambda_h() {
  python3 - "$1" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1]))["formal_lambda_h"])
PY
}

check_sha256 "${manifest}" "${manifest_sha256}"
check_sha256 "${direction_manifest}" "${direction_manifest_sha256}"
check_sha256 "${article_checkpoint}" 9759da26660c619de9c3bbf4c2dc164343ee90e08e22b3fcdcc9682dacb9bd09

v4_resume_checkpoint=
canonical_calibration_args=()

gpu=${QM9_IMPLICIT_HVP_GPU:-$(
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits |
    awk -F, '
      $2 + 0 < 100 && $3 + 0 < 5 && first == "" {
        gsub(/ /, "", $1)
        first = $1
      }
      END {print first}
    '
)}
if [[ -z "${gpu}" ]]; then
  echo "No idle GPU satisfies memory<100 MiB and utilization<5%." >&2
  exit 4
fi

case "${phase}" in
  smoke)
    max_steps=1
    learning_rate=0
    evaluation_mode=none
    checkpoint_interval=1
    eval_interval=1
    gradient_diagnostics_interval=1
    lambda_h=0.01
    phase_args=()
    ;;
  formal)
    calibration=${QM9_IMPLICIT_HVP_CALIBRATION:?Set QM9_IMPLICIT_HVP_CALIBRATION for formal phase}
    check_sha256 "${calibration}" "${QM9_IMPLICIT_HVP_CALIBRATION_SHA256:?Set calibration SHA256}"
    lambda_h=$(
      python3 - "${calibration}" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1]))["formal_lambda_h"])
PY
    )
    max_steps=10
    learning_rate=1e-7
    evaluation_mode=full
    checkpoint_interval=5
    eval_interval=10
    gradient_diagnostics_interval=5
    phase_args=(
      --alternating-hvp-updates
      --replay-learning-rate 1e-7
      --hvp-learning-rate 1e-7
      --replay-update-period 5
    )
    ;;
  resume_formal)
    calibration=${QM9_IMPLICIT_HVP_CALIBRATION:?Set QM9_IMPLICIT_HVP_CALIBRATION for resume}
    check_sha256 "${calibration}" "${QM9_IMPLICIT_HVP_CALIBRATION_SHA256:?Set calibration SHA256}"
    resume_checkpoint=${QM9_IMPLICIT_HVP_RESUME_CHECKPOINT:?Set the step-5 checkpoint}
    source_checkpoint_sha=${QM9_IMPLICIT_HVP_RESUME_CHECKPOINT_SHA256:?Set the step-5 checkpoint SHA256}
    check_sha256 "${resume_checkpoint}" "${source_checkpoint_sha}"
    run_spec="released_article_qm9_scalar_graphformer=${article_run_dir}=${resume_checkpoint}"
    lambda_h=$(
      python3 - "${calibration}" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1]))["formal_lambda_h"])
PY
    )
    max_steps=5
    learning_rate=1e-7
    evaluation_mode=full
    checkpoint_interval=5
    eval_interval=10
    gradient_diagnostics_interval=5
    phase_args=(
      --alternating-hvp-updates
      --replay-learning-rate 1e-7
      --hvp-learning-rate 1e-7
      --replay-update-period 5
      --resume-capacity-optimizer
      --skip-resume-initial-full-hessian
    )
    ;;
  reproduce)
    calibration=${QM9_IMPLICIT_HVP_CALIBRATION:?Set QM9_IMPLICIT_HVP_CALIBRATION for reproduce phase}
    check_sha256 "${calibration}" "${QM9_IMPLICIT_HVP_CALIBRATION_SHA256:?Set calibration SHA256}"
    lambda_h=1
    max_steps=2
    learning_rate=1e-7
    evaluation_mode=none
    checkpoint_interval=1
    eval_interval=2
    gradient_diagnostics_interval=1
    phase_args=(
      --alternating-hvp-updates
      --replay-learning-rate 1e-7
      --hvp-learning-rate 1e-7
      --replay-update-period 5
      --two-step-failure-reproduction
    )
    ;;
  trust_screen)
    lambda_h=1
    max_steps=4
    learning_rate=1e-7
    evaluation_mode=density_cost
    checkpoint_interval=1
    eval_interval=4
    gradient_diagnostics_interval=1
    phase_args=(
      --alternating-hvp-updates
      --replay-learning-rate 1e-7
      --hvp-learning-rate 1e-7
      --replay-update-period 5
      --density-predictor-damping 0
      --density-parameter-trust-region
      --density-parameter-trust-threshold 5e-6
      --density-parameter-trust-minimum-scale 0.0078125
      --density-predictor-corrector-first
      --density-predictor-corrector-threshold 5e-6
    )
    ;;
  trust_formal)
    lambda_h=1
    max_steps=10
    learning_rate=1e-7
    evaluation_mode=full
    checkpoint_interval=5
    eval_interval=10
    gradient_diagnostics_interval=5
    phase_args=(
      --alternating-hvp-updates
      --replay-learning-rate 1e-7
      --hvp-learning-rate 1e-7
      --replay-update-period 5
      --density-predictor-damping 0
      --density-parameter-trust-region
      --density-parameter-trust-threshold 5e-6
      --density-parameter-trust-minimum-scale 0.0078125
      --density-predictor-corrector-first
      --density-predictor-corrector-threshold 5e-6
    )
    ;;
  trust_resume_formal)
    resume_checkpoint=${QM9_IMPLICIT_HVP_RESUME_CHECKPOINT:?Set the v3 step-5 checkpoint}
    source_checkpoint_sha=${QM9_IMPLICIT_HVP_RESUME_CHECKPOINT_SHA256:?Set the v3 step-5 checkpoint SHA256}
    check_sha256 "${resume_checkpoint}" "${source_checkpoint_sha}"
    run_spec="released_article_qm9_scalar_graphformer=${article_run_dir}=${resume_checkpoint}"
    lambda_h=1
    max_steps=5
    learning_rate=1e-7
    evaluation_mode=full
    checkpoint_interval=5
    eval_interval=10
    gradient_diagnostics_interval=5
    phase_args=(
      --alternating-hvp-updates
      --replay-learning-rate 1e-7
      --hvp-learning-rate 1e-7
      --replay-update-period 5
      --density-predictor-damping 0
      --density-parameter-trust-region
      --density-parameter-trust-threshold 5e-6
      --density-parameter-trust-minimum-scale 0.0078125
      --density-predictor-corrector-first
      --density-predictor-corrector-threshold 5e-6
      --resume-capacity-optimizer
      --skip-resume-initial-full-hessian
    )
    ;;
  canonical_v4_smoke)
    lambda_h=0.01
    max_steps=1
    learning_rate=0
    evaluation_mode=none
    checkpoint_interval=1
    eval_interval=1
    gradient_diagnostics_interval=1
    phase_args=(
      --alternating-hvp-updates
      --replay-learning-rate 0
      --hvp-learning-rate 0
      --replay-update-period 5
      --density-predictor-damping 0
      --density-parameter-trust-region
      --density-parameter-trust-threshold 5e-6
      --density-parameter-trust-minimum-scale 0.0078125
      --density-predictor-corrector-first
      --density-predictor-corrector-threshold 5e-6
    )
    ;;
  canonical_v4_calibration)
    lambda_h=0.01
    max_steps=1
    learning_rate=0
    evaluation_mode=none
    checkpoint_interval=1
    eval_interval=1
    gradient_diagnostics_interval=1
    phase_args=(
      --alternating-hvp-updates
      --replay-learning-rate 0
      --hvp-learning-rate 0
      --replay-update-period 5
      --density-predictor-damping 0
      --density-parameter-trust-region
      --density-parameter-trust-threshold 5e-6
      --density-parameter-trust-minimum-scale 0.0078125
      --density-predictor-corrector-first
      --density-predictor-corrector-threshold 5e-6
    )
    ;;
  canonical_v4_formal)
    calibration=${QM9_IMPLICIT_HVP_CALIBRATION:?Set the canonical-v4 calibration artifact}
    check_sha256 "${calibration}" "${QM9_IMPLICIT_HVP_CALIBRATION_SHA256:?Set canonical-v4 calibration SHA256}"
    validate_canonical_calibration "${calibration}"
    canonical_calibration_args=(
      --hvp-calibration-artifact "${calibration}"
      --hvp-calibration-sha256 "${QM9_IMPLICIT_HVP_CALIBRATION_SHA256}"
    )
    lambda_h=$(read_canonical_lambda_h "${calibration}")
    max_steps=10
    learning_rate=1e-7
    evaluation_mode=full
    checkpoint_interval=5
    eval_interval=10
    gradient_diagnostics_interval=5
    phase_args=(
      --alternating-hvp-updates
      --replay-learning-rate 1e-7
      --hvp-learning-rate 1e-7
      --replay-update-period 5
      --density-predictor-damping 0
      --density-parameter-trust-region
      --density-parameter-trust-threshold 5e-6
      --density-parameter-trust-minimum-scale 0.0078125
      --density-predictor-corrector-first
      --density-predictor-corrector-threshold 5e-6
    )
    ;;
  canonical_v4_resume)
    calibration=${QM9_IMPLICIT_HVP_CALIBRATION:?Set the canonical-v4 calibration artifact}
    check_sha256 "${calibration}" "${QM9_IMPLICIT_HVP_CALIBRATION_SHA256:?Set canonical-v4 calibration SHA256}"
    validate_canonical_calibration "${calibration}"
    canonical_calibration_args=(
      --hvp-calibration-artifact "${calibration}"
      --hvp-calibration-sha256 "${QM9_IMPLICIT_HVP_CALIBRATION_SHA256}"
    )
    v4_resume_checkpoint=${QM9_IMPLICIT_HVP_RESUME_CHECKPOINT:?Set the canonical-v4 step-5 checkpoint}
    source_checkpoint_sha=${QM9_IMPLICIT_HVP_RESUME_CHECKPOINT_SHA256:?Set the canonical-v4 checkpoint SHA256}
    check_sha256 "${v4_resume_checkpoint}" "${source_checkpoint_sha}"
    run_spec="released_article_qm9_scalar_graphformer=${article_run_dir}=${v4_resume_checkpoint}"
    lambda_h=$(read_canonical_lambda_h "${calibration}")
    max_steps=5
    learning_rate=1e-7
    evaluation_mode=full
    checkpoint_interval=5
    eval_interval=10
    gradient_diagnostics_interval=5
    phase_args=(
      --alternating-hvp-updates
      --replay-learning-rate 1e-7
      --hvp-learning-rate 1e-7
      --replay-update-period 5
      --density-predictor-damping 0
      --density-parameter-trust-region
      --density-parameter-trust-threshold 5e-6
      --density-parameter-trust-minimum-scale 0.0078125
      --density-predictor-corrector-first
      --density-predictor-corrector-threshold 5e-6
      --resume-capacity-optimizer
      --skip-resume-initial-full-hessian
    )
    ;;
  canonical_v5_calibration)
    lambda_h=0.01
    max_steps=1
    learning_rate=0
    evaluation_mode=none
    checkpoint_interval=1
    eval_interval=1
    gradient_diagnostics_interval=1
    phase_args=(
      --alternating-hvp-updates
      --replay-learning-rate 0
      --hvp-learning-rate 0
      --replay-update-period 5
      --density-predictor-damping 0
      --density-parameter-trust-region
      --density-parameter-trust-threshold 5e-6
      --density-parameter-trust-minimum-scale 0.0078125
      --density-predictor-corrector-first
      --density-predictor-corrector-threshold 5e-6
    )
    ;;
  canonical_v5_formal)
    calibration=${QM9_IMPLICIT_HVP_CALIBRATION:?Set the canonical-v5 calibration artifact}
    check_sha256 "${calibration}" "${QM9_IMPLICIT_HVP_CALIBRATION_SHA256:?Set canonical-v5 calibration SHA256}"
    validate_canonical_calibration "${calibration}"
    canonical_calibration_args=(
      --hvp-calibration-artifact "${calibration}"
      --hvp-calibration-sha256 "${QM9_IMPLICIT_HVP_CALIBRATION_SHA256}"
    )
    lambda_h=$(read_canonical_lambda_h "${calibration}")
    max_steps=50
    learning_rate=1e-7
    evaluation_mode=full
    checkpoint_interval=10
    eval_interval=10
    gradient_diagnostics_interval=25
    phase_args=(
      --alternating-hvp-updates
      --replay-learning-rate 1e-7
      --hvp-learning-rate 1e-7
      --replay-update-period 5
      --density-predictor-damping 0
      --density-parameter-trust-region
      --density-parameter-trust-threshold 5e-6
      --density-parameter-trust-minimum-scale 0.0078125
      --density-predictor-corrector-first
      --density-predictor-corrector-threshold 5e-6
    )
    ;;
  autodiff_v8_calibration)
    lambda_h=0.01
    max_steps=1
    learning_rate=0
    evaluation_mode=none
    checkpoint_interval=1
    eval_interval=1
    gradient_diagnostics_interval=1
    phase_args=(
      --alternating-hvp-updates
      --replay-learning-rate 0
      --hvp-learning-rate 0
      --replay-update-period 5
      --density-predictor-damping 0
      --density-parameter-trust-region
      --density-parameter-trust-threshold 5e-6
      --density-parameter-trust-minimum-scale 0.0078125
      --density-predictor-corrector-first
      --density-predictor-corrector-threshold 5e-6
    )
    ;;
  autodiff_v8_formal)
    calibration=${QM9_IMPLICIT_HVP_CALIBRATION:?Set the autodiff-v8 calibration artifact}
    check_sha256 "${calibration}" "${QM9_IMPLICIT_HVP_CALIBRATION_SHA256:?Set autodiff-v8 calibration SHA256}"
    validate_canonical_calibration "${calibration}"
    canonical_calibration_args=(
      --hvp-calibration-artifact "${calibration}"
      --hvp-calibration-sha256 "${QM9_IMPLICIT_HVP_CALIBRATION_SHA256}"
    )
    lambda_h=$(read_canonical_lambda_h "${calibration}")
    max_steps=10
    learning_rate=1e-7
    evaluation_mode=full
    checkpoint_interval=5
    eval_interval=10
    gradient_diagnostics_interval=5
    phase_args=(
      --alternating-hvp-updates
      --replay-learning-rate 1e-7
      --hvp-learning-rate 1e-7
      --replay-update-period 5
      --density-predictor-damping 0
      --density-parameter-trust-region
      --density-parameter-trust-threshold 5e-6
      --density-parameter-trust-minimum-scale 0.0078125
      --density-predictor-corrector-first
      --density-predictor-corrector-threshold 5e-6
    )
    ;;
  torch_v9_calibration)
    lambda_h=0.01
    max_steps=1
    learning_rate=0
    evaluation_mode=none
    checkpoint_interval=1
    eval_interval=1
    gradient_diagnostics_interval=1
    phase_args=(
      --alternating-hvp-updates
      --replay-learning-rate 0
      --hvp-learning-rate 0
      --replay-update-period 5
      --density-predictor-damping 0
      --density-parameter-trust-region
      --density-parameter-trust-threshold 5e-6
      --density-parameter-trust-minimum-scale 0.0078125
      --density-predictor-corrector-first
      --density-predictor-corrector-threshold 5e-6
    )
    ;;
  torch_v9_formal)
    calibration=${QM9_IMPLICIT_HVP_CALIBRATION:?Set the torch-v9 calibration artifact}
    check_sha256 "${calibration}" "${QM9_IMPLICIT_HVP_CALIBRATION_SHA256:?Set torch-v9 calibration SHA256}"
    validate_canonical_calibration "${calibration}"
    canonical_calibration_args=(
      --hvp-calibration-artifact "${calibration}"
      --hvp-calibration-sha256 "${QM9_IMPLICIT_HVP_CALIBRATION_SHA256}"
    )
    lambda_h=$(read_canonical_lambda_h "${calibration}")
    max_steps=10
    learning_rate=1e-7
    evaluation_mode=full
    checkpoint_interval=5
    eval_interval=10
    gradient_diagnostics_interval=5
    phase_args=(
      --alternating-hvp-updates
      --replay-learning-rate 1e-7
      --hvp-learning-rate 1e-7
      --replay-update-period 5
      --density-predictor-damping 0
      --density-parameter-trust-region
      --density-parameter-trust-threshold 5e-6
      --density-parameter-trust-minimum-scale 0.0078125
      --density-predictor-corrector-first
      --density-predictor-corrector-threshold 5e-6
    )
    ;;
  torch_v10_joint_calibration)
    lambda_h=0.01
    max_steps=1
    learning_rate=0
    evaluation_mode=none
    checkpoint_interval=1
    eval_interval=1
    gradient_diagnostics_interval=1
    phase_args=(
      --density-predictor-damping 0
      --density-parameter-trust-region
      --density-parameter-trust-threshold 5e-6
      --density-parameter-trust-minimum-scale 0.0078125
      --density-predictor-corrector-first
      --density-predictor-corrector-threshold 5e-6
    )
    ;;
  torch_v10_joint_formal)
    calibration=${QM9_IMPLICIT_HVP_CALIBRATION:?Set the torch-v10 joint calibration artifact}
    check_sha256 "${calibration}" "${QM9_IMPLICIT_HVP_CALIBRATION_SHA256:?Set torch-v10 joint calibration SHA256}"
    validate_canonical_calibration "${calibration}"
    canonical_calibration_args=(
      --hvp-calibration-artifact "${calibration}"
      --hvp-calibration-sha256 "${QM9_IMPLICIT_HVP_CALIBRATION_SHA256}"
    )
    lambda_h=$(read_canonical_lambda_h "${calibration}")
    max_steps=10
    learning_rate=1e-7
    evaluation_mode=full
    checkpoint_interval=5
    eval_interval=10
    gradient_diagnostics_interval=5
    phase_args=(
      --density-predictor-damping 0
      --density-parameter-trust-region
      --density-parameter-trust-threshold 5e-6
      --density-parameter-trust-minimum-scale 0.0078125
      --density-predictor-corrector-first
      --density-predictor-corrector-threshold 5e-6
    )
    ;;
  torch_v10_joint_resume)
    calibration=${QM9_IMPLICIT_HVP_CALIBRATION:?Set the torch-v10 joint calibration artifact}
    check_sha256 "${calibration}" "${QM9_IMPLICIT_HVP_CALIBRATION_SHA256:?Set torch-v10 joint calibration SHA256}"
    validate_canonical_calibration "${calibration}"
    canonical_calibration_args=(
      --hvp-calibration-artifact "${calibration}"
      --hvp-calibration-sha256 "${QM9_IMPLICIT_HVP_CALIBRATION_SHA256}"
    )
    resume_checkpoint=${QM9_IMPLICIT_HVP_RESUME_CHECKPOINT:?Set the torch-v10 joint step-10 checkpoint}
    source_checkpoint_sha=${QM9_IMPLICIT_HVP_RESUME_CHECKPOINT_SHA256:?Set the torch-v10 joint checkpoint SHA256}
    check_sha256 "${resume_checkpoint}" "${source_checkpoint_sha}"
    run_spec="released_article_qm9_scalar_graphformer=${article_run_dir}=${resume_checkpoint}"
    lambda_h=$(read_canonical_lambda_h "${calibration}")
    max_steps=${QM9_IMPLICIT_HVP_RESUME_ADDITIONAL_STEPS:-40}
    learning_rate=1e-7
    evaluation_mode=full
    checkpoint_interval=${QM9_IMPLICIT_HVP_CHECKPOINT_INTERVAL:-5}
    eval_interval=${QM9_IMPLICIT_HVP_EVAL_INTERVAL:-10}
    gradient_diagnostics_interval=5
    phase_args=(
      --density-predictor-damping 0
      --density-parameter-trust-region
      --density-parameter-trust-threshold 5e-6
      --density-parameter-trust-minimum-scale 0.0078125
      --density-predictor-corrector-first
      --density-predictor-corrector-threshold 5e-6
      --resume-capacity-optimizer
      --skip-resume-initial-full-hessian
    )
    ;;
  *)
    echo "Unsupported QM9_IMPLICIT_HVP_PHASE=${phase}" >&2
    exit 5
    ;;
esac

activation_script=${QM9_NODE_ACTIVATION_SCRIPT:-${repo}/scripts/activate_qm9_node02_local.sh}
source "${activation_script}"
cd "${repo}"
# The shared node02 virtualenv is an editable install of the official worktree.
# Put the selected (possibly isolated) repo first so canonical runs cannot
# silently import stale mldft modules from that editable-install target.
if [[ "${phase}" == autodiff_v8_* ]]; then
  pyscfad_overlay=${QM9_PYSCFAD_OVERLAY:-${root}/envs/pyscfad-0.3.2-overlay}
  if [[ ! -d "${pyscfad_overlay}/pyscfad" ]]; then
    echo "Missing isolated PySCFAD overlay: ${pyscfad_overlay}" >&2
    exit 6
  fi
  export PYTHONPATH="${pyscfad_overlay}:${repo}${PYTHONPATH:+:${PYTHONPATH}}"
  python3 - <<'PY'
import jax
import pyscf
import pyscfad

assert pyscf.__version__ == "2.4.0", pyscf.__version__
assert pyscfad.__version__ == "0.3.2", pyscfad.__version__
assert jax.__version__ == "0.10.2", jax.__version__
PY
  integral_backend_args=(
    --integral-derivative-backend pyscfad_autodiff
    --integral-derivative-step 0
    --integral-directional-second-step 0
  )
elif [[ "${phase}" == torch_v9_* || "${phase}" == torch_v10_joint_* ]]; then
  dqc_overlay=${MLDFT_DQC_OVERLAY:-${root}/envs/dqc-torch-integrals-0fe821fc}
  if [[ ! -f "${dqc_overlay}/PROVENANCE.txt" || ! -d "${dqc_overlay}/src/dqc/dqc" ]]; then
    echo "Missing pinned DQC Torch-integral overlay: ${dqc_overlay}" >&2
    exit 6
  fi
  export MLDFT_DQC_OVERLAY="${dqc_overlay}"
  export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
  python3 - <<'PY'
import sys
import torch
from mldft.ofdft.torch_integrals import _load_dqc_integral_modules

assert torch.__version__.split("+")[0] == "2.4.1", torch.__version__
_load_dqc_integral_modules()
assert not any(name == "jax" or name.startswith("jax.") for name in sys.modules)
PY
  integral_backend_args=(
    --integral-derivative-backend torch_autograd_dqc
    --integral-derivative-step 0
    --integral-directional-second-step 0
  )
else
  export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
  integral_backend_args=(
    --integral-derivative-backend finite_difference_pyscf
    --integral-derivative-step 1e-4
    --integral-directional-second-step 1e-4
  )
fi
export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTHONUNBUFFERED=1
default_cpu_threads=8
if [[ "${phase}" == torch_v9_* || "${phase}" == torch_v10_joint_* ]]; then
  default_cpu_threads=4
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${default_cpu_threads}}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-${default_cpu_threads}}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-${default_cpu_threads}}
export MLDFT_SYMMETRIC_MATRIX_POWER_MODE=eigh_second_order_audit

if [[ -n "${v4_resume_checkpoint}" ]]; then
  python3 - "${v4_resume_checkpoint}" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
state = payload.get("complete_total_capacity")
if not isinstance(state, dict):
    raise SystemExit("canonical v4 resume requires complete_total_capacity state")
if state.get("protocol_id") != "qm9_graphformer_egfh_implicit_relaxed_hvp_pilot_v4":
    raise SystemExit("refusing to resume a v1-v3 or foreign-protocol capacity checkpoint")
if state.get("physical_definition_id") != "qm9_complete_total_relaxed_egfh_v1":
    raise SystemExit("canonical v4 resume checkpoint has the wrong physical definition")
PY
fi

/usr/bin/time -v -o "${output}.resource.time" \
  python scripts/qm9_complete_total_capacity_train.py \
    --protocol "${protocol}" \
    --manifest "${manifest}" \
    --direction-manifest "${direction_manifest}" \
    --direction-manifest-sha256 "${direction_manifest_sha256}" \
    --direction-role all \
    --run "${run_spec}" \
    --source-checkpoint-sha256 "${source_checkpoint_sha}" \
    --root-source-checkpoint-sha256 9759da26660c619de9c3bbf4c2dc164343ee90e08e22b3fcdcc9682dacb9bd09 \
    "${canonical_calibration_args[@]}" \
    --require-complete-total-relaxed-hvp \
    --analytic-relaxed-hvp \
    --symmetric-matrix-power-mode eigh_second_order_audit \
    --egf-label-density-replay \
    --output-dir "${output}" \
    --molecules "${molecule_id}" \
    --device cuda:0 \
    --transform-device cpu \
    --seed 20260731 \
    --max-steps "${max_steps}" \
    --learning-rate "${learning_rate}" \
    --weight-decay 0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --directions-per-step "${directions_per_step}" \
    "${direction_strategy_args[@]}" \
    --evaluation-mode "${evaluation_mode}" \
    --eval-interval "${eval_interval}" \
    --checkpoint-interval "${checkpoint_interval}" \
    --early-stop-relative-frobenius 0 \
    --gradient-clip-norm 1 \
    --gradient-diagnostics-interval "${gradient_diagnostics_interval}" \
    --lambda-e 0.1 \
    "${gradient_weight_args[@]}" \
    --lambda-f 1 \
    --lambda-h "${lambda_h}" \
    --lambda-q 0 \
    --lambda-spec 0 \
    --density-loss-scale 1e-2 \
    --strict-active-density-refresh \
    --density-response-unroll-steps 0 \
    --density-parameter-response-predictor \
    --connect-lagrange-multiplier-response \
    --implicit-density-parameter-response \
    --implicit-response-tolerance 3e-5 \
    --implicit-response-max-iterations 2500 \
    --implicit-response-damping 1e-8 \
    --implicit-response-diagonal-probes 0 \
    --implicit-response-solver direct \
    --implicit-response-warm-start \
    --base-initialization label_reference \
    --density-lr 1e-3 \
    --density-max-cycles 1000 \
    --density-first-threshold 1e-2 \
    --density-fallback-lr 3e-4 \
    --density-fallback-max-cycles 10000 \
    --density-fallback-threshold 1e-5 \
    --density-strict-threshold 5e-9 \
    --training-density-stationarity-threshold 1e-8 \
    --lbfgs-max-iterations 500 \
    --newton-max-iterations 6 \
    "${integral_backend_args[@]}" \
    --integral-derivative-workers 8 \
    --integral-cache-entries "${integral_cache_entries}" \
    --analytic-response-damping 0 \
    --analytic-response-residual-tolerance 1e-8 \
    --response-correction-fraction-max 5 \
    --cancellation-index-max 10 \
    "${phase_args[@]}" \
    2>&1 | tee "${output}.console.log"

if [[ "${phase}" == "smoke" || "${phase}" == "canonical_v4_calibration" || "${phase}" == "canonical_v5_calibration" || "${phase}" == "autodiff_v8_calibration" || "${phase}" == "torch_v9_calibration" || "${phase}" == "torch_v10_joint_calibration" ]]; then
  python scripts/calibrate_qm9_implicit_hvp_weight.py \
    --protocol "${protocol}" \
    --training-curve "${output}/training_curve.csv" \
    --output "${output}/calibration.json"
  if [[ "${phase}" == "canonical_v4_calibration" || "${phase}" == "canonical_v5_calibration" || "${phase}" == "autodiff_v8_calibration" || "${phase}" == "torch_v9_calibration" || "${phase}" == "torch_v10_joint_calibration" ]]; then
    validate_canonical_calibration "${output}/calibration.json"
  fi
elif [[ "${phase}" == "formal" || "${phase}" == "canonical_v4_formal" || "${phase}" == "canonical_v4_resume" || "${phase}" == "canonical_v5_formal" || "${phase}" == "autodiff_v8_formal" || "${phase}" == "torch_v9_formal" || "${phase}" == "torch_v10_joint_formal" || "${phase}" == "torch_v10_joint_resume" ]]; then
  cp "${calibration}" "${output}/source_calibration.json"
fi

echo "implicit_hvp_output=${output}"
