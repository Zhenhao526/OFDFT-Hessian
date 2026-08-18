#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 CHECKPOINT SOURCE_SHA256 OUTPUT_DIR GPU" >&2
  exit 2
fi

checkpoint=$1
source_sha=$2
output=$3
gpu=$4
root=/home/shenwei01/xzh_node02_20260724
repo=${root}/work/structures25_autodiff_v8_20260805
protocol=${repo}/configs/audit/qm9_graphformer_egfh_torch_autograd_relaxed_hvp_v9.yaml
manifest=${root}/artifacts/qm9_graphformer_small_molecule_assets_v1/0003374/parent_manifest.json
direction_manifest=${root}/artifacts/qm9_graphformer_small_molecule_assets_v1/0003374/directions/manifest.json
article_run_dir=${root}/models/train/runs/qm9_graphformer_egfh10_force_secant_article_warmstart_s100_v1
calibration=${root}/runs/qm9_graphformer_egfh_torch_autograd_relaxed_hvp_v9/node01_calibration_torch_v9_20260805c/calibration.json
calibration_sha=8be094f55ad3a989b9122af32a3507cf42f4e00f3a8ce53204283dd453c8e691
root_source_sha=9759da26660c619de9c3bbf4c2dc164343ee90e08e22b3fcdcc9682dacb9bd09
resume_args=()
if [[ "${source_sha}" != "${root_source_sha}" ]]; then
  resume_args=(--resume-capacity-optimizer)
fi

if [[ -e "${output}" ]]; then
  echo "refusing to reuse output: ${output}" >&2
  exit 3
fi
for binding in \
  "${checkpoint}:${source_sha}" \
  "${calibration}:${calibration_sha}" \
  "${protocol}:05b277550903ed173b362a2261542225e2a51fe70d2a46b58cebae752c5970f9" \
  "${manifest}:5eea79a72d900a99371b9bd9d334e4549f69b6035e1680244e980c650aeba0ef" \
  "${direction_manifest}:12853f299f3b0f7292ccf4e86ddc167e503df73416c5137e8078de3f4f5feafe"
do
  path=${binding%%:*}
  expected=${binding##*:}
  actual=$(sha256sum "${path}" | awk '{print $1}')
  if [[ "${actual}" != "${expected}" ]]; then
    echo "SHA256 mismatch: ${path}: ${actual} != ${expected}" >&2
    exit 4
  fi
done

export QM9_NODE02_ROOT=${root}
export QM9_CODE_ROOT=${repo}
source "${repo}/scripts/activate_qm9_node_local.sh"
cd "${repo}"
export MLDFT_DQC_OVERLAY=${root}/envs/dqc-torch-integrals-0fe821fc
export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=${gpu}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MLDFT_SYMMETRIC_MATRIX_POWER_MODE=eigh_second_order_audit

mkdir -p "$(dirname "${output}")"
/usr/bin/time -v -o "${output}.resource.time" \
  python scripts/qm9_graphformer_fixed_response_hessian_audit.py \
    --protocol "${protocol}" \
    --manifest "${manifest}" \
    --direction-manifest "${direction_manifest}" \
    --direction-manifest-sha256 12853f299f3b0f7292ccf4e86ddc167e503df73416c5137e8078de3f4f5feafe \
    --direction-role all \
    --run "released_article_qm9_scalar_graphformer=${article_run_dir}=${checkpoint}" \
    --source-checkpoint-sha256 "${source_sha}" \
    --root-source-checkpoint-sha256 "${root_source_sha}" \
    --hvp-calibration-artifact "${calibration}" \
    --hvp-calibration-sha256 "${calibration_sha}" \
    --require-complete-total-relaxed-hvp \
    --analytic-relaxed-hvp \
    --symmetric-matrix-power-mode eigh_second_order_audit \
    --egf-label-density-replay \
    --output-dir "${output}" \
    --molecules 0003374 \
    --device cuda:0 \
    --transform-device cpu \
    --seed 20260731 \
    --max-steps 0 \
    --learning-rate 1e-7 \
    --weight-decay 0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --directions-per-step 1 \
    --evaluation-mode full \
    --eval-interval 10 \
    --checkpoint-interval 5 \
    --early-stop-relative-frobenius 0 \
    --gradient-clip-norm 1 \
    --gradient-diagnostics-interval 5 \
    --lambda-e 0.1 \
    --lambda-g 0.8 \
    --lambda-f 1 \
    --lambda-h 0.2647199267010593 \
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
    --integral-derivative-backend torch_autograd_dqc \
    --integral-derivative-step 0 \
    --integral-directional-second-step 0 \
    --integral-derivative-workers 8 \
    --integral-cache-entries 4 \
    --analytic-response-damping 0 \
    --analytic-response-residual-tolerance 1e-8 \
    --analytic-response-constraint-tolerance 1e-10 \
    --center-electron-number-residual-max 1e-10 \
    --response-correction-fraction-max 5 \
    --cancellation-index-max 10 \
    --alternating-hvp-updates \
    --replay-learning-rate 1e-7 \
    --hvp-learning-rate 1e-7 \
    --replay-update-period 5 \
    --density-predictor-damping 0 \
    --density-parameter-trust-region \
    --density-parameter-trust-threshold 5e-6 \
    --density-parameter-trust-minimum-scale 0.0078125 \
    --density-predictor-corrector-first \
    --density-predictor-corrector-threshold 5e-6 \
    "${resume_args[@]}" \
    2>&1 | tee "${output}.console.log"
