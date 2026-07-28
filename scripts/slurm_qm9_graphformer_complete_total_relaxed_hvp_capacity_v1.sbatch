#!/usr/bin/env bash
#SBATCH --job-name=gf_cth_fit
#SBATCH --partition=compute
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch/xzh/logs/graphformer_relaxed_hvp/%x_%j.out
#SBATCH --error=/scratch/xzh/logs/graphformer_relaxed_hvp/%x_%j.err

set -euo pipefail
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25

parent_set=${PARENT_SET:-stable5}
direction_role=${DIRECTION_ROLE:-all}
molecules=${MOLECULE_IDS:-0028399}
max_steps=${MAX_STEPS:-200}
learning_rate=${LEARNING_RATE:-1e-6}
eval_interval=${EVAL_INTERVAL:-${max_steps}}
run_tag=${RUN_TAG:-stage1_one_parent}
prerequisite_summaries=${PREREQUISITE_SUMMARIES:?set colon-separated fail-closed prerequisite summary paths}
lambda_e=${LAMBDA_E:-1}
lambda_f=${LAMBDA_F:-1}
lambda_rho=${LAMBDA_RHO:-0.1}
lambda_h=${LAMBDA_H:-1}
directions_per_step=${DIRECTIONS_PER_STEP:-4}
checkpoint_interval=${CHECKPOINT_INTERVAL:-10}

root=/scratch/xzh/models/graphformer_relaxed_hvp/20260723/v1
case "${parent_set}" in
  stable5)
    parent_manifest="${root}/manifests/stable5_parent_manifest.json"
    direction_manifest="${root}/directions/stable5/manifest.json"
    direction_manifest_sha256=1d9d235719805b3bf298fe3baf783b1b17a262e352b4193388f0be4f9b65bd74
    ;;
  train20)
    parent_manifest=/scratch/xzh/models/complete_total_capacity/20260717/stage2_direction_v1/selection_uniform_A_v1/selected_baseline_manifest.json
    direction_manifest="${root}/directions/train20/manifest.json"
    direction_manifest_sha256=a902bb554cf05393c14b316e091f182c19bf888317c42aec153da774f02af378
    ;;
  *)
    echo "Unsupported PARENT_SET=${parent_set}" >&2
    exit 2
    ;;
esac

actual_direction_hash=$(sha256sum "${direction_manifest}" | awk '{print $1}')
if [[ "${actual_direction_hash}" != "${direction_manifest_sha256}" ]]; then
  echo "Direction manifest hash drift: ${actual_direction_hash} != ${direction_manifest_sha256}" >&2
  exit 2
fi

IFS=: read -r -a prerequisite_paths <<< "${prerequisite_summaries}"
python - "${prerequisite_paths[@]}" <<'PY'
import json
import sys
from pathlib import Path

for item in sys.argv[1:]:
    path = Path(item)
    if not path.is_file():
        raise SystemExit(f"Missing prerequisite summary: {path}")
    payload = json.loads(path.read_text())
    passed = payload.get("passed")
    if passed is None:
        passed = payload.get("stage1_gate_passed")
    if passed is not True:
        raise SystemExit(f"Prerequisite did not pass: {path}")
    if payload.get("validation_accessed") is not False:
        raise SystemExit(f"Prerequisite lacks frozen validation certification: {path}")
    if payload.get("test100_accessed") is not False:
        raise SystemExit(f"Prerequisite lacks frozen Test100 certification: {path}")
print(f"verified {len(sys.argv) - 1} prerequisite summaries")
PY

root_source_checkpoint_sha256=e6516b04917a9dfaae3c4d960b10d77c768f3e5e784f95290ff41c1202f9d9bc
source_run=${SOURCE_RUN_DIR:-/scratch/xzh/models/train/runs/qm9_hvp_curvature_v1_A_w0ep00_f1em02_seed314159_s1200}
source_checkpoint=${SOURCE_CHECKPOINT:-${source_run}/checkpoints/last.ckpt}
source_checkpoint_sha256=${SOURCE_CHECKPOINT_SHA256:-${root_source_checkpoint_sha256}}
resume_args=()
if [[ -n "${SOURCE_CHECKPOINT:-}" ]]; then
  resume_args+=(--resume-capacity-optimizer)
fi
output="${root}/capacity/${run_tag}_${parent_set}_${direction_role}_s${max_steps}_lr${learning_rate}_${SLURM_JOB_ID}"
mkdir -p "${output}" /scratch/xzh/logs/graphformer_relaxed_hvp

/usr/bin/time -v -o "${output}/resource.time" \
  python scripts/qm9_complete_total_capacity_train.py \
    --protocol configs/audit/qm9_graphformer_complete_total_relaxed_hvp_v1.yaml \
    --manifest "${parent_manifest}" \
    --direction-manifest "${direction_manifest}" \
    --direction-manifest-sha256 "${direction_manifest_sha256}" \
    --direction-role "${direction_role}" \
    --run "original_A=${source_run}=${source_checkpoint}" \
    --source-checkpoint-sha256 "${source_checkpoint_sha256}" \
    --root-source-checkpoint-sha256 "${root_source_checkpoint_sha256}" \
    --require-complete-total-relaxed-hvp \
    "${resume_args[@]}" \
    --output-dir "${output}" \
    --molecules "${molecules}" \
    --device cuda:0 \
    --transform-device cpu \
    --seed 20260723 \
    --max-steps "${max_steps}" \
    --learning-rate "${learning_rate}" \
    --weight-decay 0 \
    --hvp-loss-norm l2 \
    --loss-balance-mode none \
    --directions-per-step "${directions_per_step}" \
    --direction-repeat-steps 1 \
    --displacement 1e-3 \
    --eval-interval "${eval_interval}" \
    --checkpoint-interval "${checkpoint_interval}" \
    --early-stop-relative-frobenius 0.01 \
    --gradient-clip-norm 1 \
    --gradient-diagnostics-interval 25 \
    --lambda-e "${lambda_e}" \
    --lambda-f "${lambda_f}" \
    --lambda-rho "${lambda_rho}" \
    --lambda-h "${lambda_h}" \
    --lambda-q 0 \
    --lambda-spec 0 \
    --density-loss-scale 1e-2 \
    --density-refresh-interval 25 \
    --density-drift-trigger 1e-4 \
    --strict-active-density-refresh \
    --density-response-unroll-steps 0 \
    --connect-lagrange-multiplier-response \
    --implicit-density-parameter-response \
    --implicit-response-tolerance 3e-5 \
    --implicit-response-max-iterations 10000 \
    --implicit-response-damping 1e-8 \
    --implicit-response-diagonal-probes 16 \
    --implicit-response-warm-start \
    --base-initialization label_reference \
    --density-lr 1e-3 \
    --density-max-cycles 1000 \
    --density-first-threshold 1e-2 \
    --density-fallback-lr 3e-4 \
    --density-fallback-max-cycles 10000 \
    --density-fallback-threshold 1e-5 \
    --density-strict-threshold 1e-8 \
    --lbfgs-max-iterations 500 \
    --newton-max-iterations 6 \
    --integral-derivative-step 1e-4 \
    --integral-derivative-workers 8 \
    --integral-cache-entries 10
