#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/shenwei01/xzh_node02_20260724
REPO="$ROOT/work/structures25_autodiff_v8_20260805"
V11="$ROOT/work/qm9_graphformer_egfh_100mol_ddp8_v11"
DATA="$ROOT/data/qm9_graphformer_egfh_100mol_ddp8_v11"
CONSISTENCY="$ROOT/runs/qm9_graphformer_egfh_100mol_ddp8_v11/ddp_consistency_20260818a_attempt5"
OUTPUT="$ROOT/runs/qm9_graphformer_egfh_100mol_ddp8_v11/lambda_h_calibration_20260818a_attempt3"
LOG="$ROOT/runs/qm9_graphformer_egfh_100mol_ddp8_v11/calibration_supervisor_20260818a_attempt3.log"
TRAINER="$V11/scripts/qm9_v11_ddp_train.py"
PROTOCOL="$V11/configs/qm9_graphformer_egfh_100mol_ddp8_v11.calibration_frozen.yaml"
SPLIT="$DATA/frozen_split/final100_split_manifest.json"
SCHEDULE="$DATA/frozen_split/train80_ddp8_schedule.json"
DIRECTIONS="$DATA/candidate120_internal_bases/manifest.json"
SOURCE_RUN="$ROOT/runtime_parent/_runtime/models/train/runs/trained-on-qm9"
SOURCE_CHECKPOINT="$SOURCE_RUN/checkpoints/last.ckpt"
SOURCE_SHA=9759da26660c619de9c3bbf4c2dc164343ee90e08e22b3fcdcc9682dacb9bd09

if [[ "$(hostname)" != node05 ]]; then
  echo "fatal: must run on node05" >&2
  exit 2
fi
if [[ -e "$OUTPUT" ]]; then
  echo "fatal: isolated calibration output already exists: $OUTPUT" >&2
  exit 2
fi
exec >>"$LOG" 2>&1
echo "status=waiting_for_consistency started=$(date -Is)"
while [[ ! -f "$CONSISTENCY/consistency_report.json" ]]; do
  if ! pgrep -f 'ddp_consistency_20260818a_attempt5' >/dev/null; then
    echo "status=failed reason=consistency_process_disappeared time=$(date -Is)"
    exit 3
  fi
  sleep 30
done
consistency_status=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$CONSISTENCY/consistency_report.json")
if [[ "$consistency_status" != passed ]]; then
  echo "status=failed reason=consistency_gate_not_passed value=$consistency_status time=$(date -Is)"
  exit 4
fi
gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | wc -l)
if [[ "$gpu_processes" -ne 0 ]]; then
  echo "status=failed reason=gpus_not_exclusive count=$gpu_processes time=$(date -Is)"
  exit 5
fi

source "$REPO/scripts/activate_qm9_node_local.sh"
export MLDFT_DQC_OVERLAY="$ROOT/envs/dqc-torch-integrals-0fe821fc"
export PYTHONPATH="$REPO:$REPO/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export MLDFT_SYMMETRIC_MATRIX_POWER_MODE=eigh_second_order_audit
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
echo "status=running phase=train80_lambda_h_calibration started=$(date -Is)"
cd "$REPO"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 "$TRAINER" \
  --mode calibration \
  --protocol "$PROTOCOL" \
  --split-manifest "$SPLIT" \
  --direction-manifest "$DIRECTIONS" \
  --schedule "$SCHEDULE" \
  --source-checkpoint "$SOURCE_CHECKPOINT" \
  --source-run-dir "$SOURCE_RUN" \
  --source-checkpoint-sha256 "$SOURCE_SHA" \
  --output-dir "$OUTPUT" \
  --steps 10 \
  --learning-rate 0.0 \
  --lambda-h 0.01 \
  --checkpoint-interval 0
sha256sum "$OUTPUT/lambda_h_calibration.json" > "$OUTPUT/lambda_h_calibration.verified.sha256"
echo "status=complete phase=train80_lambda_h_calibration finished=$(date -Is)"
