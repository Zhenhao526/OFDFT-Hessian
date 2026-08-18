#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/shenwei01/xzh_node02_20260724
REPO="$ROOT/work/structures25_autodiff_v8_20260805"
V11="$ROOT/work/qm9_graphformer_egfh_100mol_ddp8_v11"
DATA="$ROOT/data/qm9_graphformer_egfh_100mol_ddp8_v11"
RUN="$ROOT/runs/qm9_graphformer_egfh_100mol_ddp8_v11/ddp_consistency_20260818a_attempt5"
TRAINER="$V11/scripts/qm9_v11_ddp_train.py"
COMPARE="$V11/scripts/compare_v11_consistency.py"
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
if [[ -e "$RUN" ]]; then
  echo "fatal: isolated consistency output already exists: $RUN" >&2
  exit 2
fi
gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | wc -l)
if [[ "$gpu_processes" -ne 0 ]]; then
  echo "fatal: node05 GPUs are not exclusive; found $gpu_processes compute processes" >&2
  exit 2
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
mkdir -p "$RUN"
exec > >(tee -a "$RUN/launcher.log") 2>&1
echo "status=running phase=reference1gpu started=$(date -Is)"

cd "$REPO"
CUDA_VISIBLE_DEVICES=0 python3 "$TRAINER" \
  --mode probe \
  --reference-global-batch \
  --protocol "$PROTOCOL" \
  --split-manifest "$SPLIT" \
  --direction-manifest "$DIRECTIONS" \
  --schedule "$SCHEDULE" \
  --source-checkpoint "$SOURCE_CHECKPOINT" \
  --source-run-dir "$SOURCE_RUN" \
  --source-checkpoint-sha256 "$SOURCE_SHA" \
  --output-dir "$RUN/reference1gpu" \
  --steps 1 \
  --learning-rate 1.0e-6 \
  --lambda-h 0.01 \
  --checkpoint-interval 0

echo "status=running phase=ddp8 started=$(date -Is)"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 "$TRAINER" \
  --mode probe \
  --protocol "$PROTOCOL" \
  --split-manifest "$SPLIT" \
  --direction-manifest "$DIRECTIONS" \
  --schedule "$SCHEDULE" \
  --source-checkpoint "$SOURCE_CHECKPOINT" \
  --source-run-dir "$SOURCE_RUN" \
  --source-checkpoint-sha256 "$SOURCE_SHA" \
  --output-dir "$RUN/ddp8" \
  --steps 1 \
  --learning-rate 1.0e-6 \
  --lambda-h 0.01 \
  --checkpoint-interval 0

echo "status=running phase=compare started=$(date -Is)"
python3 "$COMPARE" \
  --reference "$RUN/reference1gpu/consistency_probe.pt" \
  --ddp8 "$RUN/ddp8/consistency_probe.pt" \
  --output "$RUN/consistency_report.json"
sha256sum "$RUN/consistency_report.json" > "$RUN/consistency_report.sha256"
echo "status=complete phase=consistency_passed finished=$(date -Is)"
