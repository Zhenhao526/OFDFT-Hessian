#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 MOLECULE_ID [MOLECULE_ID ...]" >&2
  exit 2
fi

ROOT=/home/shenwei01/xzh_node02_20260724
REPO="$ROOT/work/structures25"
DATASET="$ROOT/data/QM9PBEForceEGFH10ScratchV1"
REFERENCE_DIR="$ROOT/artifacts/graphformer_hybrid_relaxed_hvp_rebuild_v1/pbe_hessian_train20/cache"
TRAIN_ROOT="$ROOT/models/train/runs/qm9_graphformer_egfh10_force_secant_scratch_s20_v1"
CHECKPOINT="$TRAIN_ROOT/checkpoints/last.ckpt"
RUN_NAME=EGFH10ScratchS20
OUT_ROOT="$ROOT/runs/qm9_graphformer_egfh10_total_hessian_vibration_s20_v1"
DEVICE="${EGFH10_HESSIAN_DEVICE:-0}"
EXPECTED_CHECKPOINT_SHA=ab7070d1739f57684a4955ebab9dcce903073b039721fdb391774b8989f71c1e

[[ "$(hostname)" == node02 ]] || {
  echo "This shard runner is authorized on node02 only." >&2
  exit 2
}
[[ "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" == "$EXPECTED_CHECKPOINT_SHA" ]] || {
  echo "Checkpoint hash mismatch." >&2
  exit 1
}
mkdir -p "$OUT_ROOT/per_molecule" "$OUT_ROOT/logs" "$ROOT/tmp" "$ROOT/cache"
cd "$REPO"
source "$REPO/scripts/activate_qm9_node02_local.sh"
source "$REPO/.venv/bin/activate"
export DFT_DATA="$ROOT/data"
export DFT_MODELS="$ROOT/models"
export TMPDIR="$ROOT/tmp"
export XDG_CACHE_HOME="$ROOT/cache"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="$DEVICE"

for molecule_id in "$@"; do
  molecule_out="$OUT_ROOT/per_molecule/$molecule_id"
  hessian_npz="$molecule_out/${RUN_NAME}_${molecule_id}_0000000_total_hessian.npz"
  if [[ -f "$molecule_out/summary.json" && -f "$hessian_npz" ]]; then
    echo "Skipping completed molecule $molecule_id"
    continue
  fi
  mkdir -p "$molecule_out"
  /usr/bin/time -v -o "$OUT_ROOT/logs/${molecule_id}.time.txt" \
    python scripts/qm9_total_ofdft_hessian_audit.py \
      --dataset-dir "$DATASET" \
      --reference-dir "$REFERENCE_DIR" \
      --run "${RUN_NAME}=${TRAIN_ROOT}=${CHECKPOINT}" \
      --molecules "$molecule_id" \
      --sample-id 0 \
      --output-dir "$molecule_out" \
      --displacement 1e-4 \
      --integral-derivative-step 1e-4 \
      --integral-derivative-workers 4 \
      --model-geometry-derivative autograd \
      --base-initialization label_reference \
      --optimizer adam \
      --lr 1e-3 \
      --max-cycle 1000 \
      --convergence-tolerance 1e-2 \
      --fallback-optimizer adam \
      --fallback-lr 3e-4 \
      --fallback-max-cycle 10000 \
      --fallback-convergence-tolerance 1e-5 \
      --fallback-always \
      --lbfgs-refine \
      --newton-refine \
      --device cuda:0 \
      --transform-device cpu \
      >"$OUT_ROOT/logs/${molecule_id}.log" 2>&1
done
