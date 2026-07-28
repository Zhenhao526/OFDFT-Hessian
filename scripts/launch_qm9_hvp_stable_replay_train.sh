#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <A|B|C|D|E> <curvature_weight> <seed> <run_name>" >&2
  exit 2
fi

VARIANT="$1"
HVP_WEIGHT="$2"
SEED="$3"
RUN_NAME="$4"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
source "${ENV_FILE:-/scratch/xzh/env.sh}"

export DFT_DATA="${DFT_DATA:-/scratch/xzh/data}"
export DFT_MODELS="${DFT_MODELS:-/scratch/xzh/models}"
export HVP_ROOT="${HVP_ROOT:-${DFT_MODELS}/hvp_branch_stability/20260716}"
export STABLE_SIDECARS="${STABLE_SIDECARS:-${HVP_ROOT}/formal_train100_v2/analysis/stable_sidecars}"
export IMPLICIT_SIDECARS="${IMPLICIT_SIDECARS:-${HVP_ROOT}/formal_train100_v2/implicit_targets/sidecars}"
export HVP_SIDECAR_DIR="${HVP_SIDECAR_DIR:-${STABLE_SIDECARS}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

BASE_RUN="${BASE_RUN:-${DFT_MODELS}/train/runs/qm9_random1000_egf_forcew1p0_e10_20260714_202203}"
BASE_CKPT="${BASE_CKPT:-${BASE_RUN}/checkpoints/epoch_009.ckpt}"
SPLIT_FILE="${SPLIT_FILE:-${DFT_DATA}/QM9PBEForceRandom1000PairedAug/split.pkl}"
TORCHRUN="${TORCHRUN:-/scratch/xzh/envs/structures25/bin/torchrun}"
MASTER_PORT="${MASTER_PORT:-$((31000 + RANDOM % 20000))}"
MAX_STEPS="${MAX_STEPS:-1200}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"

case "${VARIANT}" in
  A)
    EXPERIMENT="str25/qm9_pbe_force_full_egf_energy_secant"
    SECANT_WEIGHT=0
    ;;
  B)
    EXPERIMENT="str25/qm9_pbe_force_full_egf_energy_secant"
    SECANT_WEIGHT="${SECANT_WEIGHT:-0.01}"
    ;;
  C)
    EXPERIMENT="str25/qm9_pbe_force_full_egf_hvp"
    SECANT_WEIGHT=0
    ;;
  D)
    EXPERIMENT="str25/qm9_pbe_force_full_egf_relaxed_force_secant"
    SECANT_WEIGHT=0
    ;;
  E)
    EXPERIMENT="str25/qm9_pbe_force_full_egf_implicit_target_hvp"
    SECANT_WEIGHT=0
    ;;
  *)
    echo "Unknown variant: ${VARIANT}" >&2
    exit 2
    ;;
esac

for path in "${BASE_CKPT}" "${SPLIT_FILE}" "${STABLE_SIDECARS}"; do
  [[ -e "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 1; }
done
if [[ "${VARIANT}" == E ]]; then
  [[ -d "${IMPLICIT_SIDECARS}" ]] || { echo "Missing input: ${IMPLICIT_SIDECARS}" >&2; exit 1; }
fi
SIDECAR_COUNT="$(find "${STABLE_SIDECARS}" -maxdepth 1 -name '*.0000000.npz' | wc -l)"
if [[ "${SIDECAR_COUNT}" -eq 0 || "${SIDECAR_COUNT}" -gt 100 ]]; then
  echo "Expected 1-100 preregistered stable train100 sidecars, found ${SIDECAR_COUNT}" >&2
  exit 1
fi

EXTRA=(
  "data.dataset_name=QM9PBEForceRandom1000PairedAug"
  "data.datamodule.split_file=${SPLIT_FILE}"
  "data.datamodule.dataset_kwargs.limit_scf_iterations=[-1]"
  "data.datamodule.dataset_kwargs.keep_initial_guess=false"
  "++data.datamodule.dataset_kwargs.load_force_label=true"
  "++data.datamodule.dataset_kwargs.load_pair_metadata=true"
  "++data.datamodule.dataset_kwargs.load_hvp_label=true"
  "++data.datamodule.dataset_kwargs.hvp_label_dir=${STABLE_SIDECARS}"
  "++data.datamodule.dataset_kwargs.hvp_reference_scale_floor=1e-2"
  "++data.datamodule.dataset_kwargs.hvp_direction_seed=${SEED}"
  "data.datamodule.pair_grouped_train_batches=true"
  "data.datamodule.pair_batch_seed=${SEED}"
  "data.datamodule.pair_source_markers=[QM9PBEForceRandom1000PairedTrain]"
  "data.datamodule.infer_pair_metadata_from_filename=true"
  "data.datamodule.pair_replay_sidecar_dir=${STABLE_SIDECARS}"
  "data.datamodule.pair_replay_interval=4"
  "data.datamodule.pair_replay_phase=0"
  "data.datamodule.hvp_replay_sidecar_dir=${STABLE_SIDECARS}"
  "data.datamodule.hvp_replay_per_batch=true"
  "data.datamodule.hvp_replay_interval=4"
  "data.datamodule.hvp_replay_phase=1"
  "model.optimizer.lr=${LEARNING_RATE}"
  "model.loss_function.force_loss.weight=1.0"
  "model.hvp_batch_fraction=1.0"
  "model.hvp_start_step=${HVP_START_STEP:-100}"
  "model.hvp_ramp_steps=${HVP_RAMP_STEPS:-200}"
  "model.hvp_update_interval=1"
  "model.hvp_update_phase=0"
  "model.loss_gradient_norm_interval=${LOSS_GRADIENT_NORM_INTERVAL:-50}"
  "model.loss_gradient_norm_max_logs=${LOSS_GRADIENT_NORM_MAX_LOGS:-12}"
  "model.loss_balance_mode=gradnorm"
  "model.hvp_gradnorm_target_ratio=0.25"
  "model.gradnorm_multiplier_min=0.05"
  "model.gradnorm_multiplier_max=5.0"
)

if [[ "${VARIANT}" == A || "${VARIANT}" == B ]]; then
  EXTRA+=(
    "model.loss_function.energy_secant_loss.weight=${SECANT_WEIGHT}"
    "model.loss_function.energy_secant_loss.loss.strict_pairs=false"
  )
elif [[ "${VARIANT}" == C || "${VARIANT}" == E ]]; then
  EXTRA+=(
    "model.loss_function.hvp_loss.weight=${HVP_WEIGHT}"
    "model.loss_function.hvp_loss.loss.reference_scale_floor=${HVP_REFERENCE_FLOOR:-1e-2}"
    "model.loss_function.hvp_loss.loss.max_normalized_loss=${HVP_NORMALIZED_CAP:-10.0}"
  )
  if [[ "${VARIANT}" == E ]]; then
    EXTRA+=(
      "++data.datamodule.dataset_kwargs.hvp_label_dir=${IMPLICIT_SIDECARS}"
      "++data.datamodule.dataset_kwargs.hvp_target_key=implicit_complete_total_hvp_target"
    )
  fi
else
  EXTRA+=(
    "model.loss_function.relaxed_force_secant_loss.weight=${HVP_WEIGHT}"
    "model.loss_function.relaxed_force_secant_loss.loss.reference_scale_floor=${HVP_REFERENCE_FLOOR:-1e-2}"
    "model.loss_function.relaxed_force_secant_loss.loss.max_normalized_loss=${HVP_NORMALIZED_CAP:-10.0}"
  )
fi

exec "${TORCHRUN}" --standalone --nproc_per_node=1 --master_port="${MASTER_PORT}" \
  -m mldft.ml.train \
  "experiment=${EXPERIMENT}" \
  "name=${RUN_NAME}" \
  "hydra.run.dir=${DFT_MODELS}/train/runs/${RUN_NAME}" \
  "weight_ckpt_path=${BASE_CKPT}" \
  "seed=${SEED}" \
  "trainer.accelerator=gpu" \
  "trainer.devices=1" \
  "trainer.num_nodes=1" \
  "trainer.strategy=auto" \
  "trainer.sync_batchnorm=false" \
  "trainer.max_epochs=40" \
  "+trainer.max_steps=${MAX_STEPS}" \
  "trainer.accumulate_grad_batches=8" \
  "trainer.check_val_every_n_epoch=5" \
  "trainer.log_every_n_steps=${LOG_EVERY_N_STEPS:-10}" \
  "trainer.use_distributed_sampler=false" \
  "callbacks.model_checkpoint.monitor=null" \
  "callbacks.model_checkpoint.save_top_k=0" \
  "callbacks.model_checkpoint.save_last=true" \
  "callbacks.model_checkpoint.every_n_train_steps=${CHECKPOINT_EVERY_N_STEPS:-200}" \
  "callbacks.model_checkpoint.save_on_train_epoch_end=false" \
  "data.datamodule.batch_size=4" \
  "data.datamodule.num_workers=0" \
  "validate=true" \
  "test=false" \
  "extras.enforce_tags=false" \
  "extras.print_config=false" \
  "hydra.callbacks.git_logging.clean=false" \
  "${EXTRA[@]}"
