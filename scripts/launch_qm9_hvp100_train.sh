#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <A|B|C|D> <hvp_weight> <seed> <run_name>" >&2
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
export HVP_ROOT="${HVP_ROOT:-${DFT_MODELS}/hvp100/20260716}"
export HVP_SIDECAR_DIR="${HVP_SIDECAR_DIR:-${HVP_ROOT}/hvp_sidecars/sidecars}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

BASE_RUN="${BASE_RUN:-${DFT_MODELS}/train/runs/qm9_random1000_egf_forcew1p0_e10_20260714_202203}"
BASE_CKPT="${BASE_CKPT:-${BASE_RUN}/checkpoints/epoch_009.ckpt}"
SPLIT_FILE="${SPLIT_FILE:-${HVP_ROOT}/selection/split.pkl}"
TORCHRUN="${TORCHRUN:-/scratch/xzh/envs/structures25/bin/torchrun}"
MASTER_PORT="${MASTER_PORT:-$((31000 + RANDOM % 20000))}"
MAX_STEPS="${MAX_STEPS:-600}"

case "${VARIANT}" in
  A)
    EXPERIMENT="str25/qm9_pbe_force_full_egf_energy_secant"
    SECANT_WEIGHT=0
    ;;
  B)
    EXPERIMENT="str25/qm9_pbe_force_full_egf_energy_secant"
    SECANT_WEIGHT=0.01
    ;;
  C)
    EXPERIMENT="str25/qm9_pbe_force_full_egf_hvp"
    SECANT_WEIGHT=0
    ;;
  D)
    EXPERIMENT="str25/qm9_pbe_force_full_egf_energy_secant_hvp"
    SECANT_WEIGHT=0.01
    ;;
  *)
    echo "Unknown variant: ${VARIANT}" >&2
    exit 2
    ;;
esac

for path in "${BASE_CKPT}" "${SPLIT_FILE}"; do
  [[ -e "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 1; }
done
if [[ "${VARIANT}" == C || "${VARIANT}" == D ]]; then
  [[ "$(find "${HVP_SIDECAR_DIR}" -maxdepth 1 -name '*.npz' | wc -l)" -eq 120 ]] || {
    echo "Expected 120 HVP sidecars in ${HVP_SIDECAR_DIR}" >&2
    exit 1
  }
fi

EXTRA=(
  "data.dataset_name=QM9PBEForceRandom1000PairedAug"
  "data.datamodule.split_file=${SPLIT_FILE}"
  "data.datamodule.dataset_kwargs.limit_scf_iterations=[-1]"
  "data.datamodule.dataset_kwargs.keep_initial_guess=false"
  "data.datamodule.dataset_kwargs.load_force_label=true"
  "data.datamodule.dataset_kwargs.load_pair_metadata=true"
  "data.datamodule.pair_grouped_train_batches=true"
  "data.datamodule.pair_batch_seed=${SEED}"
  "data.datamodule.pair_source_markers=[QM9PBEForceRandom1000PairedTrain]"
  "data.datamodule.infer_pair_metadata_from_filename=true"
  "model.loss_gradient_norm_interval=150"
  "model.loss_gradient_norm_max_logs=4"
)
if [[ "${VARIANT}" == A || "${VARIANT}" == B ]]; then
  EXTRA+=("model.loss_function.energy_secant_loss.weight=${SECANT_WEIGHT}")
else
  EXTRA+=(
    "model.loss_function.hvp_loss.weight=${HVP_WEIGHT}"
    "data.datamodule.dataset_kwargs.load_hvp_label=true"
    "data.datamodule.dataset_kwargs.hvp_label_dir=${HVP_SIDECAR_DIR}"
    "data.datamodule.dataset_kwargs.hvp_reference_scale_floor=1e-2"
  )
  if [[ "${VARIANT}" == D ]]; then
    EXTRA+=("model.loss_function.energy_secant_loss.weight=${SECANT_WEIGHT}")
  fi
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
  "trainer.use_distributed_sampler=false" \
  "data.datamodule.batch_size=4" \
  "data.datamodule.num_workers=0" \
  "validate=true" \
  "test=false" \
  "extras.enforce_tags=false" \
  "extras.print_config=false" \
  "hydra.callbacks.git_logging.clean=false" \
  "${EXTRA[@]}"
