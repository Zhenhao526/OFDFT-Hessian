#!/usr/bin/env bash
#SBATCH --job-name=qm9_esec_smoke
#SBATCH --partition=compute
#SBATCH --nodelist=node02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xzh/logs/qm9_energy_secant_smoke/%j.out
#SBATCH --error=/scratch/xzh/logs/qm9_energy_secant_smoke/%j.err

set -euo pipefail

ROOT=/scratch/xzh/code/structures25
DATA_ROOT=/scratch/xzh/data
MODEL_ROOT=/scratch/xzh/models
DATASET=QM9PBEForceRandom1000PairedAug
WEIGHT_CKPT=/scratch/xzh/models/train/runs/qm9_random1000_egf_forcew1p0_e10_20260714_202203/checkpoints/epoch_009.ckpt
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="qm9_random1000_energy_secant_smoke_${SLURM_JOB_ID}_${RUN_STAMP}"
DEVICES="${SMOKE_DEVICES:-1}"
if (( DEVICES > 1 )); then
  STRATEGY=ddp
  TRAIN_COMMAND=(torchrun --standalone --nproc_per_node="${DEVICES}")
else
  STRATEGY=auto
  TRAIN_COMMAND=(python)
fi

source /scratch/xzh/env.sh
cd "${ROOT}"

export DFT_DATA="${DATA_ROOT}"
export DFT_MODELS="${MODEL_ROOT}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

for path in \
  "${DFT_DATA}/${DATASET}/split.pkl" \
  "${WEIGHT_CKPT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: missing required input: ${path}" >&2
    exit 1
  fi
done

echo "run_name=${RUN_NAME}"
echo "host=$(hostname) devices=${DEVICES} gpu=${CUDA_VISIBLE_DEVICES:-scheduler-assigned}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

"${TRAIN_COMMAND[@]}" -m mldft.ml.train \
  experiment=str25/qm9_pbe_force_full_energy_secant \
  data.dataset_name="${DATASET}" \
  weight_ckpt_path="${WEIGHT_CKPT}" \
  name="${RUN_NAME}" \
  seed=676368232 \
  trainer.accelerator=gpu \
  trainer.devices="${DEVICES}" \
  trainer.strategy="${STRATEGY}" \
  trainer.sync_batchnorm=false \
  trainer.max_epochs=1 \
  +trainer.max_steps=3 \
  +trainer.limit_train_batches=3 \
  +trainer.limit_val_batches=0 \
  +trainer.num_sanity_val_steps=0 \
  trainer.log_every_n_steps=1 \
  trainer.use_distributed_sampler=false \
  data.datamodule.batch_size=4 \
  data.datamodule.num_workers=0 \
  callbacks.model_checkpoint.monitor=null \
  callbacks.model_checkpoint.save_top_k=0 \
  +callbacks.model_checkpoint.every_n_train_steps=3 \
  callbacks.throughput_monitor.log_every_n_steps=1 \
  callbacks.throughput_monitor.warmup_steps=0 \
  callbacks.throughput_monitor.estimate_num_samples=4800 \
  callbacks.throughput_monitor.target_num_devices="${DEVICES}" \
  train=true validate=false test=false \
  extras.enforce_tags=false \
  extras.print_config=false \
  hydra.callbacks.git_logging.clean=false \
  hydra.run.dir="${MODEL_ROOT}/train/runs/${RUN_NAME}" \
  'tags=[qm9_random1000,energy_secant,smoke,node02]'

RUN_DIR="${DFT_MODELS}/train/runs/${RUN_NAME}"
test -f "${RUN_DIR}/checkpoints/last.ckpt"
test -f "${RUN_DIR}/throughput/throughput_summary.json"
echo "smoke_run_dir=${RUN_DIR}"
echo "smoke_status=PASS"
