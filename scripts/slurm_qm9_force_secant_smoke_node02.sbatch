#!/usr/bin/env bash
#SBATCH --job-name=qm9_fsec_smoke
#SBATCH --partition=compute
#SBATCH --nodelist=node02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xzh/logs/qm9_force_secant_smoke/%j.out
#SBATCH --error=/scratch/xzh/logs/qm9_force_secant_smoke/%j.err

set -euo pipefail

source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
mkdir -p /scratch/xzh/logs/qm9_force_secant_smoke

export DFT_DATA=/scratch/xzh/data
export DFT_MODELS=/scratch/xzh/models
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

DATASET=QM9PBEForceRandom1000PairedAug
WEIGHT_CKPT=/scratch/xzh/models/train/runs/qm9_random1000_egf_forcew1p0_e10_20260714_202203/checkpoints/epoch_009.ckpt
RUN_NAME="qm9_random1000_force_secant_smoke_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"

python -m mldft.ml.train \
  experiment=str25/qm9_pbe_force_full_egf_energy_secant \
  data.dataset_name="${DATASET}" \
  weight_ckpt_path="${WEIGHT_CKPT}" \
  name="${RUN_NAME}" \
  seed=676368232 \
  trainer.accelerator=gpu trainer.devices=1 trainer.strategy=auto \
  trainer.sync_batchnorm=false trainer.max_epochs=1 \
  +trainer.max_steps=3 +trainer.limit_train_batches=3 \
  +trainer.limit_val_batches=0 +trainer.num_sanity_val_steps=0 \
  trainer.log_every_n_steps=1 trainer.use_distributed_sampler=false \
  data.datamodule.batch_size=4 data.datamodule.num_workers=0 \
  callbacks.model_checkpoint.monitor=null callbacks.model_checkpoint.save_top_k=0 \
  +callbacks.model_checkpoint.every_n_train_steps=3 \
  callbacks.throughput_monitor.log_every_n_steps=1 \
  callbacks.throughput_monitor.warmup_steps=0 \
  callbacks.throughput_monitor.estimate_num_samples=4800 \
  callbacks.throughput_monitor.target_num_devices=1 \
  train=true validate=false test=false \
  extras.enforce_tags=false extras.print_config=false \
  hydra.callbacks.git_logging.clean=false \
  hydra.run.dir="${DFT_MODELS}/train/runs/${RUN_NAME}" \
  'tags=[qm9_random1000,force_weight_1,energy_secant_w0p01,smoke,node02]'

RUN_DIR="${DFT_MODELS}/train/runs/${RUN_NAME}"
test -f "${RUN_DIR}/checkpoints/last.ckpt"
test -f "${RUN_DIR}/throughput/throughput_summary.json"
echo "smoke_run_dir=${RUN_DIR}"
echo "smoke_status=PASS"
