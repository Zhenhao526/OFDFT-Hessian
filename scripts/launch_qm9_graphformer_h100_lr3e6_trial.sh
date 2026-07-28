#!/usr/bin/env bash
set -euo pipefail

root=/scratch/xzh/models/graphformer_relaxed_hvp/20260723/v1
repo=/scratch/xzh/code/structures25

export PARENT_SET=stable5
export DIRECTION_ROLE=all
export MOLECULE_IDS=0028399
export MAX_STEPS=20
export LEARNING_RATE=3e-6
export EVAL_INTERVAL=20
export CHECKPOINT_INTERVAL=10
export RUN_TAG=stage1_one_parent_h100_lr3e6_trial_from20_to40
export LAMBDA_E=1
export LAMBDA_F=1
export LAMBDA_RHO=0.1
export LAMBDA_H=100
export DIRECTIONS_PER_STEP=4
export SOURCE_RUN_DIR=/scratch/xzh/models/train/runs/qm9_hvp_curvature_v1_A_w0ep00_f1em02_seed314159_s1200
export SOURCE_CHECKPOINT="${root}/capacity/bound/h100_step20_bound.ckpt"
export SOURCE_CHECKPOINT_SHA256=c18d4352584379b2f3607b0057fa633305b0ee8994d705fef060d33852c0876b
export PREREQUISITE_SUMMARIES="${root}/verification/derivative_0028399_d0_p16_4056/summary.json:${root}/verification/full-basis_0028399_d0_p16_4058/summary.json"

exec sbatch --parsable \
  --job-name=gf_cth_lr3 \
  --partition=compute \
  --nodelist=node01 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=160G \
  --gres=gpu:1 \
  --time=7-00:00:00 \
  --output=/scratch/xzh/logs/graphformer_relaxed_hvp/%x_%j.out \
  --error=/scratch/xzh/logs/graphformer_relaxed_hvp/%x_%j.err \
  --export=ALL \
  --wrap="bash ${repo}/scripts/slurm_qm9_graphformer_complete_total_relaxed_hvp_capacity_v1.sbatch"
