# QM9 Random1000 8xA100 EG/EGF Training

Date: 2026-07-13

## Goal

Launch EG and EGF lambda=1.0 training on the completed `QM9PBEForceRandom1000` dataset using one 8xA100 node.

This is a 1000-molecule demo/training validation, not the full 133,885-molecule QM9 run. No new labels were generated in this training step.

## Data Snapshot

Remote data root:

```text
/scratch/xzh/data/QM9PBEForceRandom1000
```

Counts:

```text
raw random subset: 1000 molecules
label .zarr.zip files: 4000
.chk files under kohn_sham: 4000
cached transformed labels: 4000
dataset size: 44G
```

Split file:

```text
/scratch/xzh/data/QM9PBEForceRandom1000/split.pkl
```

Split contents:

```text
label-file split: train=3200, val=400, test=400
expanded sample sizes recorded in split["sizes"]: train=42670, val=5404, test=5373
```

Cached transformed labels:

```text
/scratch/xzh/data/QM9PBEForceRandom1000/labels_local_frames_global_symmetric_natrep
```

Dataset statistics:

```text
/scratch/xzh/data/QM9PBEForceRandom1000/dataset_statistics/dataset_statistics_labels_local_frames_global_symmetric_natrep_e_kin_plus_xc.zarr
```

## Scripts

Local and remote scripts:

```text
scripts/launch_qm9_random1000_train_8xa100.sh
scripts/slurm_qm9_random1000_train_8xa100.sbatch
```

The Slurm script requests:

```text
node: node05
GPUs: 8x A100
CPUs: 144
memory: 900G
wall time: 12 h
exclusive: yes
```

`node04` had an existing non-Slurm 8-GPU job owned by `shenwei01`, so this run used the free 8xA100 `node05` instead.

One-line launch command from the remote code directory:

```bash
cd /scratch/xzh/code/structures25 && sbatch scripts/slurm_qm9_random1000_train_8xa100.sbatch
```

The launcher explicitly sets:

```text
TRAIN_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
EG per-GPU batch size=8
EGF per-GPU batch size=4
accumulate_grad_batches=1
epochs=10
```

Force remains derived from scalar energy autograd. No force head was added.

## Estimate Before Launch

The launcher prints a conservative budget:

```text
split/cache/statistics: 0.5-2 h
EG 10 epoch:           1-3 h
EGF 10 epoch:          2-5 h
end-to-end expected:   4-10 h
Slurm wall time:       12 h
```

For the successful job, split, cached transformed labels, and statistics already existed from previous attempts, so the actual wall time only covered EG plus EGF training and validation.

## Launch Attempts

Failed attempts:

```text
job 477: failed during transform because a Hydra callback override tried to set an unsupported enabled field.
job 478: completed split/cache/statistics, then failed at EG startup because Git logging tried to instantiate the callback with inherited clean=true in a non-git remote code tree.
```

Fixes:

```text
mldft/utils/log_utils/hydra_callbacks.py now skips git logging when no git repo is available and clean=false.
the training launcher uses hydra.callbacks.git_logging.clean=false.
the training launcher now records throughput estimates using split["sizes"]["train"] after split creation, not raw label count.
```

No existing labels, cached labels, or checkpoints were deleted during these fixes.

## Successful Run

Slurm job:

```text
job id: 479
state: COMPLETED
exit code: 0:0
node: node05
start: 2026-07-13T16:28:28
end: 2026-07-13T17:17:34
elapsed: 00:49:06
```

Slurm logs:

```text
/scratch/xzh/logs/qm9_random1000_train/479.out
/scratch/xzh/logs/qm9_random1000_train/479.err
```

Run root:

```text
/scratch/xzh/models/train_random1000/20260713_162829
```

## Training Results

| model | run dir | best checkpoint | train time | mean samples/s | peak GPU memory | final val metrics |
| --- | --- | --- | ---: | ---: | ---: | --- |
| EG | `/scratch/xzh/models/train/runs/qm9_random1000_eg_e10_20260713_162829` | `checkpoints/epoch_009.ckpt` | 15:44.80 | 521.53 | 808.99 MiB | energy loss 0.02444537, total 0.00249360 |
| EGF lambda=1.0 | `/scratch/xzh/models/train/runs/qm9_random1000_egf_lam1_e10_20260713_162829` | `checkpoints/epoch_009.ckpt` | 33:19.14 | 218.87 | 825.62 MiB | energy loss 0.02273401, force loss 0.00388426, total 0.00268172 |

Checkpoint files:

```text
/scratch/xzh/models/train/runs/qm9_random1000_eg_e10_20260713_162829/checkpoints/epoch_008.ckpt
/scratch/xzh/models/train/runs/qm9_random1000_eg_e10_20260713_162829/checkpoints/epoch_009.ckpt
/scratch/xzh/models/train/runs/qm9_random1000_eg_e10_20260713_162829/checkpoints/last.ckpt

/scratch/xzh/models/train/runs/qm9_random1000_egf_lam1_e10_20260713_162829/checkpoints/epoch_008.ckpt
/scratch/xzh/models/train/runs/qm9_random1000_egf_lam1_e10_20260713_162829/checkpoints/epoch_009.ckpt
/scratch/xzh/models/train/runs/qm9_random1000_egf_lam1_e10_20260713_162829/checkpoints/last.ckpt
```

Throughput JSON files:

```text
/scratch/xzh/models/train/runs/qm9_random1000_eg_e10_20260713_162829/throughput/throughput_summary.json
/scratch/xzh/models/train/runs/qm9_random1000_egf_lam1_e10_20260713_162829/throughput/throughput_summary.json
```

Important caveat: the completed job's throughput JSON has `estimate_num_samples=4000`, because the run used the old launcher value before the final script patch. The measured `mean_samples_per_sec`, wall time, and peak memory are still usable. Future runs with the patched launcher use `split["sizes"]["train"]=42670` for estimate reporting.

## Resource Use

The job used all 8 visible A100s under DDP. Observed GPU memory stayed low, roughly below 1 GiB by the throughput callback and a few GiB in `nvidia-smi` during training. This suggests the random1000 run is compute/autograd or data-pipeline limited rather than GPU-memory limited.

EGF was about 2.1x slower than EG in wall time for the same 10 epochs:

```text
EG:  15:44.80
EGF: 33:19.14
```

The relative slowdown is expected because EGF enables the force autograd path needed for `F_pred = -dE_pred/dR`.

## Current Status

Completed:

- random1000 force labels are available and validated;
- cached transformed labels and dataset statistics are available;
- EG 10 epoch 8xA100 training completed;
- EGF lambda=1.0 10 epoch 8xA100 training completed;
- checkpoints and throughput summaries are saved;
- launcher scripts are synced to `/scratch/xzh/code/structures25`.

Not done:

- no full-QM9 535k-label EG/EGF training was started;
- no density-relaxed Hessian evaluation was run in this training job;
- no new labels were generated during training;
- no force head was introduced.

## Next Steps

1. Run force validation on the test split for the random1000 EG and EGF checkpoints.
2. If the random1000 result is acceptable, decide whether to generate the full force-label dataset or use an existing external force/Hessian dataset.
3. For full QM9 training, use the same DDP path but expect the bottleneck to shift to data I/O and EGF force autograd cost; run a short calibration after the full label cache exists.
