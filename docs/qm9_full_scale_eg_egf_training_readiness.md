# QM9 Full-Scale EG/EGF Training Readiness

Date: 2026-07-07

## Scope

This prepares the training code for a later 8xA100 full-QM9 fixed-density EG/EGF job.

Constraints kept:

- no new labels generated;
- no existing labels, checkpoints, cached labels, or Hessian references deleted;
- no independent force head added;
- EGF force remains `F_pred = -dE_pred/dR` from scalar energy autograd;
- training and validation do not run density-relaxed Hessian.

Target full-scale dataset assumption:

- dataset directory: `${DFT_DATA}/QM9PBEForceFull`;
- split: `${DFT_DATA}/QM9PBEForceFull/split.pkl`;
- labels: about `133885 * 4 = 535540` geometries.

## Changed Files

Core training:

- `mldft/ml/models/mldft_module.py`
- `mldft/ml/callbacks/throughput.py`
- `mldft/ml/callbacks/__init__.py`

Configs:

- `configs/datagen/preset/qm9_pbe_force_full.yaml`
- `configs/ml/data/qm9_pbe_force_full.yaml`
- `configs/ml/trainer/ddp_8xa100.yaml`
- `configs/ml/callbacks/qm9_full_scale.yaml`
- `configs/ml/callbacks/throughput_monitor.yaml`
- `configs/ml/experiment/str25/qm9_pbe_force_full_eg.yaml`
- `configs/ml/experiment/str25/qm9_pbe_force_full_egf.yaml`

Launch scripts:

- `scripts/launch_qm9_full_scale_8xa100.sh`
- `scripts/launch_qm9_full_scale_4gpu.sh`
- `scripts/launch_qm9_throughput_calibration.sh`
- `scripts/launch_qm9_full_force_labels_shard.sh`
- `scripts/slurm_qm9_full_force_labels_array.sbatch`
- `scripts/download_hessian_qm9_figshare.py`

Tests:

- `tests/ml/test_mldft_module.py`

## DDP Readiness

The training stack uses Lightning DDP:

- `torchrun` launch is supported by `scripts/launch_qm9_full_scale_8xa100.sh`;
- `trainer.strategy=ddp`, `accelerator=gpu`, `devices=8`, `num_nodes=1`;
- per-rank CUDA device assignment is handled by Lightning from `LOCAL_RANK`;
- distributed train sampling is handled by Lightning's distributed sampler path;
- TensorBoard logging and checkpointing remain rank-zero behavior through Lightning callbacks;
- project loggers use `RankedLogger(rank_zero_only=True)`;
- validation/test loss and metric logging now uses explicit `sync_dist=True`;
- full-scale EG validation uses energy loss only;
- full-scale EGF validation uses energy + force loss only;
- resume remains `ckpt_path=/path/to/last.ckpt`.

Resume check:

```text
DFT_DATA=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1
DFT_MODELS=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models
.venv/bin/python -m pytest tests/ml/test_train.py::test_train_resume -q
```

Result:

```text
1 passed in 28.66s
```

## Batch and Accumulation

The data batch size is per GPU/process under DDP.

Defaults:

| model | per-GPU batch | accumulation | effective 8-GPU global batch |
| --- | ---: | ---: | ---: |
| EG | 8 | 1 | 64 |
| EGF lambda=1.0 | 4 | 1 | 32 |

EGF batch 8 can be tested with:

```bash
PER_GPU_BATCH_SIZE=8 scripts/launch_qm9_full_scale_8xa100.sh egf 10
```

Gradient accumulation can be changed with:

```bash
ACCUMULATE_GRAD_BATCHES=2 scripts/launch_qm9_full_scale_8xa100.sh egf 10
```

The throughput callback records `per_gpu_batch_size`, `accumulate_grad_batches`, and `effective_global_batch_size`.

## Autograd Path

The model still computes density gradients from scalar energy for EG and EGF.

Optimization applied:

- EG has `force_supervision=false`, so it does not compute `pred_forces`.
- EGF computes force only when `force_supervision=true`.
- Force autograd uses `create_graph=True` only when training and an active nonzero `force_loss` exists.
- Validation/test force autograd uses `create_graph=False`.
- Full-scale validation disables default density-gradient metrics with `metric_interval=0`.
- Full-scale validation uses a separate `validation_loss_function`.
- Retain-graph is restricted to active-loss cases that still need shared scalar-energy graph outputs.
- Coarse timing is optional through `model.profile_timing=true`.

Important nuance:

- EG still needs high-order autograd for the density-gradient loss path, because the loss is applied to `dE/dcoeffs`.
- The avoided overhead is the force-specific `dE/dR` graph.

## Throughput Calibration

New callback:

- class: `mldft.ml.callbacks.ThroughputMonitor`;
- CSV: `${paths.output_dir}/throughput/throughput.csv`;
- JSON: `${paths.output_dir}/throughput/throughput_summary.json`;
- records samples/sec, steps/sec, data loading time, net forward time, density-gradient autograd time, force autograd time, backward time, optimizer time, peak GPU memory, epoch estimate.

Development calibration script:

```bash
DFT_DATA=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1 \
DFT_MODELS=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models \
CUDA_VISIBLE_DEVICES=5 \
RUN_NAME=qm9_full_eg_calibration_smoke_1gpu_4steps_v2 \
PER_GPU_BATCH_SIZE=2 \
NUM_WORKERS=0 \
scripts/launch_qm9_throughput_calibration.sh eg 4 1
```

DDP calibration example:

```bash
DFT_DATA=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1 \
DFT_MODELS=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models \
CUDA_VISIBLE_DEVICES=2,5 \
RUN_NAME=qm9_full_egf_calibration_smoke_2gpu_2steps_v2 \
PER_GPU_BATCH_SIZE=1 \
NUM_WORKERS=0 \
scripts/launch_qm9_throughput_calibration.sh egf 2 2
```

## Smoke Tests

Static/config checks:

| check | result |
| --- | --- |
| `py_compile mldft/ml/models/mldft_module.py mldft/ml/callbacks/throughput.py` | pass |
| full EG config compose | pass |
| full EGF config compose | pass |
| `bash -n` launch scripts | pass |
| `pytest tests/ml/test_mldft_module.py -q` | 7 passed |
| `pytest tests/ml/test_train.py::test_train_resume -q` with `DFT_DATA/DFT_MODELS` | 1 passed |

Training smoke:

| run | devices | steps | per-GPU batch | result | wall | max RSS |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| EG calibration smoke | 1 | 4 | 2 | pass | 0:27.95 | 1.91 GB |
| EGF calibration smoke | 1 | 4 | 1 | pass | 0:30.14 | 1.94 GB |
| EG DDP smoke | 2 | 2 | 1 | pass | 0:38.08 | 2.03 GB |
| EGF DDP smoke | 2 | 2 | 1 | pass | 0:42.63 | 2.08 GB |
| EG validation smoke | 1 | 1 train + 1 val | 1 | pass | 0:27.45 | 1.87 GB |
| EGF validation smoke | 1 | 1 train + 1 val | 1 | pass | 0:25.16 | 1.86 GB |
| EGF validation-only smoke | 1 | 1 val | 1 | pass | 0:23.01 | 1.49 GB |

DDP backend:

- `distributed_backend=nccl`;
- both EG and EGF reached `Trainer.fit stopped: max_steps reached`;
- EGF DDP included `train_loss/force_loss`, confirming the force path participates under DDP.
- EGF validation-only reported only `val_loss/energy_loss`, `val_loss/force_loss`, and `val_loss/total`.

Observed warning:

- DDP emitted `Grad strides do not match bucket view strides`; this is a performance warning, not a correctness failure.

## Smoke Throughput

These numbers are from tiny P1-410 calibration runs with very small batch sizes and few steps. They validate the measurement path but should not be used as final full-scale performance estimates.

| run | measured samples/sec | linear 8-GPU samples/sec | peak GPU memory | 10 epoch estimate | 20 epoch estimate |
| --- | ---: | ---: | ---: | ---: | ---: |
| EG 1-GPU, batch 2, 4 steps | 2.227 | 17.819 | 379.7 MB | 83.5 h / 667.9 GPU-h | 167.0 h / 1335.8 GPU-h |
| EGF 1-GPU, batch 1, 4 steps | 1.107 | 8.860 | 378.5 MB | 167.9 h / 1343.3 GPU-h | 335.8 h / 2686.6 GPU-h |
| EGF 2-GPU, batch 1, 2 steps | 1.228 | 4.914 | 451.8 MB | 302.8 h / 2422.0 GPU-h | 605.5 h / 4844.1 GPU-h |

The 2-GPU short run is dominated by startup/warmup and is pessimistic. Before submitting full 10/20 epoch jobs, run a calibration with realistic batch sizes and at least 50-200 measured steps.

## Formal 8xA100 Launch

EG, 10 epochs:

```bash
DFT_DATA=/path/to/data_root \
DFT_MODELS=/path/to/model_root \
scripts/launch_qm9_full_scale_8xa100.sh eg 10
```

EG, 20 epochs:

```bash
DFT_DATA=/path/to/data_root \
DFT_MODELS=/path/to/model_root \
scripts/launch_qm9_full_scale_8xa100.sh eg 20
```

EGF lambda=1.0, 10 epochs:

```bash
DFT_DATA=/path/to/data_root \
DFT_MODELS=/path/to/model_root \
scripts/launch_qm9_full_scale_8xa100.sh egf 10
```

EGF lambda=1.0, 20 epochs:

```bash
DFT_DATA=/path/to/data_root \
DFT_MODELS=/path/to/model_root \
scripts/launch_qm9_full_scale_8xa100.sh egf 20
```

Resume:

```bash
DFT_DATA=/path/to/data_root \
DFT_MODELS=/path/to/model_root \
scripts/launch_qm9_full_scale_8xa100.sh egf 20 qm9_full_egf_e20_resume \
  ckpt_path=/path/to/previous/run/checkpoints/last.ckpt
```

Optional EGF batch 8 test:

```bash
PER_GPU_BATCH_SIZE=8 \
DFT_DATA=/path/to/data_root \
DFT_MODELS=/path/to/model_root \
scripts/launch_qm9_full_scale_8xa100.sh egf 10 qm9_full_egf_e10_batch8
```

## Formal 4-GPU Launch

For a direct 4-GPU full-scale job, use the wrapper below. It sets:

- `CUDA_VISIBLE_DEVICES=0,1,2,3` by default;
- `NPROC_PER_NODE=4`;
- `DFT_MODELS=<repo>/_runtime/qm9_full_models` if unset;
- EG per-GPU batch `8`;
- EGF per-GPU batch `4`;
- throughput target device count `4`.

The wrapper validates `${DFT_DATA}/QM9PBEForceFull/split.pkl` before launching. If `DFT_DATA` is unset, it tries common local roots and exits before training if the full dataset cannot be found.

EG, 10 epochs:

```bash
scripts/launch_qm9_full_scale_4gpu.sh eg 10
```

EGF lambda=1.0, 10 epochs:

```bash
scripts/launch_qm9_full_scale_4gpu.sh egf 10
```

EGF lambda=1.0, 20 epochs:

```bash
scripts/launch_qm9_full_scale_4gpu.sh egf 20
```

If the full dataset is not in one of the auto-detected paths, keep the launch to one line by prefixing `DFT_DATA`:

```bash
DFT_DATA=/path/to/data_root scripts/launch_qm9_full_scale_4gpu.sh egf 10
```

Resume:

```bash
scripts/launch_qm9_full_scale_4gpu.sh egf 20 qm9_full_egf_resume \
  ckpt_path=/path/to/previous/run/checkpoints/last.ckpt
```

## Current Risks

- Full `QM9PBEForceFull` data and split were not present on the development machine, so full-scale data I/O was not tested.
- Throughput numbers are smoke-only and pessimistic; run a realistic 8-GPU calibration before reserving a long job.
- `num_workers=8` is a starting point. Dataloader throughput should be tuned on the actual filesystem.
- DDP grad stride warnings may reduce scaling efficiency and should be profiled if throughput is poor.

## Node01 `/scratch/xzh` Deployment

On 2026-07-08, the project was staged on `node01` under `/scratch/xzh`.

Layout:

- code: `/scratch/xzh/code/structures25`
- environment: `/scratch/xzh/envs/structures25`
- data root: `/scratch/xzh/data`
- model root: `/scratch/xzh/models`
- setup notes: `/scratch/xzh/README_qm9_setup.md`

Environment:

- Python `3.11.15`
- PyTorch `2.4.1+cu121`
- Lightning `2.5.0`
- PyG extensions installed from `torch-2.4.1+cu124` wheels
- PySCF `2.4.0`
- CUDA visible through PyTorch; wrapper defaults to GPUs `0,1,2,3`
- 2-GPU standalone NCCL all-reduce smoke passed.

Data staged:

- `/scratch/xzh/data/QM9PBEForcePilot`
- size: about `7.7G`
- labels: `1640`
- `split.pkl`: present

Full data status:

- `/scratch/xzh/data/QM9PBEForceFull` is not present yet.
- Do not symlink pilot data to `QM9PBEForceFull`; full training should fail fast unless the real full dataset is copied.
- QM9 raw xyz files were copied to `/scratch/xzh/data/QM9/raw`; count is `133885`.
- Raw xyz files are not sufficient for EG/EGF training; `QM9PBEForceFull` labels, split, and dataset statistics are still required.
- Full label count target is `133885 * 4 = 535540` because each molecule has reference + 3 perturbations.
- A full force-label preset exists at `configs/datagen/preset/qm9_pbe_force_full.yaml`.
- A shard launcher exists at `scripts/launch_qm9_full_force_labels_shard.sh START_IDX N_MOLECULES [NUM_PROCESSES]`.
- A Slurm-array template exists at `scripts/slurm_qm9_full_force_labels_array.sbatch`.
- The same full-label generation files were synced to `/scratch/xzh/code/structures25` and passed remote `bash -n` plus Hydra config expansion checks.

One-line remote launch wrappers:

```bash
/scratch/xzh/run_qm9_full_4gpu.sh egf 10
/scratch/xzh/run_qm9_full_4gpu.sh eg 10
```

P1 calibration/smoke wrapper:

```bash
/scratch/xzh/run_qm9_p1_calibration_4gpu.sh egf 20
```

Remote smoke results:

- EG 4-GPU P1 calibration, `max_steps=2`: passed.
- EGF 4-GPU P1 calibration, `max_steps=2`: passed and logged `train_loss/force_loss`.
- Smoke run dirs:
  - `/scratch/xzh/models/train/runs/remote_p1_eg_4gpu_2step_20260708_170525`
  - `/scratch/xzh/models/train/runs/remote_p1_egf_4gpu_2step_20260708_170635`

The 2-step throughput summaries are startup-dominated and should not be used for scheduling.

2026-07-08 node02 status:

- node02 has 8 x A100 visible, but GPU 1-6 were occupied by two long-running `flow_grpo` reward server processes.
- Because node02 was not actually free and `QM9PBEForceFull` was missing, full EG/EGF jobs were not submitted.
- Remote 8-GPU wrappers were prepared:
  - `/scratch/xzh/run_qm9_full_8gpu.sh`;
  - `/scratch/xzh/slurm_qm9_full_node02_eg.sbatch`;
  - `/scratch/xzh/slurm_qm9_full_node02_egf.sbatch`.
- Full-scale validation still needs a policy for validation subset size/frequency to avoid long validation stalls.
- Fixed-density Hessian autograd remains an optional checkpoint-after evaluation path, not part of training.

2026-07-08 node03 / full-label generation status:

- `node03` is not usable at this time: it is not listed by Slurm and direct SSH from the login node cannot resolve the hostname.
- No full label-generation or full EG/EGF training job was submitted on `node03`.
- Direct GPU checks show `node04`, `node05`, and `node06` each have 8 x A100 visible with zero GPU memory used.
- Based on P1 batches, 100 molecules / 400 samples took about `3.8`, `5.7`, and `7.9` hours with 4 processes. Extrapolated ideal 4-process full runtime is about `323` days before safety margin.
- With `node04-node06` and about `16-20` PySCF processes per node, ideal runtime is about `21-27` days; with 1.5-2.0x safety margin for larger molecules, retries, and I/O, budget roughly `32-54` days.
- Using all five registered nodes at about `20` processes per node would ideally be about `13` days; with safety margin, budget roughly `19-26` days.
- Storage projection from the P1-410 snapshot is about `2.1-2.5 TB` for KS/labels/transformed artifacts, so reserve `3-5 TB`.

2026-07-08 Figshare Hessian QM9 dataset status:

- User requested using `https://figshare.com/articles/dataset/b_Hessian_QM9_Dataset_b/26363959`.
- Metadata fetched locally:
  - DOI: `10.6084/m9.figshare.26363959.v4`;
  - license: `CC0`;
  - main archive: `hessian_qm9_DatasetDict.zip`, `6281831499` bytes;
  - four small `params_*.npz` files.
- Article description states the dataset contains `41645` optimized QM9 molecules at `omegaB97x/6-31G*`, with Hessians in vacuum, water, THF, and toluene.
- Expected Hugging Face dataset fields are `energy`, `positions`, `atomic_numbers`, `forces`, `frequencies`, `normal_modes`, `hessian`, and `label`.
- This dataset can support external force/Hessian/frequency benchmarking.
- It cannot directly replace `QM9PBEForceFull` for the current OFDFT EG/EGF objective because it does not provide PBE `.chk`, density coefficients, density-gradient labels, local-frame transformed labels, or the P1/P2 reference + 3 perturbation sample design.
- New document: `docs/qm9_figshare_hessian_dataset_adoption.md`.
- New downloader/audit script: `scripts/download_hessian_qm9_figshare.py`.
- `datasets>=2,<4` was added to `pyproject.toml`, but the active local/remote environments still need to be updated before `load_from_disk` audit.
- Keep this dataset under `/scratch/xzh/data/HessianQM9Figshare`; do not symlink it to `/scratch/xzh/data/QM9PBEForceFull`.

2026-07-08 node04 demo1000 status:

- A node04-only end-to-end demo pipeline was prepared but not submitted because SSH through the jump host currently times out during banner exchange.
- Demo dataset: `QM9PBEForceDemo1000`.
- Scope: `1000` molecules, reference + 3 perturbations, expected `4000` labels.
- Pipeline script: `scripts/launch_qm9_node04_demo1000_pipeline.sh`.
- Slurm wrapper: `scripts/slurm_qm9_node04_demo1000_pipeline.sbatch`.
- Plan document: `docs/qm9_node04_demo1000_plan.md`.
- Label generation uses `LABEL_NUM_PROCESSES=20`, `LABEL_NUM_THREADS=1`, and `MAX_MEMORY_PER_PROCESS=4000 MB`.
- Training uses 8GPU DDP with EG per-GPU batch `8` and EGF per-GPU batch `4`.
- Runtime estimate:
  - labelgen ideal: about `11.6 h`;
  - labelgen conservative: about `17-29 h`;
  - transform/statistics/training: about `1.5-6 h`;
  - Slurm wall time: `48 h`.
- Local validation passed:
  - shell syntax for both scripts;
  - Hydra datagen config expansion for `QM9PBEForceDemo1000`;
  - Hydra training config expansion with `data.dataset_name=QM9PBEForceDemo1000`.
