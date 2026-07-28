# QM9 Node04 Demo1000 Plan

Date: 2026-07-08

## Goal

Run a self-contained 1000 molecule demo on `node04`:

1. Generate PBE KS `.chk` files and labels with forces.
2. Validate force labels.
3. Create grouped train/val/test split.
4. Cache local-frame transformed labels.
5. Compute dataset statistics.
6. Train EG and EGF lambda=1.0 with 8 GPUs.

The demo dataset is named `QM9PBEForceDemo1000` and is separate from:

- `QM9PBEForcePilot`;
- `QM9PBEForceFull`;
- external Figshare Hessian QM9.

## Time Estimate

P1 historical timing for 100 molecules / 400 samples with 4 PySCF processes:

| checkpoint | KS + labelgen |
| --- | ---: |
| 110 molecule batch | `3.8 h` |
| 210 molecule batch | `5.7 h` |
| 310 molecule batch | `7.9 h` |

Average: `5.78 h / 100 molecules / 4 processes`.

For 1000 molecules with 20 PySCF processes:

```text
5.78 * 10 * 4 / 20 = 11.6 h ideal labelgen
```

Conservative budget:

- labelgen: `17-29 h`;
- transform/statistics: `0.5-2 h`;
- EG 10 epoch + EGF 10 epoch on 8 GPUs: `1-4 h`;
- recommended Slurm wall time: `48 h`.

Storage estimate from P1-410:

- raw KS/labels/transformed labels for 1000 molecules: roughly `18-20 GB`;
- reserve at least `50 GB` for logs, temporary files, and retries.

## Parallelism

Label generation:

- `LABEL_NUM_PROCESSES=20`;
- `LABEL_NUM_THREADS=1`;
- `MAX_MEMORY_PER_PROCESS=4000 MB`;
- total PySCF memory cap about `80 GB` on a `103 GB` node;
- Slurm memory request is `95G`, leaving some headroom for OS and process overhead.

Training:

- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`;
- `NPROC_PER_NODE=8`;
- EG per-GPU batch: `8`;
- EGF per-GPU batch: `4`;
- train dataloader workers per rank: `4`.

## Files

Pipeline:

```bash
scripts/launch_qm9_node04_demo1000_pipeline.sh
```

Slurm wrapper:

```bash
scripts/slurm_qm9_node04_demo1000_pipeline.sbatch
```

Submit when remote SSH is stable:

```bash
ssh -J root@101.200.216.125 -p 2200 shenwei01@localhost \
  'cd /scratch/xzh/code/structures25 && sbatch scripts/slurm_qm9_node04_demo1000_pipeline.sbatch'
```

Monitor:

```bash
ssh -J root@101.200.216.125 -p 2200 shenwei01@localhost \
  'squeue -u shenwei01; tail -f /scratch/xzh/logs/qm9_demo1000/<JOBID>.out'
```

## Current Status

Prepared locally:

- node04 demo pipeline script;
- node04 Slurm wrapper;
- plan document.

Blocked:

- remote SSH currently times out during banner exchange, so files have not yet been synced to `/scratch/xzh/code/structures25`;
- no node04 job has been submitted yet.
