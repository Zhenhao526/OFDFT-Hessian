# QM9 Node04 Random1000 Force Label Generation

Date: 2026-07-09

## Goal

Use `node04` only to generate PBE force labels for a random subset of 1000 QM9 molecules.

This task does not launch EG/EGF training.

## Dataset

- output dataset: `QM9PBEForceRandom1000`
- random seed: `20260709`
- source raw data: `/scratch/xzh/data/QM9/raw`
- random subset raw symlink dir: `/scratch/xzh/data/QM9RandomSubsets/random1000_seed20260709/raw`
- samples per molecule: reference + 3 perturbations
- expected labels: `1000 * 4 = 4000`

The subset preserves original QM9 file names so labels keep original molecule ids.

## Runtime Estimate

Based on P1 historical batches, 100 molecules / 400 samples took on average:

```text
5.78 h with 4 PySCF processes
```

For 1000 molecules with 20 processes:

```text
5.78 * 10 * 4 / 20 = 11.6 h ideal
```

Conservative budget:

- expected practical runtime: `17-29 h`
- Slurm wall time: `48 h`

## Parallelism

- `LABEL_NUM_PROCESSES=20`
- `LABEL_NUM_THREADS=1`
- `MAX_MEMORY_PER_PROCESS=4000 MB`
- Slurm memory request: `95G`

## Files

- `scripts/prepare_qm9_random_subset.py`
- `scripts/launch_qm9_node04_random1000_labels.sh`
- `scripts/slurm_qm9_node04_random1000_labels.sbatch`

## Submit

```bash
ssh -J root@101.200.216.125 -p 2200 shenwei01@localhost \
  'cd /scratch/xzh/code/structures25 && sbatch scripts/slurm_qm9_node04_random1000_labels.sbatch'
```

## Outputs

Expected output roots:

```text
/scratch/xzh/data/QM9PBEForceRandom1000/kohn_sham
/scratch/xzh/data/QM9PBEForceRandom1000/labels
/scratch/xzh/models/labelgen/runs/QM9PBEForceRandom1000_<timestamp>
```

The run directory contains:

- random subset manifest;
- per-stage logs and `/usr/bin/time -v` files;
- force check summary JSON;
- artifact file list and size summary.

## 2026-07-09 Launch Record

Remote environment issue:

- First Slurm attempt `422` failed immediately with exit code `127` because `python` was not available on `node04`.
- Root cause: `/scratch/xzh/envs/structures25/bin/python` pointed to a uv Python under node01-local `/home/shenwei01/.local/...`, which was not visible from `node04`.
- Fix: copied the CPython 3.11 runtime to `/scratch/xzh/uv-python/cpython-3.11-linux-x86_64-gnu` and repointed the venv `python`, `python3`, and `python3.11` symlinks to the `/scratch` runtime.
- Verified on `node04`: `source /scratch/xzh/env.sh && python -V` reports `Python 3.11.15`.

Relaunched job:

- Slurm job id: `423`
- Node: `node04`
- Run root: `/scratch/xzh/models/labelgen/runs/QM9PBEForceRandom1000_20260709_103746`
- Slurm stdout: `/scratch/xzh/logs/qm9_random1000_labels/423.out`
- Slurm stderr: `/scratch/xzh/logs/qm9_random1000_labels/423.err`
- Random subset manifest: `/scratch/xzh/models/labelgen/runs/QM9PBEForceRandom1000_20260709_103746/random_subset_manifest.json`

Initial status:

- Random subset creation completed.
- `linked_count=1000`, `missing_ids=[]`, `extra_ids=[]`.
- `01_kohn_sham` stage started with 20 worker processes.

## 2026-07-13 Final Status

Slurm job `423` completed successfully:

```text
state: COMPLETED
exit code: 0:0
node: node04
start: 2026-07-09T10:37:46
end: 2026-07-11T10:05:05
elapsed: 1-23:27:19
```

Final artifact counts:

```text
random subset raw symlinks: 1000
.chk files: 4000
label .zarr.zip files: 4000
```

Output sizes:

```text
/scratch/xzh/data/QM9PBEForceRandom1000: 40G
/scratch/xzh/models/labelgen/runs/QM9PBEForceRandom1000_20260709_103746: 6.4M
```

Force-label validation passed:

```json
{
  "checked_files": 4000,
  "expected_molecules": 1000,
  "expected_samples": 4,
  "failures": [],
  "force_labels": 4000,
  "force_norm_max": 0.13995190142033215,
  "molecules": 1000
}
```

Stage completion timestamps:

```text
00_prepare_random_subset: 2026-07-09 10:37
01_kohn_sham:             2026-07-10 19:22
02_labelgen:              2026-07-11 10:01
03_force_check:           2026-07-11 10:05
```

The job used almost the full 48 h wall-time budget. The runtime was much longer than the initial ideal estimate because the random subset included larger and fluorinated high-index QM9 molecules. Future random-1000 force-label jobs should request at least 60-72 h wall time, or increase parallelism after confirming memory headroom.

This task generated labels only. No EG/EGF training was launched in this job.
