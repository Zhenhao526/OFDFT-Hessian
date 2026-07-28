# QM9 node02 Local Rehome (2026-07-24)

## Decision

The degraded BeeGFS `/scratch` tree is no longer part of the active execution
path. Code, environments, inputs, outputs, caches, and temporary files for the
next run must use node02's local ext4 filesystem.

No file was recovered from or written to `/scratch` during this rehome.

## node02 Resources

Local root:

```text
/home/shenwei01/xzh_node02_20260724
```

Observed resources:

- local filesystem: 914 GiB total, 776 GiB free before migration and 743 GiB
  free after the environment, runtime mirror, and smoke outputs;
- GPUs: 8 x NVIDIA A100-SXM4-80GB;
- RAM: about 1 TiB;
- GPU availability at setup time: all eight visible, zero allocated memory.

## Source Copies

Two separate source copies are retained:

| purpose | path | state |
| --- | --- | --- |
| clean baseline | `/home/shenwei01/xzh_node02_20260724/github/structures25` | clean clone at `a6c4c0b0e73cad5d3b89dbf8d6194b274c6cc4b6` |
| active work | `/home/shenwei01/xzh_node02_20260724/work/structures25` | exact local WIP, including uncommitted project work |

The clean clone was reconstructed from the local verified Git object database
because the direct HTTPS clone stalled. Its remote remains
`https://github.com/sciai-lab/structures25.git`, and `git fsck --full
--no-reflogs` passes.

Verification:

```text
clean HEAD:      a6c4c0b0e73cad5d3b89dbf8d6194b274c6cc4b6
origin/main:     a6c4c0b0e73cad5d3b89dbf8d6194b274c6cc4b6
Git bundle hash: 749f6ec823eb4dbe6d82f783471958f5081c7066773dcb9a2512d996a285dec2
```

The active work copy is the execution tree. Do not replace it with the clean
baseline: the current force/Hessian implementation and reports are mostly
uncommitted work.

## Environment

The environment is entirely local to node02:

```text
Python: /home/shenwei01/xzh_node02_20260724/python
uv:     /home/shenwei01/xzh_node02_20260724/tools/uv
venv:   /home/shenwei01/xzh_node02_20260724/work/structures25/.venv
cache:  /home/shenwei01/xzh_node02_20260724/cache
tmp:    /home/shenwei01/xzh_node02_20260724/tmp
```

Installed from `uv.lock` with Python 3.11.15. Runtime checks passed for:

- PyTorch 2.4.1 with CUDA 12.1;
- all eight A100 GPUs and a real float64 CUDA operation;
- PySCF 2.4.0;
- zarr 2.18.4;
- PyG 2.6.1 CUDA extensions;
- tensorframes at the pinned Git commit.

`setuptools<81` is explicitly required because the pinned torchmetrics 0.x
stack still imports `pkg_resources`, which setuptools 82 removed.

Activate the local paths with:

```bash
source /home/shenwei01/xzh_node02_20260724/work/structures25/scripts/activate_qm9_node02_local.sh
cd "${QM9_CODE_ROOT}"
```

## Local Runtime Mirror

The local workstation `_runtime` snapshot is mirrored under:

```text
/home/shenwei01/xzh_node02_20260724/runtime_parent/_runtime
```

This snapshot includes the P1-410 labels, P1 model runs, Hessian references,
the locally retained Graphformer checkpoints, P0/HORM artifacts, and reports.
The transfer is independent of BeeGFS.

Full source/destination verification passed:

```text
regular files:            144455
regular-file bytes:       18765821111
content manifest SHA256:  8dc8edb789bd039aa553ce6fc3a2ed24898de822a1a28d53c2df8bf211698fb9
symlink manifest SHA256:  e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

The last value is the SHA256 of an empty symlink manifest: neither side
contains symlinks in `_runtime`.

The node02 P1-410 inventory contains 1,640 `.chk` files, 1,640 raw labels,
1,640 cached labels, `split.pkl`, and dataset statistics. A complete label
scan found 410 parents x 4 samples, 1,640 force labels, no non-finite arrays,
and no force-shape failures. Its main EG/EGF checkpoints and 20-molecule PBE
Hessian references are available locally.

Key checkpoint hashes match the source:

```text
EG_s3000:       52783661970aa9b6eea3668fa991c92423f6ae1397913a5de7d927ab7328ff19
EGF_lam1_s3000: 3f2b7d1e5cdeeb5b41dc6c5a3fb8104dbc4a1bd05d66fe5d3ee5d4afbb48f5e2
local Graphformer checkpoint:
                 9759da26660c619de9c3bbf4c2dc164343ee90e08e22b3fcdcc9682dacb9bd09
```

## Assets Not Recovered

The local snapshot does not contain a complete copy of:

- `QM9PBEForceRandom1000` (the remote version was about 40 GiB);
- the full 535k-label QM9 dataset;
- the registered original-A Graphformer checkpoint with SHA256
  `e6516b04917a9dfaae3c4d960b10d77c768f3e5e784f95290ff41c1202f9d9bc`;
- the registered bound step-20 Graphformer checkpoint with SHA256
  `c18d4352584379b2f3607b0057fa633305b0ee8994d705fef060d33852c0876b`;
- the complete stable5/train20 relaxed-HVP direction/reference trees.

The local `trained-on-qm9/last.ckpt` is a different artifact (SHA256
`9759da26660c619de9c3bbf4c2dc164343ee90e08e22b3fcdcc9682dacb9bd09`)
and must not be silently substituted for registered original-A.

Consequences:

- P1-410 smoke/retraining can be prepared from the mirrored local snapshot;
- clean code and CUDA tests can run immediately;
- random1000 and full-QM9 training need datasets generated or restored to
  node02 local storage first;
- the strict Graphformer complete-total relaxed-HVP continuation remains
  fail-closed until its hash-bound starting checkpoint and direction/reference
  assets have an independent non-BeeGFS copy.

## Verification

Current code checks on node02:

```text
py_compile latest HVP modules/tests: pass
internal-direction and complete-total training tests: 23 passed
MLDFT EG/EGF/HVP model tests: 9 passed
checkpoint save/resume test: 1 passed
uv dependency check: 268 packages compatible
```

The active WIP source also matches the local workspace exactly for 1,106
source/config/test/doc/root files. The current per-file SHA256 manifest is
stored outside the repository at
`/home/shenwei01/xzh_node02_20260724/work/source_manifest.sha256`.

P1 single-GPU training-path smoke runs completed on GPU 1:

| arm | steps | result | output |
| --- | ---: | --- | --- |
| EG | 2 | pass | `${DFT_MODELS}/train/runs/qm9_node02_rehome_eg_smoke_2steps_20260724` |
| EGF lambda=1 | 2 | pass, finite `force_loss` logged | `${DFT_MODELS}/train/runs/qm9_node02_rehome_egf_smoke_2steps_20260724` |

These two-step throughput numbers are startup-dominated and are not valid
full-training estimates.

The newest internal Hutchinson/Rademacher direction code is still a work in
progress. Passing unit tests does not establish the requested analytic KKT HVP
audit, parameter-gradient finite-difference gate, or tenfold performance gate.
Do not start stable5/train20 from this WIP.

## Current GPU Blocker

At the final check, Slurm reported node02 as idle, but five root-owned
non-Slurm GROMACS processes occupied GPUs 0, 3, 4, 5, and 6. GPUs 1, 2, and 7
were free. This scheduler/runtime mismatch means:

- single-GPU smoke on an explicitly checked free GPU is possible;
- an eight-GPU job is not currently safe;
- recheck `nvidia-smi --query-compute-apps` immediately before submission;
- do not terminate or preempt the external processes.

## Start Policy

Before any new run:

1. source `scripts/activate_qm9_node02_local.sh`;
2. confirm `hostname -s` is `node02`;
3. confirm inputs and starting checkpoint against a recorded SHA256 manifest;
4. confirm all output, cache, and temporary paths begin with
   `/home/shenwei01/xzh_node02_20260724`;
5. confirm the intended GPUs are idle;
6. run the smallest valid smoke for that experiment;
7. only then launch the formal job.

No active command may use `/scratch` as a fallback.
