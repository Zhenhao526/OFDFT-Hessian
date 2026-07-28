# QM9 Scratch Integrity Audit (2026-07-24)

## Scope

This audit follows `docs/qm9_force_hessian_project_handoff.md` and checks the
current project code, the remote `/scratch/xzh` workspace, and the artifacts
needed to resume the Graphformer complete-total relaxed-HVP work. It is
read-only for all existing labels, checkpoints, references, and evaluation
outputs.

## Executive Result

The local source workspace is intact. The remote shared BeeGFS filesystem is
not currently healthy enough for a complete file-level audit or training:

- storage targets 5 and 6 on `node_storage_3` are `Offline / Good`;
- all other storage targets are `Online / Good`;
- node04 is down and node01 is drained with `Kill task failed`;
- reads or creates that touch the unavailable targets can block in D-state;
- directory existence therefore does not certify file readability.

Do not launch training from `/scratch/xzh` until targets 5 and 6 are online and
the checks below are rerun.

## Local Source Audit

Workspace:

```text
/mnt/afs/home/xiazhenhao/dft/structures25
```

Checks completed:

- `git fsck --full --no-reflogs`: passed;
- `/usr/bin/python3 -m compileall -q mldft scripts tests`: passed;
- broken symlinks under `mldft`, `scripts`, `configs`, `tests`, and `docs`: 0;
- unexpected zero-length source/config/report files: 0.

The zero-length files found are intentional package markers or the registered
empty callback config:

```text
configs/ml/callbacks/none.yaml
mldft/__init__.py
mldft/utils/log_utils/__init__.py
tests/__init__.py
tests/helpers/__init__.py
tests/ml/__init__.py
tests/utils/__init__.py
```

The workspace is intentionally dirty and contains the accumulated project
work. It must not be replaced by a clean upstream checkout.

## Remote Filesystem State

Observed through the jump host and node01 on 2026-07-24:

```text
target 1-4:   Online / Good
target 5-6:   Offline / Good
target 7-12:  Online / Good
node_storage_3 connections: none
```

The target status is available without touching BeeGFS file data:

```bash
cat /proc/fs/beegfs/c22801-6A1925D1-node01/storage_target_state
cat /proc/fs/beegfs/c22801-6A1925D1-node01/storage_nodes
```

File probes demonstrated the failure mode:

- `/scratch/xzh/envs/structures25/bin/python` metadata resolution blocked;
- a checksum scan of the old code tree blocked;
- creating and populating a new recovery tree under `/scratch/xzh` also
  produced communication errors or blocked metadata operations.

Their client sessions were disconnected and termination signals were sent.
Any uninterruptible kernel wait must be allowed to clear when the storage
target returns; repeated probes only create more D-state tasks.

At the final process check, eight D-state entries were visible. Five are
bounded probes from this audit whose termination signals cannot complete while
the target is offline: the old-code checksum receiver, environment `readlink`,
the first recovery-tree receiver, one checkpoint `sha256sum`, and one recovery
`mkdir`. The remaining entries are a long-lived BeeGFS flusher and processes
from other workspaces/shells. Do not submit further `/scratch` probes to try to
clear them; storage recovery is the required event.

## Verified Critical Artifacts

The following artifacts were fully read and matched their registered SHA256:

| artifact | size | SHA256 status |
| --- | ---: | --- |
| stable5 direction manifest | 7,539 B | matches `1d9d235719805b3bf298fe3baf783b1b17a262e352b4193388f0be4f9b65bd74` |
| train20 direction manifest | 24,948 B | matches `a902bb554cf05393c14b316e091f182c19bf888317c42aec153da774f02af378` |
| bound Graphformer step-20 checkpoint | 657,975,874 B | matches `c18d4352584379b2f3607b0057fa633305b0ee8994d705fef060d33852c0876b` |

The original-A checkpoint path exists and reports size 224,572,200 B, but its
full SHA256 read did not finish within the bounded audit window. Its registered
hash remains:

```text
e6516b04917a9dfaae3c4d960b10d77c768f3e5e784f95290ff41c1202f9d9bc
```

It is **unverified under the current storage outage**, not declared corrupt.

The following directories are visible, but their complete contents were not
certified because recursive reads can block:

```text
/scratch/xzh/data/QM9PBEForcePilot
/scratch/xzh/data/QM9PBEForceRandom1000
/scratch/xzh/models/eval/qm9_random1000_pbe_hessian_refs/test100_pbe_hessians
/scratch/xzh/models/eval/qm9_random1000_test100_hessian_node04_gpu4pyscf/20260714_135600
/scratch/xzh/models/eval/qm9_random1000_test100_density_relaxed_hessian_node04/20260714_150546
```

Do not infer label/reference completeness from those directory entries.

## Non-BeeGFS Recovery Copy

Because new writes to `/scratch` are not reliable, an exact code-only recovery
copy was staged on node01 local XFS:

```text
/home/shenwei01/xzh_recovery_20260724/structures25
```

Verification:

- 1,208 entries in the checksum comparison, including 1,129 regular files;
- total compared regular-file size: 10,252,647 B;
- checksum dry-run: 0 created, 0 deleted, 0 transferred;
- remote `git fsck --full --no-reflogs`: passed;
- remote Python compileall for `mldft`, `scripts`, and `tests`: passed.

This is a code/config/test/document recovery copy only. It contains the current
WIP analytic relaxed-HVP refactor and must not be treated as a validated
training release. It does not contain datasets, checkpoints, the Python 3.11
environment, or scientific outputs.

The attempted paths below are incomplete and must not be used:

```text
/scratch/xzh/recovery_20260724/structures25
/scratch/xzh/recovery_20260724/structures25_v2
/scratch/xzh/recovery_20260724/structures25_source_20260724.tar.gz
```

They are preserved for audit and should only be removed after BeeGFS service
is restored and no process still references them.

## Current Training State

- no Slurm jobs are active for this user;
- job 4066 is canceled, not active;
- it produced valid logs through cumulative step 25 but no newer accepted
  checkpoint than the bound step-20 file above;
- the attempted larger-learning-rate job did not start Python;
- no stable5 or train20 training may start from the partial recovery trees.

## Required Recovery Sequence

1. Restore `node_storage_3` or otherwise return storage targets 5 and 6 to
   `Online / Good`.
2. Confirm D-state tasks have cleared and undrain node01 only after the
   `Kill task failed` condition is resolved.
3. Recheck the original-A checkpoint SHA256 and the remote environment
   interpreter before any run.
4. Run bounded per-tree inventories for code, labels, PBE references,
   Graphformer manifests/checkpoints, and node04 evaluation outputs.
5. Compare record counts and registered hashes; restore only missing or
   unreadable files from an independent source.
6. Sync the verified node01-local code copy to a fresh `/scratch` code path,
   then run a checksum dry-run before changing launchers.
7. Rebuild or restore a Python 3.11 environment and verify at least:

   ```bash
   python -c "import torch, pyscf, zarr, hydra; print(torch.__version__)"
   ```

8. Only then resume the single-molecule analytic relaxed-HVP correctness and
   performance audit.

No labels, checkpoints, references, or existing evaluation outputs were
deleted or overwritten during this audit.
