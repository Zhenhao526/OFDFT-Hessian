# GPU4PySCF Force Label Migration

Date: 2026-07-13

## Scope

This note describes the GPU path for generating the extra force labels needed by EGF training.

The original EG labels remain unchanged:

- density coefficients;
- energies;
- density-gradient labels.

The EGF-specific addition is:

```text
metadata/pbe_derivatives/forces
metadata/pbe_derivatives/nuclear_gradient
```

with:

```text
forces = - nuclear_gradient = - dE_PBE / dR
```

## New GPU Force Appender

New script:

```bash
scripts/append_gpu4pyscf_force_labels.py
```

It reads existing `.zarr.zip` labels, rebuilds the PySCF molecule from `geometry/*`, runs
same-level PBE/6-31G(2df,p), computes the analytic nuclear gradient, and appends force metadata
back into the label file.

This is intentionally force-only. It does not regenerate the original OFDFT label arrays and does
not alter `of_labels/*`, `ks_labels/*`, or `geometry/*`.

## Backend

Default backend:

```text
gpu4pyscf
```

CPU backend is kept for A/B validation:

```bash
python scripts/append_gpu4pyscf_force_labels.py /path/to/labels \
  --backend cpu \
  --max-labels 10 \
  --summary-json /tmp/cpu_force_summary.json
```

GPU backend:

```bash
python scripts/append_gpu4pyscf_force_labels.py /path/to/labels \
  --backend gpu4pyscf \
  --num-processes 8 \
  --gpu-device-ids 0,1,2,3,4,5,6,7 \
  --summary-json /tmp/gpu_force_summary.json \
  --records-jsonl /tmp/gpu_force_records.jsonl
```

The script fails fast if `gpu4pyscf` is requested but not installed.
PySCF output is quiet by default through `--pyscf-verbose 0`; raise it only for debugging.

The appender explicitly mirrors the original local `ksdft()` SCF defaults:

```text
max_cycle: 50
diis_start_cycle: 0
diis_space: 8
convergence_tolerance: 1e-9
```

## GPU Worker Isolation

GPU4PySCF can internally use every CUDA device visible to a process. For this label-generation
workflow, each worker process should own exactly one GPU. The appender therefore treats
`--gpu-device-ids` as visible-device indices from the parent process and narrows each worker's
`CUDA_VISIBLE_DEVICES` before importing CuPy/GPU4PySCF.

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python scripts/append_gpu4pyscf_force_labels.py /path/to/labels \
  --backend gpu4pyscf \
  --num-processes 2 \
  --gpu-device-ids 0,1
```

Worker 0 sees only physical GPU 0 and records `gpu_device_id=0`; worker 1 sees only physical GPU 1
and records `gpu_device_id=1`. This avoids GPU4PySCF's internal multi-GPU path being triggered by
multiple independent label workers.

Use one worker per GPU. The script rejects `--num-processes` values larger than the number of
`--gpu-device-ids` entries for the GPU backend.

## Geometry Unit and Checkpoint Fallback

Default geometry handling:

```text
--geometry-unit auto
```

When a matching Kohn-Sham `.chk` file is available, `auto` compares `geometry/atom_pos` against the
checkpoint molecule in both Angstrom and Bohr and uses the matching unit. Without a checkpoint it
falls back to Angstrom.

Checkpoint lookup is either explicit:

```bash
python scripts/append_gpu4pyscf_force_labels.py /path/to/labels \
  --kohn-sham-dir /path/to/kohn_sham
```

or implicit when the label directory has a sibling `kohn_sham` directory.

The appender can also use force derivatives already stored in matching checkpoints:

```text
--chk-derivatives-mode off       # default, pure PySCF/GPU4PySCF run
--chk-derivatives-mode fallback  # use chk Derivatives/forces only after SCF/gradient failure
--chk-derivatives-mode prefer    # read chk Derivatives/forces before SCF
```

Fallback records are explicitly marked with:

```text
metadata/pbe_derivatives/backend = chk_derivatives
metadata/pbe_derivatives/source_backend = gpu4pyscf
metadata/pbe_derivatives/fallback_reason = ...
metadata/pbe_derivatives/kohn_sham_chk = ...
```

This keeps GPU-generated labels distinguishable from checkpoint-recovered labels.

## Node Wrapper

New wrapper:

```bash
scripts/launch_gpu4pyscf_force_labels.sh
```

Example for an existing dataset on `/scratch`:

```bash
DFT_DATA=/scratch/xzh/data \
DFT_MODELS=/scratch/xzh/models \
DATASET_NAME=QM9PBEForceRandom1000 \
NUM_PROCESSES=8 \
GPU_DEVICE_IDS=0,1,2,3,4,5,6,7 \
GEOMETRY_UNIT=auto \
CHK_DERIVATIVES_MODE=off \
scripts/launch_gpu4pyscf_force_labels.sh --overwrite
```

This writes:

```text
${DFT_MODELS}/force_labels/gpu4pyscf_${DATASET_NAME}_*/summary.json
${DFT_MODELS}/force_labels/gpu4pyscf_${DATASET_NAME}_*/records.jsonl
```

## Installation

GPU4PySCF must match the CUDA environment. For CUDA 12 systems, the upstream project documents:

```bash
pip install gpu4pyscf-cuda12x cutensor-cu12
```

Use the CUDA 11 or CUDA 13 packages when the node uses those CUDA runtimes.

Local validation environment after installing CUDA 12 packages:

```text
gpu4pyscf-cuda12x==1.7.5
gpu4pyscf-libxc-cuda12x==0.8.1
cupy-cuda12x==14.1.1
cutensor-cu12==2.7.0
pyscf==2.13.1
```

Note: installing GPU4PySCF upgraded the local `.venv` PySCF from the project pin `2.4.0` to
`2.13.1`. Treat the GPU force-label environment as a separate validation/runtime environment until
the broader codebase has been checked against PySCF 2.13.

## Current Smoke Results

One copied QM9 P1 label was tested on GPU 5 through:

```bash
CUDA_VISIBLE_DEVICES=5 \
.venv/bin/python scripts/append_gpu4pyscf_force_labels.py \
  /tmp/gpu4pyscf_force_ab/gpu_labels \
  --backend gpu4pyscf \
  --gpu-device-ids 0 \
  --num-processes 1 \
  --overwrite
```

Result:

```text
written_force_labels: 1
failures: []
force_norm_max: 0.058549889320111355
```

CPU vs GPU comparison on the same copied label:

```text
force_component_mae: 4.346000770493682e-08
force_component_rmse: 5.9093813036559844e-08
force_component_max_abs: 1.1789303722142819e-07
```

`scripts/check_qm9_force_smoke.py` passed on the GPU-written label:

```text
checked_files: 1
force_labels: 1
failures: []
```

The smoke used a label that already contained force metadata, so `ZipStore` emitted duplicate-name
warnings during overwrite. This is expected for overwrite tests on zip labels. The intended use case
is appending force metadata to original EG labels that do not yet contain
`metadata/pbe_derivatives/forces`.

The intended append-only case was also tested with a minimal copied label containing only
`geometry/*` and no existing force metadata:

```text
written_force_labels: 1
failures: []
force_norm_max: 0.058549889320111466
```

`scripts/check_qm9_force_smoke.py` passed on this append-only output:

```text
checked_files: 1
force_labels: 1
failures: []
```

A two-worker append-only GPU smoke also passed after adding per-worker single-GPU isolation:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
.venv/bin/python scripts/append_gpu4pyscf_force_labels.py \
  /tmp/gpu4pyscf_force_multismoke_easy/labels \
  --backend gpu4pyscf \
  --gpu-device-ids 0,1 \
  --num-processes 2
```

Result:

```text
checked_files: 2
written_force_labels: 2
failures: []
force_norm_max: 0.06684068119218466
```

The written metadata confirmed one process per GPU:

```text
0000001.0000000.zarr.zip gpu_device_id=0 cuda_visible_devices=0 forces_shape=(5, 3)
0000002.0000000.zarr.zip gpu_device_id=1 cuda_visible_devices=1 forces_shape=(4, 3)
```

`scripts/check_qm9_force_smoke.py` passed on the two-label output:

```text
checked_files: 2
force_labels: 2
failures: []
```

Earlier, running two GPU workers while both processes could see both GPUs produced GPU4PySCF
internal CUDA errors. The per-worker `CUDA_VISIBLE_DEVICES` narrowing is therefore required for
multi-process production runs.

A final 8-worker / 8-GPU append-only smoke was run on a fresh copied scratch shard with 8 labels:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/python scripts/append_gpu4pyscf_force_labels.py \
  /tmp/gpu4pyscf_force_8gpu_smoke_final/labels \
  --backend gpu4pyscf \
  --gpu-device-ids 0,1,2,3,4,5,6,7 \
  --num-processes 8 \
  --kohn-sham-dir _runtime/qm9_p1/QM9PBEForcePilot/kohn_sham \
  --chk-derivatives-mode fallback
```

Result:

```text
checked_files: 8
written_force_labels: 8
failures: []
force_norm_max: 0.029857166815255805
```

Source breakdown:

```text
gpu4pyscf: 7
chk_derivatives fallback: 1
```

The one fallback was `0000004.0000000.zarr.zip`, where GPU4PySCF SCF did not converge under the
same nominal settings in the current PySCF/GPU4PySCF environment. The fallback copied
`Derivatives/forces` from the matching historical checkpoint and marked the label metadata with
`backend=chk_derivatives`.

The 8-label output passed `scripts/check_qm9_force_smoke.py`:

```text
checked_files: 8
force_labels: 8
failures: []
```

Against the original P1 force labels on the same 8 files:

```text
force_component_mae: 1.7862967592639037e-07
force_component_rmse: 4.990640642702384e-07
force_component_max_abs: 3.0911579287362656e-06
```

This smoke also exposed that some P1 pilot labels store `geometry/atom_pos` in a coordinate unit
that matches the checkpoint Bohr coordinates. `--geometry-unit auto` with a matching checkpoint is
therefore required for this legacy P1 shard.

## Validation Plan

Before production use:

1. Run CPU backend on 5-10 labels into a copied scratch label directory.
2. Run GPU backend on the same copied labels with `--overwrite`.
3. Compare `metadata/pbe_derivatives/forces` between CPU and GPU outputs:

```bash
python scripts/compare_force_label_dirs.py \
  /path/to/cpu_labels \
  /path/to/gpu_labels \
  --summary-json /tmp/force_cpu_gpu_compare.json
```

4. Confirm `scripts/check_qm9_force_smoke.py` passes.
5. Run a small multi-process smoke with the intended number of visible GPUs.
6. Only then run the multi-GPU appender on the target label directory.

The current implementation has passed one-label GPU correctness, CPU/GPU force comparison on one
label, append-only single-label generation, append-only two-worker/two-GPU generation, and an
8-worker/8-GPU scratch smoke on 8 labels. It has not yet been validated as a full production shard
over thousands of labels.

Some labels may still fail because the underlying PBE SCF calculation does not converge within
`--max-cycle`. Those failures are recorded in `records.jsonl`; retry them with a larger
`--max-cycle`, a different initialization, CPU PySCF, or `--chk-derivatives-mode fallback` when a
matching checkpoint already contains trusted `Derivatives/forces`.
