# Hessian QM9 Figshare Dataset Adoption

Date: 2026-07-08

## Source

Dataset:

- Figshare article: `https://figshare.com/articles/dataset/b_Hessian_QM9_Dataset_b/26363959`
- DOI: `10.6084/m9.figshare.26363959.v4`
- License: `CC0`

Figshare metadata was fetched through:

```bash
scripts/download_hessian_qm9_figshare.py --metadata-only
```

Manifest:

```bash
_runtime/external_datasets/HessianQM9Figshare/figshare_manifest.json
```

## Files

Figshare API reports:

| file | size |
| --- | ---: |
| `hessian_qm9_DatasetDict.zip` | `6281831499` bytes |
| `params_thf_f128_i5_b16.npz` | `3175111` bytes |
| `params_vacuum_f128_i5_b16.npz` | `3175111` bytes |
| `params_toluene_f128_i5_b16.npz` | `3175111` bytes |
| `params_water_f128_i5_b16.npz` | `3175111` bytes |

## Dataset Definition

The article description states that Hessian QM9 contains `41645` optimized QM9 molecules at the
`omegaB97x/6-31G*` level. Hessians were calculated in:

- vacuum;
- water;
- tetrahydrofuran;
- toluene.

The dataset is stored in Hugging Face `datasets` format. Expected fields are:

```text
energy
positions
atomic_numbers
forces
frequencies
normal_modes
hessian
label
```

Expected splits/environments:

```text
vacuum: 41645 rows
thf: 41645 rows
toluene: 41645 rows
water: 41645 rows
```

Only H/C/N/O molecules are included.

## Relationship To Current OFDFT EG/EGF Pipeline

This Figshare dataset is not a drop-in replacement for `QM9PBEForceFull`.

Current full-scale OFDFT training expects:

- PBE KS/OFDFT-derived labels;
- density coefficients and density-gradient labels;
- local-frame transformed labels and dataset statistics;
- molecule reference + 3 perturbation samples;
- current objective EG: energy + density-gradient;
- current objective EGF: energy + density-gradient + force.

The Figshare dataset provides:

- optimized geometries;
- energy;
- force;
- Hessian;
- vibrational frequencies and normal modes;
- labels matching QM9 molecule ids.

It does not provide:

- PBE `.chk` files;
- OFDFT density coefficients;
- density-gradient labels;
- the P1/P2 perturbation geometry set;
- the same functional/basis as the current PBE pilot.

Therefore it should be kept under a separate external dataset root and used for one of these purposes:

- external force/Hessian benchmark;
- Hessian/frequency evaluation pipeline validation;
- training a separate energy-force-Hessian atomistic model;
- a future OFDFT variant only if the training objective is changed or density labels are generated separately.

It should not be symlinked to `QM9PBEForceFull`.

## Local Integration

New script:

```bash
scripts/download_hessian_qm9_figshare.py
```

Metadata-only check:

```bash
.venv/bin/python scripts/download_hessian_qm9_figshare.py \
  --metadata-only \
  --output-dir _runtime/external_datasets/HessianQM9Figshare
```

Download and extract when network access is stable:

```bash
DFT_DATA=/scratch/xzh/data \
/scratch/xzh/envs/structures25/bin/python /scratch/xzh/code/structures25/scripts/download_hessian_qm9_figshare.py \
  --download-archive \
  --extract \
  --audit \
  --output-dir /scratch/xzh/data/HessianQM9Figshare
```

Individual file download is also supported:

```bash
DFT_DATA=/scratch/xzh/data \
/scratch/xzh/envs/structures25/bin/python /scratch/xzh/code/structures25/scripts/download_hessian_qm9_figshare.py \
  --download-files \
  --audit \
  --output-dir /scratch/xzh/data/HessianQM9Figshare
```

The audit step needs the Python package `datasets`.

## Current Status

Completed:

- Figshare metadata fetched locally.
- Manifest written locally.
- Download/audit script added and `py_compile` checked.
- `datasets>=2,<4` added to project dependencies for Hugging Face dataset loading.

Not completed:

- Current local `.venv` has not installed `datasets` yet.
- Full 6.28 GB archive has not been downloaded yet.
- Dataset has not been extracted or audited with `load_from_disk`.
- No conversion into OFDFT label format exists.
- No full EG/EGF training has been launched from this dataset.

Current blocker:

- SSH to the remote path became unstable during remote network checks (`Connection timed out during banner exchange`).

## Recommended Next Step

Once remote SSH is stable:

1. Sync the new script and dependency changes to `/scratch/xzh/code/structures25`.
2. Install/update the remote environment so `datasets` is available.
3. Download `hessian_qm9_DatasetDict.zip` into `/scratch/xzh/data/HessianQM9Figshare`.
4. Extract and run the audit.
5. Decide the use case:
   - external Hessian benchmark only; or
   - a separate force/Hessian model; or
   - generate missing PBE/OFDFT density labels for overlapping QM9 ids before using OFDFT EG/EGF.
