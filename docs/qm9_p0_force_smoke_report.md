# QM9 P0 Force Smoke Report

Date: 2026-07-01

## Scope

- Dataset: QM9 first 100 molecule ids.
- Geometry samples per molecule: 2 total, sample `0` reference geometry and sample `1` deterministic Gaussian coordinate perturbation.
- KS level: PBE/6-31G(2df,p).
- Force labels: enabled via same-level PySCF nuclear gradients in `metadata/pbe_derivatives/forces`.
- Hessian labels: disabled for P0.

## Runtime Paths

- Runtime root: `/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p0`
- QM9 raw xyz: `/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p0/QM9/raw`
- KS chk files: `/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p0/QM9PBEForceSmoke/kohn_sham`
- Label zarr files: `/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p0/QM9PBEForceSmoke/labels`
- Logs and summaries: `/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p0_logs`

## Results

- QM9 raw xyz files extracted: 133885.
- KS chk files generated and verified: 200.
- Label zarr files generated: 200.
- Force label validation: 200/200 labels contain finite `metadata/pbe_derivatives/forces`.
- Label numeric validation: no NaN/Inf found in numeric zarr arrays.
- Maximum per-atom force norm observed by the smoke checker: `0.09798267278000408`.

`check_qm9_force_smoke.py` summary:

```json
{
  "checked_files": 200,
  "expected_molecules": 100,
  "expected_samples": 2,
  "failures": [],
  "force_labels": 200,
  "force_norm_max": 0.09798267278000408,
  "molecules": 100
}
```

Tiny force-supervision forward/backward smoke:

```json
{
  "bias": 0.0,
  "final_loss": 0.0006080805502358812,
  "initial_loss": 0.0008129049017408304,
  "samples": 8,
  "scale": 0.0070316326768313136,
  "steps": 3
}
```

## Notes

- `/tmp` remains full because the large removable files are owned by other users. P0 commands used `TMPDIR=/mnt/afs/home/xiazhenhao/dft/_tmp`.
- The QM9 reader now accepts `*^` scientific notation at read time, so the full extracted raw xyz directory does not need an additional in-place conversion pass before P0 use.
