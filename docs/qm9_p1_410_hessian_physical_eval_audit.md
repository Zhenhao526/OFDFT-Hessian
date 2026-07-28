# QM9 P1-410 Hessian Physical Evaluation Audit

Date: 2026-07-03

Scope: this note only covers the QM9PBEForcePilot P1-410 early pilot. It does not claim conclusions for the final P1/P2 data scale.

## Inputs

- Dataset snapshot: `_runtime/qm9_p1/QM9PBEForcePilot`
- PBE Hessian reference manifest: `_runtime/qm9_p1_models/eval/qm9_p1_410_lambda1_extended_hessian_validation/pbe_hessian_reference_manifest_20.json`
- Previous fixed-density proxy eval: `_runtime/qm9_p1_models/eval/qm9_p1_410_lambda1_extended_hessian_validation/full_hessian_s3000_20mol_disp1e-3.json`
- New physical audit output dir: `_runtime/qm9_p1_models/eval/qm9_p1_410_hessian_physical_eval_audit`

Model checkpoints:

| Run | Run dir | Checkpoint |
|---|---|---|
| EG_s3000 | `_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200` | `checkpoints/epoch_006.ckpt` |
| EGF_lam1_s3000 | `_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000` | `checkpoints/epoch_000.ckpt` |

## What The Existing Hessian Eval Really Computes

Reviewed script: `scripts/qm9_hessian_eval_reference_set.py`.

The current 20 molecule Hessian eval is not a final density-optimized OFDFT Hessian.

Relevant behavior:

- It uses a selected cached dataloader sample and SCF iteration. `--scf-iteration` defaults to `1`, and targets are keyed as `(molecule_id, sample_id, args.scf_iteration)`.
- It perturbs `batch.pos` directly, reuses the same cached batch density/coefficient state, calls `model.net(batch)`, and obtains forces as `-dE_model/dR`.
- It finite-differences those derived forces to form the Hessian.
- It does not rerun OFDFT density optimization after each displaced geometry.

Therefore the correct name is:

> single-SCF fixed-density derived-force finite-difference Hessian proxy

It should not be called a full OFDFT Hessian or a final density-optimized physical Hessian.

## Proxy Baseline Reminder

On the existing 20 molecule reference set, the fixed-density proxy still shows a large EGF advantage at s3000:

| Eval | Run | n | Hessian MAE | Hessian RMSE | Relative Fro | Symmetry max abs |
|---|---:|---:|---:|---:|---:|---:|
| fixed-density proxy, 20 mol | EG_s3000 | 20 | 0.354476 | 1.725975 | 14.042393 | 0.004433 |
| fixed-density proxy, 20 mol | EGF_lam1_s3000 | 20 | 0.016708 | 0.048423 | 0.425208 | 0.000098 |

For the first 5 molecules used below, the same proxy gives:

| Eval | Run | n | Hessian MAE | Hessian RMSE | Relative Fro | Symmetry max abs |
|---|---:|---:|---:|---:|---:|---:|
| fixed-density proxy, first 5 | EG_s3000 | 5 | 0.435238 | 1.856111 | 10.087106 | 0.003304 |
| fixed-density proxy, first 5 | EGF_lam1_s3000 | 5 | 0.021418 | 0.058155 | 0.387190 | 0.000098 |

This is strong evidence for the proxy metric, but it is not by itself the final physical Hessian claim.

## Density-Relaxed Derived-Force Audit

Added script: `scripts/qm9_hessian_density_relaxed_eval.py`.

Definition:

> density-relaxed derived-force proxy: for each displaced geometry, rebuild the PySCF molecule, regenerate an OFDFT sample, run the normal density optimization machinery, then compute forces from the model scalar energy using `F_pred = -dE_model/dR` at the optimized density.

This keeps the force constraint: no independent force head is used.

Important limitation:

- Classical OFDFT terms are used during density optimization.
- The reported force still differentiates the model scalar energy path only. The current PySCF/integral path is not differentiable with respect to nuclear coordinates, so nuclear derivatives of the classical integral terms are not included.
- The result is therefore still a proxy, not a full total OFDFT Hessian.

Command run:

```bash
DFT_DATA=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1 \
DFT_MODELS=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models \
CUDA_VISIBLE_DEVICES=1 \
/usr/bin/time -v -o _runtime/qm9_p1_models/eval/qm9_p1_410_hessian_physical_eval_audit/density_relaxed_5mol_sad50_time.txt \
.venv/bin/python scripts/qm9_hessian_density_relaxed_eval.py \
  --manifest-json _runtime/qm9_p1_models/eval/qm9_p1_410_lambda1_extended_hessian_validation/pbe_hessian_reference_manifest_20.json \
  --dataset-dir _runtime/qm9_p1/QM9PBEForcePilot \
  --run EG_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200=_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200/checkpoints/epoch_006.ckpt \
  --run EGF_lam1_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000/checkpoints/epoch_000.ckpt \
  --output-json _runtime/qm9_p1_models/eval/qm9_p1_410_hessian_physical_eval_audit/density_relaxed_5mol_sad50.json \
  --output-csv _runtime/qm9_p1_models/eval/qm9_p1_410_hessian_physical_eval_audit/density_relaxed_5mol_sad50.csv \
  --optimization-csv _runtime/qm9_p1_models/eval/qm9_p1_410_hessian_physical_eval_audit/density_relaxed_5mol_sad50_optimizations.csv \
  --max-molecules 5 \
  --max-cycle 50 \
  --displacement 1e-3 \
  --device cuda:0 \
  --initialization sad_default
```

Runtime:

- Exit status: 0
- Wall time: 4:39.85
- Max RSS: 1,533,160 KB
- Device argument: `cuda:0` with `CUDA_VISIBLE_DEVICES=1`

The JSON from this first run has a legacy naming issue: its per-run `total_elapsed_s` field contains the mean per-molecule elapsed time. The values below use the per-row `elapsed_s` fields and `/usr/bin/time`.

## Density-Relaxed 5 Molecule Results

Molecules: `0000010`, `0000323`, `0000109`, `0000171`, `0000062`.

| Run | n | Hessian MAE | Hessian RMSE | Relative Fro | Symmetry max abs | Opt converged | Mean final density grad | Mean molecule seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EG_s3000 | 5 | 0.727102 | 2.277429 | 12.198326 | 8.620611 | 0 / 204 | 0.613275 | 22.73 |
| EGF_lam1_s3000 | 5 | 0.034219 | 0.081952 | 0.524189 | 0.165261 | 0 / 204 | 0.811007 | 25.89 |

Per-molecule details:

| Molecule | Natoms | Run | Hessian MAE | Hessian RMSE | Relative Fro | Symmetry max abs | Opt converged |
|---|---:|---|---:|---:|---:|---:|---:|
| 0000010 | 6 | EG_s3000 | 0.936660 | 4.498509 | 26.436113 | 34.728767 | 0 / 36 |
| 0000010 | 6 | EGF_lam1_s3000 | 0.045670 | 0.095970 | 0.563984 | 0.536794 | 0 / 36 |
| 0000323 | 6 | EG_s3000 | 1.734870 | 3.631442 | 15.667112 | 4.542910 | 0 / 36 |
| 0000323 | 6 | EGF_lam1_s3000 | 0.041973 | 0.093000 | 0.401230 | 0.109335 | 0 / 36 |
| 0000109 | 7 | EG_s3000 | 0.720543 | 2.500017 | 12.860298 | 3.264167 | 0 / 42 |
| 0000109 | 7 | EGF_lam1_s3000 | 0.023791 | 0.050239 | 0.258432 | 0.128050 | 0 / 42 |
| 0000171 | 7 | EG_s3000 | 0.104508 | 0.344861 | 2.913348 | 0.115135 | 0 / 42 |
| 0000171 | 7 | EGF_lam1_s3000 | 0.042482 | 0.121875 | 1.029587 | 0.024655 | 0 / 42 |
| 0000062 | 8 | EG_s3000 | 0.138927 | 0.412315 | 3.114760 | 0.452077 | 0 / 48 |
| 0000062 | 8 | EGF_lam1_s3000 | 0.017180 | 0.048676 | 0.367712 | 0.027473 | 0 / 48 |

Failure list:

- No molecule-level failures.
- All 408 individual displaced density optimizations produced finite forces.
- None of the displaced density optimizations met the current convergence criterion within `max_cycle=50`.

Interpretation:

- EGF lambda=1.0 remains much better than EG on the 5 molecule density-relaxed derived-force proxy.
- The result is not yet a converged density-optimized physical Hessian because every density optimization is truncated.
- The larger symmetry errors in this density-relaxed audit, especially for EG, are a useful warning that convergence and/or force-path completeness still matter.

## Original-Structure Density Optimization Check

To separate finite-difference displacement effects from the density optimizer itself, the same 5 molecules were also tested at the original `sample_id=0` geometry with no displacement.

Added script: `scripts/qm9_density_optimization_original_structures.py`.

Command run:

```bash
DFT_DATA=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1 \
DFT_MODELS=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models \
CUDA_VISIBLE_DEVICES=1 \
/usr/bin/time -v -o _runtime/qm9_p1_models/eval/qm9_p1_410_hessian_physical_eval_audit/original_density_optimization_5mol_sad50_time.txt \
.venv/bin/python scripts/qm9_density_optimization_original_structures.py \
  --dataset-dir _runtime/qm9_p1/QM9PBEForcePilot \
  --run EG_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200=_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200/checkpoints/epoch_006.ckpt \
  --run EGF_lam1_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000/checkpoints/epoch_000.ckpt \
  --output-json _runtime/qm9_p1_models/eval/qm9_p1_410_hessian_physical_eval_audit/original_density_optimization_5mol_sad50.json \
  --output-csv _runtime/qm9_p1_models/eval/qm9_p1_410_hessian_physical_eval_audit/original_density_optimization_5mol_sad50.csv \
  --sample-id 0 \
  --max-cycle 50 \
  --device cuda:0 \
  --initialization sad_default
```

Runtime:

- Exit status: 0
- Wall time: 0:43.93
- Max RSS: 1,525,032 KB

Here `final density loss` means the projected density-gradient norm used by the density optimizer's convergence check.

| Molecule | Natoms | Run | Converged | Cycles | Final density loss | Final total energy |
|---|---:|---|---|---:|---:|---:|
| 0000010 | 6 | EG_s3000 | false | 50 | 0.535450 | -132.763390 |
| 0000323 | 6 | EG_s3000 | false | 50 | 0.736105 | -299.041550 |
| 0000109 | 7 | EG_s3000 | false | 50 | 0.714162 | -225.052573 |
| 0000171 | 7 | EG_s3000 | false | 50 | 0.601652 | -262.195542 |
| 0000062 | 8 | EG_s3000 | false | 50 | 0.475655 | -283.091462 |
| 0000010 | 6 | EGF_lam1_s3000 | false | 50 | 0.694966 | -132.792397 |
| 0000323 | 6 | EGF_lam1_s3000 | false | 50 | 0.884037 | -299.614553 |
| 0000109 | 7 | EGF_lam1_s3000 | false | 50 | 0.915541 | -225.368877 |
| 0000171 | 7 | EGF_lam1_s3000 | false | 50 | 0.873479 | -262.328402 |
| 0000062 | 8 | EGF_lam1_s3000 | false | 50 | 0.682170 | -283.497704 |

Summary:

| Run | n success | n converged | Mean cycles | Mean final density loss | Mean final total energy |
|---|---:|---:|---:|---:|---:|
| EG_s3000 | 5 | 0 | 50.0 | 0.612605 | -240.428903 |
| EGF_lam1_s3000 | 5 | 0 | 50.0 | 0.810039 | -240.720387 |

This confirms that the non-convergence observed in the density-relaxed Hessian audit is not caused only by finite-difference displacements. Under the current `sad_default` + SGD settings, the original sample geometries also fail the `1e-4` projected-gradient convergence threshold within 50 cycles.

## Mass-Weighted Hessian / Frequency Pipeline

Added script: `scripts/qm9_mass_weighted_hessian_check.py`.

Definition:

- Load cached PBE analytic Hessian.
- Symmetrize Hessian.
- Mass-weight using PySCF atomic masses in atomic mass units.
- Build translational/rotational basis in mass-weighted coordinates.
- Project out external modes.
- Convert eigenvalues to signed frequencies in cm^-1.

Assumed units:

- Hessian: Hartree / Bohr^2
- Mass: atomic mass unit
- Conversion: `5140.4871436115645 cm^-1 per sqrt(au/amu)`

Command run:

```bash
.venv/bin/python scripts/qm9_mass_weighted_hessian_check.py \
  --manifest-json _runtime/qm9_p1_models/eval/qm9_p1_410_lambda1_extended_hessian_validation/pbe_hessian_reference_manifest_20.json \
  --dataset-dir _runtime/qm9_p1/QM9PBEForcePilot \
  --output-json _runtime/qm9_p1_models/eval/qm9_p1_410_hessian_physical_eval_audit/pbe_mass_weighted_frequency_self_check_20.json \
  --output-csv _runtime/qm9_p1_models/eval/qm9_p1_410_hessian_physical_eval_audit/pbe_mass_weighted_frequency_self_check_20.csv \
  --max-molecules 20
```

Self-check result:

| Metric | Value |
|---|---:|
| PBE references checked | 20 |
| Finite frequency rows | 20 |
| Max absolute external mode frequency after projection | 6.688842e-05 cm^-1 |
| Rows with negative vibrational modes below -1 cm^-1 | 16 |

The translation/rotation projection and unit conversion are internally consistent on the PBE reference Hessians. However, frequency MAE is not reported yet because:

- The model Hessian path is still a proxy and not a full total OFDFT Hessian.
- Many reference geometries are perturbed QM9 samples rather than confirmed PBE minima, so imaginary modes are expected and frequency MAE would be hard to interpret as a vibrational-quality metric.

## Can We Claim Hessian Improvement?

Current defensible statement:

> Within the P1-410 early pilot, EGF lambda=1.0 shows a stable Hessian-proxy improvement over EG. This is true for the 20 molecule single-SCF fixed-density proxy and also persists in a 5 molecule density-relaxed derived-force audit.

Current non-defensible statement:

> EGF lambda=1.0 has been proven to improve the final physical density-optimized total OFDFT Hessian.

That stronger claim is not yet supported because:

- The original 20 molecule eval is fixed-density at `scf_iteration=1`.
- The new density-relaxed audit uses truncated density optimization; 0 / 408 displacement optimizations converged under `max_cycle=50`.
- The force used for the density-relaxed audit remains the derivative of model scalar energy only and excludes non-differentiable classical integral/nuclear force terms.

## Recommendation

Do not expand this density-relaxed Hessian audit to 20 or 50 molecules yet. The current 5 molecule result is promising, but expanding before fixing the physical definition would mostly multiply a proxy.

Recommended next step:

1. Resolve the physical force path for total OFDFT energy, or explicitly switch to a total-energy finite-difference Hessian on 1-2 molecules where each geometry is density-optimized and the scalar total energy is compared.
2. Improve density optimization convergence before expansion. Try better initialization or optimizer settings first, then rerun the 5 molecule audit until most displaced optimizations converge.
3. Keep the current 20 molecule fixed-density proxy benchmark as a useful regression baseline.
4. Defer 50 molecule PBE Hessian reference expansion until the density-optimized evaluation definition is settled.
5. Do not return to 1000 molecule label generation yet; the immediate bottleneck is physical Hessian validation, not more labels.

## Artifacts Produced

- `scripts/qm9_hessian_density_relaxed_eval.py`
- `scripts/qm9_mass_weighted_hessian_check.py`
- `scripts/qm9_density_optimization_original_structures.py`
- `_runtime/qm9_p1_models/eval/qm9_p1_410_hessian_physical_eval_audit/density_relaxed_smoke_1mol.*`
- `_runtime/qm9_p1_models/eval/qm9_p1_410_hessian_physical_eval_audit/density_relaxed_5mol_sad50.*`
- `_runtime/qm9_p1_models/eval/qm9_p1_410_hessian_physical_eval_audit/original_density_optimization_5mol_sad50.*`
- `_runtime/qm9_p1_models/eval/qm9_p1_410_hessian_physical_eval_audit/pbe_mass_weighted_frequency_self_check_20.*`
