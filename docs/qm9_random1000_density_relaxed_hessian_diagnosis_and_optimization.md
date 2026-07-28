# QM9 random1000 Density-Relaxed Hessian Diagnosis and Model Optimization

Last updated: 2026-07-15

## Scope

This report tracks the systematic diagnosis and optimization of the random1000 Test100
density-relaxed derived-force Hessian workflow. It does not change the scalar-energy force
definition, add a force head, or claim final P1/P2 performance.

Primary baseline artifacts:

- dataset: `/scratch/xzh/data/QM9PBEForceRandom1000`;
- PBE analytic Hessian manifest:
  `/scratch/xzh/models/eval/qm9_random1000_test100_hessian_node04_gpu4pyscf/20260714_135600/pbe_hessian_manifest_test100_gpu4pyscf.json`;
- strict Test100 density-relaxed result:
  `/scratch/xzh/models/eval/qm9_random1000_test100_density_relaxed_hessian_node04/20260714_150546`;
- EG checkpoint: `qm9_random1000_eg_e10_20260713_162829/checkpoints/epoch_009.ckpt`;
- historical EGF checkpoint:
  `qm9_random1000_egf_lam1_e10_20260713_162829/checkpoints/epoch_009.ckpt`.

## Reproduction

Remote environment:

```bash
ssh -J root@101.200.216.125 -p 2200 shenwei01@localhost
source /scratch/xzh/env.sh
cd /scratch/xzh/code/structures25
```

The executed and dependency-chained entry points are:

```bash
sbatch scripts/slurm_qm9_random1000_hessian_protocol_scan_node04.sbatch
sbatch scripts/slurm_qm9_random1000_hessian_physics_audits_node04.sbatch
sbatch scripts/slurm_qm9_random1000_force_lambda_sweep_8xa100.sbatch
sbatch scripts/slurm_qm9_random1000_paired_train_labels_node02.sbatch
sbatch scripts/slurm_qm9_random1000_validation8_pbe_hessian_node02.sbatch
sbatch scripts/slurm_qm9_random1000_anomaly_tol1e6_full_hessian_node06.sbatch

sbatch --dependency=afterok:487 \
  scripts/slurm_qm9_random1000_protocol_bad_point_rescue.sbatch
sbatch --dependency=afterok:492 \
  scripts/slurm_qm9_random1000_lambda_tier1_node05.sbatch
sbatch --dependency=afterok:493 \
  scripts/slurm_qm9_random1000_paired_augmentation_prepare_node02.sbatch
sbatch --dependency=afterok:494:495 \
  scripts/slurm_qm9_random1000_paired_candidate_8xa100.sbatch
sbatch --dependency=afterok:487:509:510 \
  scripts/slurm_qm9_random1000_validation_tier2_strict_hessian_node04.sbatch
sbatch --dependency=afterok:511 \
  scripts/slurm_qm9_random1000_frozen_test100_node05.sbatch
```

Each launcher writes resolved input paths, checkpoints, raw per-point/per-molecule CSV or JSON,
optimizer traces where requested, `/usr/bin/time -v` output, and aggregate plots under its stamped
`/scratch/xzh/models/{train,eval,labelgen}` directory. Job numbers above identify this run; the
launchers themselves do not require those fixed numbers when started as a fresh chain.

## Force Definition Audit

The trained scalar target is `kin_plus_xc`. Density optimization evaluates

```text
E_total = E_model(kin_plus_xc) + E_Hartree + E_electron-nuclear + E_nuclear-repulsion.
```

The current reported force is

```text
F_current = - partial E_model / partial R |rho
```

after density optimization. It omits classical nuclear derivatives and optimized-density response.
Consequently it is not guaranteed to equal `-d E_total(rho*(R), R) / dR`, and its finite-difference
Hessian need not be symmetric even when every density point meets its convergence threshold.

## Total-OFDFT Derivative Implementation Audit

The current code cannot obtain a conservative total force by merely changing the final
`autograd.grad` call:

- `FunctionalFactory.evaluate_functional()` converts Hartree, nuclear-attraction, and model
  energies through `.item()` and asserts that the returned coefficient gradient has no graph;
- `Energies` stores Python/NumPy scalars and obtains nuclear repulsion from `mol.energy_nuc()`;
- PySCF integral tensors are rebuilt from a `gto.Mole` and are not functions of `sample.pos` in
  Torch;
- `basis_integrals.py` contains useful nuclear-attraction first-derivative helpers, but there is no
  unified derivative provider for Coulomb, overlap/normalization, moving-basis Pulay terms, and
  nuclear repulsion;
- local-frame and natural-representation transforms are rebuilt outside the nuclear autograd graph.

For coefficients `c`, nuclear coordinates `R`, electron-number constraint `q(R)^T c=N`, and
Lagrange multiplier `mu`, define

```text
L(c,R,mu) = E_total(c,R) + mu * (q(R)^T c - N).
```

At a converged constrained minimum, the conservative force is the explicit constrained total
derivative

```text
F = -L_R(c*(R), R, mu*(R)).
```

Density response cancels from the first derivative only after all total-energy, constraint, and
moving-basis derivatives are included. For the Hessian, solve the KKT response system

```text
[ L_cc   q ] [ dc/dR  ] = -[ L_cR ]
[ q^T    0 ] [ dmu/dR ]    [ g_R  ]
```

and contract the response into `dF/dR`. In an electron-number tangent basis this reduces to the
familiar Schur complement

```text
H_relaxed = L_RR - L_Rc * (L_cc)^(-1) * L_cR,
```

with the constraint and Pulay terms understood. The recommended implementation order is:

1. Add a tensor-valued total-energy API that never calls `.item()` on the differentiable path.
2. Add a `GeometryIntegralProvider` returning values and coordinate JVP/VJP for Coulomb,
   nuclear attraction, overlap/normalization, nuclear repulsion, and basis transforms.
3. Validate total force against central differences of fully relaxed scalar total energy and
   require closed-loop work to approach zero with tighter numerical tolerance.
4. Expose projected coefficient-Hessian HVP and mixed `R-c` VJP interfaces.
5. Solve tangent-space response with CG when positive definite; use MINRES for the full indefinite
   KKT system, with residual and conditioning logs.
6. Validate implicit HVP against finite differences before assembling a full Hessian by basis-vector
   HVPs.

Acceptance gates for that implementation should be numerical rather than visual:

- relaxed total force vs central difference of `E_total`: relative error below `1e-4` on the
  small audit set;
- square-loop work divided by loop area converges to zero as density and response residuals are
  tightened;
- float64 total-Hessian `||H_asym||F/||H_sym||F < 1e-5` away from singular electronic response;
- implicit total-energy HVP vs strict relaxed-force finite difference: relative Frobenius below
  `1e-3` with logged linear-solver residual below `1e-8`;
- translation/rotation projection, frequencies, and modes are evaluated only after those gates.

Post-hoc Hessian symmetrization is not an acceptable substitute for these interfaces. Until they
exist, the official random1000 result must be named a density-relaxed incomplete-derived-force
Hessian proxy.

## Representative Molecules

| molecule | natoms | reason |
|---|---:|---|
| 0000777 | 7 | smallest representative and EGF MAE outlier |
| 0043905 | 14 | central-error sample |
| 0056566 | 16 | EG median-error sample |
| 0072895 | 17 | EGF median-error sample |
| 0054659 | 18 | central-error sample |
| 0093887 | 19 | elevated relaxed-force asymmetry |
| 0040728 | 21 | dominant EG error/asymmetry outlier |
| 0060531 | 27 | largest-size representative |

The selection manifest is
`/scratch/xzh/models/eval/qm9_random1000_hessian_protocol_scan/20260714_194632/representative_manifest_8mol.json`.

## Baseline Decomposition

At `h=1e-3 Bohr` and final projected density-gradient threshold `1e-4`:
Hessian MAE/RMSE and symmetry elements below are in `Hartree/Bohr^2`; relative Frobenius and
asymmetric/symmetric ratios are dimensionless.

| model | mean raw MAE | mean symmetrized MAE | mean raw RMSE | mean asym/sym Fro ratio | mean symmetry max abs |
|---|---:|---:|---:|---:|---:|
| EG | 0.092028 | 0.088822 | 0.271440 | 0.199930 | 2.124786 |
| historical EGF | 0.015199 | 0.014648 | 0.043705 | 0.185151 | 0.299432 |

All selected baseline displacement points meet `1e-4`. The large antisymmetric fraction therefore
cannot be dismissed as simply failed convergence, although the threshold scan is still required to
test whether residual magnitude controls it.

The completed `h=1e-3`, threshold `3e-5` condition has 834/834 strict points per model. Tightening
from `1e-4` changes mean raw MAE by `-7.2e-7` (EG) and `+4.1e-7` (EGF), and changes mean
asymmetric/symmetric Frobenius ratio by only `-2.0e-6` and `-3.9e-6`, respectively. This is below
the meaningful model-comparison scale and supports `1e-4` as the routine cost/accuracy setting.

### Full Protocol Scan

Slurm 487 completed all eight new conditions with zero worker failures in `7:09:21`; together with
the reused baseline this is the requested 3x3 grid. At threshold `1e-4`, all 834 displacement
points per model and step are strict:

| model | h (Bohr) | raw MAE | sym MAE | asym/sym Fro | symmetry max abs |
|---|---:|---:|---:|---:|---:|
| EG | `3e-3` | 0.078255 | 0.076053 | 0.154997 | 0.776135 |
| EG | `1e-3` | 0.092028 | 0.088822 | 0.199930 | 2.124786 |
| EG | `3e-4` | 0.094950 | 0.093309 | 0.231136 | 3.435678 |
| historical EGF | `3e-3` | 0.013650 | 0.013165 | 0.131102 | 0.138692 |
| historical EGF | `1e-3` | 0.015199 | 0.014648 | 0.185151 | 0.299432 |
| historical EGF | `3e-4` | 0.015971 | 0.015425 | 0.208366 | 0.563327 |

The lower apparent PBE error at `3e-3` is not by itself evidence of a better local derivative.
Relative to `h=1e-3`, the five regular molecules change by about `5.6e-4`--`2.8e-3` relative
Frobenius, but the hard cases change substantially:

| model | molecule | relative change at `3e-3` | relative change at `3e-4` |
|---|---|---:|---:|
| EG | 0040728 | 0.697 | 1.384 |
| EG | 0060531 | 0.281 | 0.843 |
| EG | 0093887 | 0.235 | 0.265 |
| historical EGF | 0040728 | 0.621 | 1.335 |
| historical EGF | 0060531 | 0.141 | 0.416 |
| historical EGF | 0093887 | 0.345 | 0.391 |

The density-threshold scan also shows that asking for a smaller target does not guarantee a
stricter realized solution under a fixed 10000-cycle cap:

| h | model | strict points at `1e-4` | at `3e-5` | at `1e-5` |
|---:|---|---:|---:|---:|
| `3e-3` | EG | 834/834 | 833/834 | 806/834 |
| `3e-3` | historical EGF | 834/834 | 834/834 | 824/834 |
| `1e-3` | EG | 834/834 | 834/834 | 816/834 |
| `1e-3` | historical EGF | 834/834 | 834/834 | 820/834 |
| `3e-4` | EG | 834/834 | 832/834 | 807/834 |
| `3e-4` | historical EGF | 834/834 | 833/834 | 818/834 |

At the formal `h=1e-3, tol=1e-4` point, mean per-molecule absolute-error quantiles are:

| model | raw q50 | raw q90 | raw q95 | raw q99 | raw max | sym q50 | sym q90 | sym q99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EG | 0.00877 | 0.26611 | 0.51616 | 1.18883 | 3.07276 | 0.00950 | 0.25105 | 1.04773 |
| historical EGF | 0.00229 | 0.03813 | 0.06363 | 0.19965 | 0.62047 | 0.00225 | 0.03553 | 0.18217 |

Across all 72 molecule-condition rows per model, Pearson correlation between mean density residual
and asym/sym Frobenius is only `0.188` (EG) and `0.008` (EGF); using maximum residual gives
`0.290` and `0.144`. The raw tables retain q50/q90/q95/q99/q100 for raw error, symmetric error,
and absolute antisymmetry, plus every optimization trace and failure classification.

Artifact:

```text
/scratch/xzh/models/eval/qm9_random1000_hessian_protocol_scan/20260714_194632/analysis
```

### Full-Hessian Anomaly Supplement at `1e-6`

Slurm 515 completed all six `(model,h)` conditions for `0000777` and `0040728` in 7:25:04 with
zero process failures. Slurm 516 then merged the supplement with the main scan through the same
metric and trace-classification implementation. Requested tolerance `1e-6` was not realized at
many hard displacement points:

| model | h (Bohr) | strict displacement points | fully strict molecules | mean raw / sym MAE | mean asym/sym Fro | mean symmetry max abs |
|---|---:|---:|---:|---:|---:|---:|
| EG | `3e-3` | 112/168 | 0/2 | 0.1193 / 0.1120 | 0.4248 | 2.425 |
| EG | `1e-3` | 113/168 | 1/2 | 0.1728 / 0.1619 | 0.4911 | 7.159 |
| EG | `3e-4` | 145/168 | 0/2 | 0.1778 / 0.1732 | 0.5023 | 12.203 |
| historical EGF | `3e-3` | 109/168 | 0/2 | 0.02434 / 0.02347 | 0.2628 | 0.260 |
| historical EGF | `1e-3` | 112/168 | 0/2 | 0.03030 / 0.02910 | 0.4097 | 0.652 |
| historical EGF | `3e-4` | 103/168 | 0/2 | 0.03246 / 0.03125 | 0.4801 | 1.837 |

The aggregate means combine a smooth structural case with a solver/branch hard case and are not a
new accuracy benchmark. On `0000777`, EGF maximum antisymmetry is `0.05972`, `0.05899`, and
`0.05900` at the three steps, matching the independently measured `1e-6` curl. Even though only
33--37/42 full-stencil points meet `1e-6`, the relevant closed-loop corners are strict and the
curl is threshold and step stable. On `0040728`, only 67--76/126 EGF points are strict and maximum
antisymmetry changes from `0.461` to `1.244` to `3.616` as the step shrinks. It remains a
non-local/solver hard case and must not be used to infer a converged local tensor.

Artifacts:

```text
/scratch/xzh/models/eval/qm9_random1000_anomaly_tol1e6_full_hessian/20260714_230523
/scratch/xzh/models/eval/qm9_random1000_hessian_protocol_scan/20260714_194632/analysis_complete_with_anomaly1e6
```

## Numerical Precision Audit

Fixed-density full-edge autograd/FD/HVP scan:

```text
/scratch/xzh/models/eval/qm9_random1000_fixed_hessian_precision_scan/20260714_200631
```

All 96 full-Hessian model/dtype/displacement cases are finite. Direct fixed-density autograd
`float32-vs-float64` relative differences average `9.8e-6` for EG and `2.9e-6` for EGF. The best
autograd-vs-FD agreement is near `h=1e-3`; `h=3e-4` has more rounding noise. These errors are far
smaller than the density-relaxed antisymmetric components, so ordinary float precision is not the
leading explanation.

The reproducible aggregation is in
`/scratch/xzh/models/eval/qm9_random1000_fixed_hessian_precision_scan/20260714_200631/analysis`.
All 96 full-Hessian and 192 HVP cases are finite. At `h=1e-3`, float64 mean autograd-vs-FD
relative Frobenius errors are `1.40e-4` (EG) and `1.25e-4` (historical EGF).

## Model Smoothness Audit

Artifact:

```text
/scratch/xzh/models/eval/qm9_random1000_model_second_order_smoothness/20260714
```

- actual nonlinearities are GELU and SiLU, with finite float64 first and second derivatives;
- all model cutoffs are disabled (`null`);
- selected original geometries use fixed complete directed graphs with self loops;
- self-loop distance is a constant zero through `_safe_edge_lengths`;
- no neighbor-list or cutoff switching occurs within the audited Hessian path.

The remaining representation caveat is that local-frame/cached basis transforms are rebuilt
between independently displaced geometries but are not part of the current nuclear-coordinate
autograd graph. This is subordinate to, but consistent with, the incomplete total-force issue.

## Closed-Loop Conservativity Audit

For historical EGF on `0000777`, `h=1e-3`, target tolerance `1e-4`:

| force field | loop work (Ha) | curl estimate (Ha/Bohr2) |
|---|---:|---:|
| fixed density | `3.81e-12` | `9.53e-7` |
| relaxed density, current force | `-2.36e-7` | `-5.906e-2` |

All four relaxed corners converged below `1e-4`. The completed threshold and step audit is in
`/scratch/xzh/models/eval/qm9_random1000_hessian_physics_audits/20260714_213743`.
For `0000777`, the curl is invariant while the final density residual is reduced by two orders of
magnitude:

| model | threshold | max final density gradient | relaxed curl | curl / baseline max asymmetry |
|---|---:|---:|---:|---:|
| EG | `1e-4` | `9.72e-5` | -0.107629 | 0.99963 |
| EG | `1e-5` | `9.64e-6` | -0.107593 | 0.99930 |
| EG | `1e-6` | `9.93e-7` | -0.107581 | 0.99919 |
| EGF | `1e-4` | `9.41e-5` | -0.059058 | 1.00004 |
| EGF | `1e-5` | `9.16e-6` | -0.058963 | 0.99844 |
| EGF | `1e-6` | `9.62e-7` | -0.058985 | 0.99881 |

At strict `1e-5`, the same curl is recovered at `h=3e-3`, `1e-3`, and `3e-4` to about 0.2% for
EG and EGF. Loop work scales with loop area, as expected for a finite nonzero curl. Fixed-density
loop work remains near `1e-12`--`1e-8 Ha`; its small residual grows at the largest step and is
ordinary finite-difference truncation. Across all loop cases, Pearson correlation between final
density residual and absolute curl is only 0.109 (EG) and 0.148 (EGF).

This is direct quantitative evidence that the dominant smooth-case asymmetry is structural: it is
the differential signature of the current non-conservative relaxed derived-force field. It is not
caused by matrix formatting, float precision, or the requested density tolerance. `0040728`
remains a separate hard case: its curl changes strongly with step and some `1e-6` corners do not
converge, so no single local derivative is trusted there.

## Relaxed Scalar-Energy Hessian Audit

A naturally symmetric 2x2 block was built from nine fully relaxed scalar-energy stencil points at
`h=3e-3` and `1e-3`. `total_energy` includes model, Hartree, electron-nuclear, and nuclear
repulsion terms. Only 9/9 strict rows are used.

- Historical EGF reached 9/9 strict convergence for all three molecules at both steps. Its total
  energy blocks changed by only 0.22%--0.86% relative Frobenius between the two steps, yet current
  force-difference blocks differ from them by mean MAE 0.568--0.569.
- For the common strict EG molecule `0021889`, total-energy step stability is 0.88% relative Fro;
  current force-vs-total-energy block MAE is 1.5845 at `h=1e-3`.
- On that same molecule, current EG derived-force vs PBE MAE is 0.1258 while relaxed total-energy
  vs PBE MAE is 1.4587; for EGF the corresponding values are 0.0493 and 0.4315. A numerically
  smaller PBE error from the incomplete force therefore does not make it a derivative of the
  implemented total OFDFT energy.

The energy audit establishes a stable conservative reference for the implemented scalar total
energy and shows that it is materially different from the current force-difference Hessian.
Failed EG stencil rows (`0000777` at both steps and `0000242` at `h=3e-3`) remain in raw tables but
are excluded from strict aggregates.

## Vibrational Baseline

The postprocessor symmetrizes each Hessian, mass weights it, projects out translation/rotation, and
matches modes to PBE by maximum absolute overlap. Symmetrization here is diagnostic only.

| model | molecules | mean frequency MAE (cm-1) | mean frequency RMSE (cm-1) | mean mode overlap | model imaginary modes | PBE imaginary modes |
|---|---:|---:|---:|---:|---:|---:|
| EG | 100 | 1779.9 | 3198.5 | 0.636 | 1265 | 229 |
| historical EGF | 100 | 575.2 | 719.7 | 0.618 | 1260 | 229 |

The frequency MAE improvement does not translate into improved mode overlap or acceptable imaginary
mode counts. Model selection must report all of these metrics.

## Active Experiments

- Slurm 487: completed 8-molecule `h={3e-3,1e-3,3e-4}` and
  `threshold={1e-4,3e-5,1e-5}` protocol scan in 7:09:21 with zero condition-process failures.
- Slurm 505: completed 2-D force-loop and relaxed scalar-energy block audit with zero worker
  failures. Raw points, traces, strict-only summaries, stability tables, and plots are preserved.
- Full per-cycle optimization curves are saved as compressed NPZ files for new conditions.
- Slurm 492: actual force-weight `0.3,1,3,10` controlled training on node01.
- Slurm 494: first Tier-1 attempt failed before metrics because six concurrent force evaluators
  each spawned eight DataLoader workers, exhausting multiprocessing descriptor/shared-memory
  transfers. Replacement Slurm 517 used `num_workers=0` and completed the six-model validation
  force plus fixed-density Hessian/HVP screen in 11:36.
- Slurm 493: completed CPU-only generation of 1600 exact paired labels for 800 train parents in
  9:11:22. It verified 1600/1600 chk and labels, 800 complete pairs, finite force labels, and no
  failures. SCF and label transform wall times were 5:07:27 and 4:02:00.
- Slurm 495 failed before transform because a Hydra field needed `+` insertion; no data was
  removed. Replacement 525 completed transform/split generation. Job 529 then exposed an invalid
  launcher precheck for a synthetic unified label directory. The builder was strengthened to
  resolve-check all 5200 split entries from their actual source datasets. Replacement 532 passed:
  train/val/test entries are 4800/400/400, sources are 4000 base plus 1600 paired entries, and all
  parent overlaps are empty. The frozen split SHA256 is
  `919c5f6c250b0d2e9f14f43522894fcac8b179d6d4fd5b774c2cec1ea249034b`. Job 533 failed before
  training because `limit_train_batches` also
  required Hydra `+` insertion. A full single-process config compose then passed. Job 536 completed
  the eight-A100 compute-matched training run on node01 in 49:18: 12330 optimizer steps, mean
  162.9 samples/s, callback peak 816 MB/GPU, and final `epoch_009.ckpt` plus `last.ckpt` saved.
- Slurm 510: completed eight independent validation-set PBE analytic Hessians on GPU4PySCF in
  3:27 wall time (8/8 success). These references make strict Tier 2 validation independent of
  Test100.
- Slurm 508 was cancelled before execution once the required `h=1e-3`, `1e-5` condition finished,
  so rescue could run without waiting for the other steps. First replacement 522 exited in one
  second because it still referenced the not-yet-created final analysis CSV. A separate immutable
  `analysis_pre_rescue_20260715` was then generated, and Slurm 523 started the intended
  bad-point-only comparison from that explicit CSV. Serial timing showed that 32 points x 3
  variants would exceed its wall limit, so 523 was cancelled with its log retained. Slurm 524,
  the eight-GPU sharded replacement on node05, completed in 1:42:40. The merger verified all 96
  unique variant results for the 32 bad points and all 96 compressed per-cycle curves. No good
  point was rerun.
- Slurm 515 completed the six-worker full-Hessian anomaly supplement for `0000777` and `0040728`
  at all three steps and density target `1e-6`; every displacement trace and non-converged point is
  retained. Slurm 516 completed the unified baseline, 3x3, and anomaly analysis in 2:38.
- Slurm 536 completed compute-matched paired training. Slurm 537 then completed the strict Val8
  density-relaxed comparison. Actual-force-weight 3.0 and 1.0 passed the validation freeze; the
  paired candidate did not pass the pre-registered energy gate. Slurm 538 completed the one-shot
  frozen Test100 confirmation in 2:45:38 with exit code zero. Actual weight 1.0 is the sole stable
  candidate under all frozen gates. Invalidated dependency jobs and preflight failures produced no
  scientific model output.
- Slurm 503: completed one-molecule two-model correctness/timing smoke for the optional shared
  prepared-geometry cache added to `qm9_hessian_density_relaxed_eval.py`.

## Cost Reduction Audit

### Strict Bad-Point Solver Rescue

The completed `h=1e-3, tolerance=1e-5` scan had 32 non-strict displacement points: 18 for EG and
14 for historical EGF. Slurm 524 tested three rescue variants on exactly those points, with no
process errors:

| model | rescue variant | converged | mean / median / max cycles | total serial point time (s) |
|---|---|---:|---:|---:|
| EG | label warm-start + Adam `3e-4` | 7/18 | 13010 / 20000 / 20000 | 5901 |
| EG | label warm-start + Adam then SLSQP | 15/18 | 9668 / 6878 / 31673 | 4701 |
| EG | SAD + Adam then SLSQP | **18/18** | 8409 / 10057 / 10108 | 3858 |
| EGF | label warm-start + Adam `3e-4` | 8/14 | 9862 / 3067 / 20000 | 5525 |
| EGF | label warm-start + Adam then SLSQP | 10/14 | 11374 / 3200 / 31636 | 6629 |
| EGF | SAD + Adam then SLSQP | **14/14** | 2418 / 1911 / 10079 | 1361 |

The evidence rejects label/reference density as a universal hard-point initializer: it helps some
points but places others in a persistent bad basin. The reliable strict rescue is `sad_default`,
Adam `lr=3e-4`, followed by SLSQP. For routine evaluation, retain the inexpensive
base-density-warm-start two-stage Adam path first; only points that miss the target should be
rerun with the SAD + Adam-to-SLSQP rescue. A label-warm-start attempt may precede it as an
opportunistic optimization, but cannot replace the SAD branch.

Artifact:

```text
/scratch/xzh/models/eval/qm9_random1000_hessian_protocol_bad_point_rescue_parallel/20260715_031203
```

The completed Test100 artifact attributes about 72.1% of EG model time and 80.4% of historical
EGF model time to density optimization. The remaining 20%--28% includes repeated PySCF integral
and transformed-sample construction plus force autograd. New evaluator records split each point
into `sample_build_elapsed_s`, `density_optimization_elapsed_s`, `force_autograd_elapsed_s`, and
`total_point_elapsed_s`.

Pair-response extrapolation initializes each minus displacement with
`2*rho(base)-rho(plus)`. It is numerically neutral for the resulting Hessian but not uniformly
faster:

| molecule | effect on minus cycles | Hessian change vs baseline | timing interpretation |
|---|---:|---:|---|
| 0000777 | `119.3 -> 96.1` mean, -19.5% | relative Fro `7.0e-4` | useful small-case speedup |
| 0040728 | `125.7 -> 125.4` mean, -0.3% | relative Fro `2.64e-4` | no cycle benefit; wall time noisy/worse |

All audited points remained strict at `1e-4`, and PBE metrics were unchanged at practical
precision. Response extrapolation therefore remains opt-in rather than the formal default.

The new `--share-prepared-geometry-across-runs` option caches model-independent transformed
geometry/integral samples on CPU and clones them for EG/EGF. It validates identical basis metadata,
transforms, and target key before sharing. Slurm 503 produced 43 builds and 43 cross-model hits on
`0000777`. Cached-vs-uncached Hessian relative Frobenius differences were `7.4e-9` (EG) and
`8.74e-5` (historical EGF), with unchanged PBE MAE. Combined model time changed
`368.5 -> 360.6 s`, only a 1.022x speedup; maximum RSS was 2.34 GB. The cache is numerically
acceptable; it was enabled with per-process lifetime in the representative and Test100 funnels.
Density optimization still dominates, and Test100 showed that host memory, not CUDA memory, is the
cache-lifetime constraint.

Artifact:

```text
/scratch/xzh/models/eval/qm9_random1000_prepared_geometry_cache_smoke/20260714_job503
```

Slurm 504 completed the cache audit on hard case `0040728`: 127 builds/127 hits, cached-vs-uncached
relative Frobenius differences `1.16e-7` (EG) and `4.21e-7` (EGF), combined model time
`1384.9 -> 1149.0 s` (1.205x), and peak RSS 7.70 GB. The cache is enabled for the small
representative-model funnel. Test100 requires per-shard or per-molecule cache lifetime limits
because CPU memory grows with the number and size of retained displaced geometries.

Hard-case artifact:

```text
/scratch/xzh/models/eval/qm9_random1000_prepared_geometry_cache_smoke/20260714_job504_0040728
```

Two independent evaluator workers sharing one A100 were calibrated on the same four-molecule
historical-EGF workload. One worker required `2640 s`; two balanced workers required `1395 s`, a
`1.892x` wall-time speedup. Both paths completed 4/4 molecules. Their maximum Hessian relative
Frobenius difference was `2.89e-5`, and the maximum absolute change in reported molecule MAE was
`3.26e-7`. A separate `0040728` thread sweep kept all 126/126 points strict and gave:

| CPU threads per worker | wall time (s) | speedup vs 1 thread | Hessian relative difference |
|---:|---:|---:|---:|
| 1 | 878 | 1.000 | 0 |
| 4 | 761 | 1.154 | `1.04e-6` |
| 8 | 730 | 1.203 | `7.29e-7` |

The frozen Test100 launcher therefore uses 16 dynamically load-balanced shards, two workers per
A100, and eight BLAS/OpenMP threads per worker. It records both settings in the final summary.
These controls alter scheduling only; displacement, convergence threshold, scalar-energy force,
and strict-point accounting are unchanged.

Artifacts:

```text
/scratch/xzh/models/eval/qm9_random1000_same_gpu_concurrency/20260714_223057
/scratch/xzh/models/eval/qm9_random1000_cpu_thread_calibration/20260714_224136
```

Slurm 499/500 were invalid wrapper submissions; 501/502 were a duplicate-output submission and
were cancelled. None of those four jobs produced usable scientific results.

## Training Audit

The historical random1000 run named `EGF_lam1` has actual
`model.loss_function.force_loss.weight=0.1`. It is therefore the effective-weight-0.1 baseline, not
an actual weight-1 run. The prepared controlled sweep uses the same seed and trains actual weights
`0.3`, `1.0`, `3.0`, and `10.0`.

Existing geometries are one reference plus three independent Gaussian perturbations. The split is
already parent grouped with 800/100/100 parents and no overlap. Exact paired perturbation support was
added and Slurm 493 completed 1600 labels for the 800 train parents. The combined split contains
4800/400/400 train/validation/test entries, preserves the original validation/test entries, and
rejects any parent overlap.

The added pair for each train parent is `R +/- delta` with the same deterministic direction,
translation removed, Gaussian coordinate scale `0.01 Angstrom` (about `1.89e-2 Bohr`) and per-atom
cap `0.05 Angstrom`. It is therefore a finite local-force-response augmentation at a wider scale
than the `1e-3 Bohr` Hessian audit, not a direct full-Hessian label.

The paired candidate was compute matched to the original EGF runs: 10 epochs, exactly 1233
training batches per epoch, 12330 optimizer updates, the same seed, batch size, and cosine schedule.
This prevented the larger augmented split from receiving more updates. The selected force weight
was actual `3.0`, selected by the independent tier-1 screen. The completed run is:

```text
/scratch/xzh/models/train/runs/qm9_random1000_paired_egf_forcew3p0_computematched_20260715_055837
```

It finished in 49:18 with final checkpoint `checkpoints/epoch_009.ckpt`. The throughput callback
reports 162.9 global samples/s over 12330 steps and 394560 sampled training examples. The final
validation-ground-state force evaluation used by Tier 2 gives energy MAE `0.024181` and force
component MAE `0.002275`. This is only a 0.7% force improvement over non-paired actual weight 3.0
(`0.002292`) and its energy MAE exceeds the pre-registered 10% historical-EGF eligibility limit;
it is therefore not eligible to advance to Test100 under the frozen Tier-2 rule.

### Independent Selection Protocol

The first tier-1 launcher draft used Test100 for force and PBE-Hessian ranking. That would be test
set reuse for hyperparameter selection even without parent leakage, so it has been corrected before
job 494 starts:

- `qm9_force_eval.py --split val` evaluates all validation parents;
- 20 validation parents are selected deterministically across the atom-count range;
- fixed-density autograd HVPs along their three existing perturbation directions are compared with
  PBE force secants from validation labels;
- force weight selection uses validation energy, validation force, and validation HVP/secant only;
- no newly trained candidate is evaluated on Test100 until the rule and top 1--2 candidates are
  frozen;
- Slurm 507 is the completed final-density one-model runtime smoke for this validation funnel.
  The earlier 506 run exercised the all-SCF compatibility path and is not used for selection.

No PBE analytic Hessian reference from Test100 is used in tier-1 selection. Test100 was already
used to diagnose the numerical/force-definition protocol with EG and historical EGF, so it is not
a pristine protocol-development holdout. The defensible independence claim is narrower: no
candidate weight or paired-augmentation decision sees its Test100 outcome before freezing.

Validation smoke 507 results for historical EGF (`actual force weight=0.1`):

| quantity | result |
|---|---:|
| final-density validation geometries | 400 |
| energy MAE / RMSE | 0.021247 / 0.030134 |
| force component MAE / RMSE | 0.003624 / 0.006076 |
| force vector MAE | 0.007390 |
| full-Hessian autograd-vs-FD relative Fro, `0000751` | `6.67e-5` |
| mean HVP-vs-PBE-force-secant MAE, 3 directions | 0.064141 |
| mean HVP-vs-PBE-force-secant relative Fro | 0.858552 |

The HVP is finite and agrees with model finite differences; its comparison with PBE force secants
is explicitly a fixed-density directional curvature proxy because the PBE force response includes
KS density relaxation.

### Tier-1 Weight Selection

Replacement Slurm 517 completed the pre-registered validation-only screen. All 120 full
fixed-density autograd Hessians and all 480 HVP cases were finite. Mean HVP-vs-own-FD relative
Frobenius error ranged from `2.27e-4` to `4.57e-4`, so the directional implementation remained
self-consistent. The selection metrics are:

| actual force weight | validation energy MAE | force component MAE | HVP vs PBE secant MAE | HVP relative Fro | energy eligible | Tier-1 outcome |
|---:|---:|---:|---:|---:|:---:|---|
| 0.1 historical | 0.021261 | 0.003624 | 0.025506 | 0.5186 | yes | baseline |
| 0.3 | 0.022246 | 0.002934 | 0.023314 | 0.4774 | yes | not advanced |
| 1.0 | 0.023097 | 0.002496 | 0.021242 | 0.4342 | yes | second Tier-2 candidate |
| 3.0 | 0.022768 | 0.002292 | 0.018609 | 0.3784 | yes | first Tier-2 candidate |
| 10.0 | 0.024450 | 0.002117 | 0.017443 | 0.3472 | no | rejected by energy gate |

The eligibility limit is `1.10 * 0.0212605 = 0.0233866`. Thus weight 10 was not selected despite
its best force/HVP proxy values. Weight 3 was supplied automatically to the train-parent paired
candidate; weights 3 and 1, plus that paired candidate and both historical controls, proceeded to
strict Val8. No Test100 result entered this decision.

Artifact:

```text
/scratch/xzh/models/eval/qm9_random1000_lambda_tier1/20260714_234814
```

### Tier-2 Strict Val8 Freeze

Slurm 537 evaluated the two selected weights, the train-parent-only paired candidate, historical
effective-weight-0.1 EGF, and EG on the independent eight-molecule validation Hessian set. Every
one of the 858 displaced density optimizations for each model reached the strict `1e-4` projected
density-gradient target. Candidate eligibility retained the pre-registered energy limit
`1.10 * 0.02124466 = 0.0233691` measured on this validation subset.

| model | energy MAE | force component MAE | symmetric Hessian MAE | Hessian RMSE | relative Fro | max asymmetry | frozen for Test100 |
|---|---:|---:|---:|---:|---:|---:|:---:|
| EG | 0.022520 | 0.067729 | 0.077051 | 0.219956 | 3.2037 | 0.472145 | control |
| historical EGF, weight 0.1 | 0.021245 | 0.003624 | 0.015520 | 0.048505 | 0.6600 | 0.270424 | control |
| EGF, weight 1.0 | 0.023099 | 0.002496 | 0.013372 | 0.043778 | 0.5959 | 0.254044 | **yes** |
| EGF, weight 3.0 | 0.022767 | 0.002292 | 0.012200 | 0.036986 | 0.5075 | 0.256502 | **yes** |
| paired train-parent EGF, weight 3.0 | 0.024181 | 0.002275 | 0.012128 | 0.036699 | 0.5004 | 0.247006 | no: energy gate |

The paired candidate has the best validation force/Hessian values by a small margin, but its energy
MAE exceeds the frozen limit. It is therefore not claimed as an improved model and was not passed
to Test100. Test100 candidates were frozen as actual-force-weight 3.0 and 1.0 before any new
candidate Test100 result was read.

Artifact:

```text
/scratch/xzh/models/eval/qm9_random1000_validation_tier2_strict_hessian/20260715_064755
```

The Tier-2 and Test100 analysis scripts were also hardened to derive strict counts, fallback counts,
and cycle statistics directly from per-point optimization rows when evaluator summaries omit those
aggregate fields. Focused synthetic contract tests cover both analyzers.

### Frozen Test100 Confirmation

Slurm 538 evaluated the frozen candidates once on all 100 test parents. EG and historical EGF
density-relaxed results were reused from the immutable baseline artifact; the two new candidates
were recomputed with the same `h=1e-3 Bohr`, strict `1e-4` protocol. All 400 fixed-density full
autograd Hessians and 1600 HVP cases were finite. Each candidate completed 100/100 relaxed Hessians,
100/100 strict base densities, and 10638/10638 strict displaced densities.

| model | energy MAE | force MAE | fixed H MAE | HVP/secant MAE | relaxed H MAE | relaxed RMSE | relaxed rel Fro | symmetry max | frequency MAE cm-1 | overlap | imaginary modes | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| EG | 0.026129 | 0.070257 | 0.066813 | 0.132938 | 0.070755 | 0.207097 | 3.0464 | 0.444027 | 1779.9 | 0.6361 | 1265 | control |
| historical EGF, weight 0.1 | 0.026314 | 0.003578 | 0.010723 | 0.022384 | 0.012074 | 0.037043 | 0.5163 | 0.126402 | 575.2 | 0.6182 | 1260 | reference |
| EGF, weight 1.0 | 0.027573 | 0.002463 | 0.009316 | 0.019276 | 0.010291 | 0.031211 | 0.4301 | 0.089227 | 524.8 | 0.6312 | 1227 | **stable candidate** |
| EGF, weight 3.0 | 0.029096 | 0.002207 | 0.008497 | 0.017320 | 0.009525 | 0.027463 | 0.3814 | 0.108941 | 491.1 | 0.6333 | 1175 | Pareto only; energy gate failed |

Relative to historical effective-weight-0.1 EGF, actual weight 1.0 changes energy, force,
fixed-Hessian, and strict relaxed-Hessian MAE by `+4.79%`, `-31.16%`, `-13.12%`, and `-14.77%`.
Frequency MAE falls 8.76%, mean mode overlap rises from 0.6182 to 0.6312, and model imaginary modes
fall from 1260 to 1227. It therefore passes the frozen requirements of lower force and relaxed
Hessian error, energy MAE no more than 10% worse, zero failed molecules, and all displacement points
strict. Actual weight 3.0 improves force and relaxed Hessian MAE by 38.32% and 21.11%, but its energy
MAE ratio is `1.10573`, just above the `1.10` gate; it is not the formal stable candidate.

Per-molecule robustness supports rather than hides the aggregate result:

- actual weight 1.0 improves relaxed-Hessian MAE on 95/100 molecules; its median candidate/historical
  ratio is 0.8421 and worst ratio is 1.1096 on `0072602`;
- actual weight 3.0 improves 98/100, with median ratio 0.7916, but remains energy-ineligible;
- `0000777` has the largest weight-1 MAE (`0.03203`) but low symmetry error (`0.03138`), so it is a
  model-error hard case rather than a density-curl outlier;
- `0040728` remains the structural hard case: weight-1 MAE `0.02283`, relative Frobenius `1.3275`,
  and symmetry max `0.97534`, despite every point meeting the strict threshold. Tight-threshold and
  loop audits already showed that this cannot be repaired by post-hoc symmetrization.

Fixed-density screening and strict relaxation are correlated but not interchangeable. Across all
four models, fixed and relaxed evaluation choose the same best model for 86/100 molecules. For
actual weight 1.0, fixed-vs-relaxed per-molecule MAE has Pearson/Spearman correlation
`0.8917/0.8975`; fixed error is lower on 96/100 molecules. This supports fixed autograd/HVP as a
cheap ranking funnel while requiring strict relaxation for final proxy confirmation.

Final artifact and machine-readable evidence:

```text
/scratch/xzh/models/eval/qm9_random1000_frozen_test100/20260715_081030
analysis/test100_summary.json
analysis/test100_model_summary.csv
analysis/test100_density_relaxed_per_molecule.csv
analysis/test100_fixed_vs_relaxed_per_molecule.csv
analysis/test100_hvp_per_direction.csv
analysis/test100_vibrational_per_molecule.csv
analysis/test100_tradeoff_ratios.png
analysis/test100_candidate_relaxed_scatter.png
density_relaxed/optimization_points.csv
```

### Final Performance Profile

| stage | parallelism | wall time | host memory | observation |
|---|---|---:|---:|---|
| Test100 energy/force, four models | four GPUs | 57 s | 1.35--1.42 GiB/worker | not limiting |
| fixed full autograd + FD/HVP, four models | one process/GPU | 31:19 | 1.72 GiB | underused seven GPUs in executed run |
| candidate strict relaxed Hessian | 16 workers, two/A100 | 2:12:44 | 28.2--30.5 GiB/worker | dominant stage |
| complete frozen job | eight A100 | 2:45:38 | Slurm aggregate peak 475 GiB | 22.1 allocated GPU-hours |

The strict stage used coordinate-cost-balanced shards. Worker wall times were 1:52:18--2:12:43,
an 18% max/min spread; sampled aggregate CUDA memory was about 2.3 GiB per GPU, while prepared
PySCF geometry/integral caches dominate host memory. Compared with the earlier two-model baseline
wall of 4:11:56, two workers per GPU reduce the same class of strict work by about 1.90x. The new
launcher now evaluates independent fixed-density models concurrently and merges their summaries;
the executed 31-minute serial fixed stage is retained as the measured baseline for that change.

PBE timing is not clean enough for a formal OFDFT speedup claim. The final GPU4PySCF recovery log is
168 s but loaded 92/100 references from cache. Fresh five-molecule validation measured 29.98--53.71
s per molecule, mean about 39.8 s. Even an ideal eight-way extrapolation is much shorter than the
132.7-minute strict OFDFT proxy stage. Therefore current density-relaxed OFDFT has **no demonstrated
speed advantage** over analytic GPU4PySCF/PBE Hessians; fixed-density autograd remains the only
practical fast screening path. Future memory work should bound prepared-geometry cache lifetime,
and a shared task queue should replace static shards if hard-case cycle counts become less
predictable at larger scale.

## Protocol Decision

The completed evidence resolves the full random1000 pilot gates:

1. Tightening `1e-4 -> 1e-6` does not remove smooth-case curl or asymmetry. Use `1e-4` for routine
   strict screening and `1e-5` for representative confirmation; reserve `1e-6` for diagnosis.
2. Fixed-density float64 autograd is the preferred Tier 1 Hessian/HVP path. For force finite
   differences, `h=1e-3 Bohr` is the reliable center of the numerical plateau; `3e-4` amplifies
   rounding/solver noise and `3e-3` shows more truncation on hard cases.
3. The current density-relaxed result is formally frozen as an **incomplete-derived-force proxy**,
   not a physical total-OFDFT Hessian. Raw and symmetric errors are both reported, but
   symmetrization is never described as a repair.
4. Use actual force weight 1.0 as the next-model baseline. Weight 3.0 is an energy/Hessian Pareto
   point, not the formal winner. The plain train-parent paired augmentation is rejected because it
   failed the independent Val8 energy gate before Test100.
5. The formal funnel is validation energy/force plus float64 fixed autograd/HVP, independent strict
   Val8, then one frozen Test100 confirmation. Parent grouping and the frozen test boundary remain
   mandatory. Response extrapolation is optional and is not part of the formal protocol.

## Error-Source Evidence Chain

| candidate source | quantitative test | finding | role in current result |
|---|---|---|---|
| float32/float64 | fixed-density full autograd comparison | relative difference `2.9e-6`--`9.8e-6` | negligible for observed relaxed asymmetry |
| force finite-difference error | fixed autograd vs FD | relative Fro about `1.3e-4` at `h=1e-3` | small; fixes preferred step |
| density residual | loop threshold `1e-4 -> 1e-6` | smooth-case curl changes about 0.1%, residual changes 100x | not leading smooth-case cause |
| displacement | `3e-3,1e-3,3e-4` | five regular representatives stable; `0040728`, `0060531`, `0093887` step sensitive | hard-case uncertainty; `1e-3` formal compromise |
| incomplete total force | fixed vs relaxed loops; relaxed scalar-energy blocks | fixed loops close; relaxed curl equals Hessian asymmetry; force and total-energy blocks disagree strongly | dominant source of non-conservativity |
| model approximation | symmetric Hessian vs PBE after numerical controls | substantial residual MAE remains, especially EG | genuine model/target error to optimize |

Thus the existing comparison is numerically reproducible as an incomplete-derived-force proxy, but
it is not physically equivalent to a conservative total-OFDFT Hessian. Model improvements reported
below must retain that naming until the tensor total-energy and KKT response path is implemented.

## Curvature-Supervision Design Gate

The available labels and derivative order favor a staged comparison:

| method | target | training derivative cost | current status |
|---|---|---|---|
| paired geometry force augmentation | `F(R+delta)`, `F(R-delta)` separately | same mixed parameter/coordinate double backward as EGF | completed; rejected by Tier-2 energy gate |
| explicit local force secant | `(F+ - F-)/(2 delta)` from the same pair | two force forwards plus paired batching; no PBE Hessian required | next ablation if plain augmentation is insufficient |
| base-geometry HVP loss | `H(R)v` | differentiating a nuclear second derivative with respect to parameters introduces third-order autograd | evaluation/screening only for now |
| complete Hessian loss | all coordinate HVPs | `O(3N)` HVPs and third-order parameter gradients | rejected for random1000 training; Test100 references must not leak |

This makes local paired force/secant supervision the only practical curvature-training path in the
current data budget. Fixed-density HVP remains a cheap evaluation funnel, not a training target.

## Recommended Next Implementation

The next model iteration should start from actual force weight 1.0, not the historical mislabeled
run. If curvature supervision is pursued, use the existing train-parent-only exact `R+/-delta`
pairs to train a local force-secant objective while retaining a stronger energy constraint or
multi-objective early stopping. The plain paired-geometry candidate showed that adding samples
alone is insufficient: its Val8 force/Hessian gain was marginal and its energy gate failed. HVP
and full-Hessian parameter training remain unattractive because they require third-order parameter
autograd and would create a much larger memory/cost surface.

The higher-priority physics implementation is conservative total OFDFT differentiation:

1. Replace scalar `.item()` boundaries with a tensor total-energy API.
2. Add coordinate JVP/VJP providers for electron-nuclear, Hartree/Coulomb, overlap and basis
   normalization, nuclear repulsion, moving-basis Pulay terms, and the electron-number constraint.
3. Validate first derivatives against strict relaxed scalar-energy central differences and require
   loop work per area to vanish as numerical tolerances tighten.
4. Expose projected coefficient-Hessian and mixed coordinate-density HVPs, solve the constrained
   response with tangent-space CG or full-KKT MINRES, and log residuals/conditioning.
5. Only after those gates, validate total Hessian symmetry, frequencies, imaginary modes, and mode
   overlap. Do not use post-hoc symmetrization as a substitute.

The current quantitative answer is therefore narrow but defensible: actual weight 1.0 genuinely
improves independent Test100 energy-constrained force and incomplete-derived-force curvature proxy
metrics without parent leakage. It does not yet demonstrate a physical total-OFDFT Hessian
improvement, because that conservative derivative object is not implemented.

## Superseding implementation status: conservative total derivative and scalar secants

The last sentence above is retained as the historical decision at the time of the proxy study.
The complete scalar tensor total-energy force, overlap/Pulay path, strict loop acceptance, KKT
density response and strict total-Hessian evaluator have since been implemented and locally
validated; see `docs/qm9_total_ofdft_conservative_force_hessian.md`. Historical Test100 matrices
in this report remain incomplete-derived-force proxies and are not retroactively renamed.

The scalar-secant-only candidate completed but was rejected because its Test100 force error and all
ten fixed-density proxy Hessians regressed sharply. The retained parent-safe candidate keeps actual
force weight 1.0 and adds scalar energy-secant weight 0.01. It completed ten 8-A100 epochs in
Slurm 880. Independent Test100 energy/force improve by 21.1%/11.6%, and all ten fixed-density
proxy Hessians improve.

Strict complete-total confirmation is now complete. Candidate improves complete directional `Hv`
MAE on 4/5 representative molecules and lowers mean MAE/RMSE by 11.5%/13.7%; `0003027` regresses
9.46% and remains mandatory in future gates. On full `0000777`, the candidate lowers MAE/RMSE/
relative Frobenius by 21.8%/26.4%/26.4%, with the ordering stable at `h=1e-5` and `3e-5 Bohr`.
The candidate is therefore retained as a random1000 improvement trend, not a universal or final
QM9 model.

The final frozen coordinate-0 Test100 complete-total HVP run strengthens this decision: candidate
wins 92/100 molecules and lowers mean MAE/RMSE/relative Frobenius by 27.3%/26.1%/13.5%. Eight
regression IDs are retained in the final report; no additional tuning was performed after exposing
these Test100 outcomes.

The final report, machine-readable inputs and plots are in
`docs/qm9_random1000_total_ofdft_hessian_model_optimization.md` and
`/scratch/xzh/models/eval/qm9_random1000_force_secant_total_analysis/20260715`.
