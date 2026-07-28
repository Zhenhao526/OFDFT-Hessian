# QM9 random1000 HVP branch-stability protocol v1

Frozen: 2026-07-16. Machine-readable configuration:
`configs/audit/qm9_hvp_branch_stability_v1.yaml`.

## Purpose and data boundary

This protocol separates electronic-branch and second-response failures from ordinary model
error before adding further curvature supervision. It is frozen before the new branch audit is
run. Threshold changes require a new protocol version; v1 results are not reclassified
retrospectively.

The source split SHA256 is
`919c5f6c250b0d2e9f14f43522894fcac8b179d6d4fd5b774c2cec1ea249034b`. Energy/force replay uses
all original train800 parents. Curvature supervision is restricted to the previously selected
train100 parents (selection SHA256
`8982fc6349b520b9b98f52fa684e81558b6afab16d14a100f3c3843608503f0b`) that pass this protocol.
Validation uses validation parents only. Test100 labels, metrics and Hessians remain unread until
a candidate and selection rule have been frozen.

The baseline stability surface is the existing EGF force-weight 1.0 model, seed `676368232`.
Candidate models are evaluated on the same frozen stable set and are additionally checked for new
instabilities. Stability is a property of a model/density branch, not of the PBE label alone.

## Audit design

Each parent uses `sample_id=0` and four deterministic unit directions: random internal, bond
stretch, angle or torsion, and a low-frequency direction. Translation and rotation are projected
out. Strict relaxed-force HVPs use `h={3e-5,1e-5,3e-6}` Bohr; `1e-5` is primary. Geometry mixed
derivatives and integral derivatives use `1e-4` Bohr.

The density solver is float64. It runs Adam (`1e-3`, at most 1000 cycles, target `1e-2`), Adam
(`3e-4`, at most 10000 cycles, target `1e-5`), LBFGS (at most 200 evaluations, target `1e-8`) and
up to three Newton refinements (target `1e-8`). Base geometries compare SAD and reference-label
initial densities. Displacements compare independent SAD, base-density continuation and the
available reference/continuation branch. Density agreement is measured in the Coulomb metric,
not by raw coefficient cosine.

The initial eight validation diagnostics are `0000751`, `0027926`, `0044504`, `0059830`,
`0087088`, `0103559`, `0118217` and `0132890`. They cover small, median, large, small-reference and
known hard behavior without consulting Test100.

## Mandatory stability gates

A direction is stable only when every mandatory check passes:

| Check | Frozen v1 threshold |
|---|---:|
| final projected density-gradient norm | `<=1e-8` |
| electron-number constraint residual | `<=1e-8` |
| total-energy spread across converged initializations | `<=1e-5 Ha` |
| Coulomb-metric density cosine | `>=0.9999` |
| relative Coulomb-metric density distance | `<=2e-2` |
| complete-total force RMSE across branches | `<=1e-3 Ha/Bohr` |
| tangent KKT minimum eigenvalue | `>=1e-8` |
| tangent KKT condition number | `<=1e10` |
| response relative residual | `<=1e-8` |
| response stationarity residual | `<=1e-6` |
| response constraint residual | `<=1e-8` |
| HVP step instability | `<=5%`, with `0.05 Ha/Bohr^2` component floor |
| implicit versus strict-relaxed HVP difference | `<=10%`, same scale floor |
| relaxed-energy versus relaxed-force curvature mismatch | `<=5%`, with `0.05 Ha/Bohr^2` floor |

The tangent Hessian must be positive on the electron-number-conserving subspace. An indefinite
stationary point is excluded even when its projected gradient is tiny. A parent is eligible for
regular curvature training only if all of its labelled directions pass. Direction-level failures
remain available for a separate pathology report but receive zero HVP/secant weight and do not
enter normal validation means. Absolute MAE, scale-floored relative error, medians, quantiles and
win fractions are retained so that a small reference norm cannot dominate selection.

## Training comparison

All variants replay identical train800 energy, density-gradient and force data from the same
initial checkpoint, step budget, batch size, learning-rate schedule and seed:

- A: energy + force.
- B: A + secant100.
- C: A + fixed-density HVP100.
- D: A + complete-total relaxed-force secant100.
- E: A + a small, stable subset of implicit complete-total HVP.

Any quantity not differentiated from a scalar total energy must not be called complete-total.
The current SciPy/detached implicit-response evaluator is validation-only. E is admitted only after
a differentiable training implementation passes scalar-energy, response-residual and finite-
difference gradient checks; otherwise E is reported as blocked rather than replaced by a mislabeled
proxy.

HVP directions are resampled deterministically by epoch from the frozen 3--6-direction pool.
Curvature loss starts after an energy/force warm-up and is applied on alternating updates. Weight
and reference-scale-floor scans record per-loss parameter-gradient norms and pairwise cosine
conflicts. Per-parent normalized curvature gradients are clipped before aggregation so a hard
branch or tiny target cannot control a batch.

## Promotion rule

Seeds are `20260716`, `314159` and `676368232`. Relative to matched-seed A, energy MAE may worsen
by at most 5%. Force MAE and stable-only strict complete-total HVP MAE must improve in at least two
seeds, and most stable molecules and directions must win. Representative complete-total Hessian,
frequency, imaginary-mode and mode-overlap diagnostics must not degrade. These rules and the
candidate are frozen before the single permitted Test100 confirmation.

