# QM9 Structured Density-Conditioned Hessian Head Pilot v1

Date: 2026-07-29 Asia/Shanghai

## Question

Can a small Euclidean-covariant Cartesian block head predict transferable
train-only PBE Hessians, and do converged OFDFT density coefficients add useful
signal beyond molecular geometry?

This is a direct density-conditioned Hessian-head pilot. It is not an analytic
Schur-complement response head: the clean assets do not contain supervised
`E_cc`, `E_cR`, or fixed-versus-relaxed response blocks.

## Frozen protocol

- 20 clean train parents, `sample_id=0` only.
- Five-fold parent-held-out cross-validation; each parent is held out exactly
  once.
- Validation and Test100 are locked and were not accessed.
- PBE Hessian manifest SHA256:
  `acc5d00a984f17f73043aafcc5954fba7722537ab55c82f803a64f99a61f30c1`.
- Clean train-only dataset manifest SHA256:
  `9506bf6573ecdea705555d204c389ce7508735baccc3c13533a7b23c623370ed`.
- Fixed 800-step AdamW training; no held-parent model selection.
- Matched ablations:
  - geometry only: 11,979 parameters;
  - geometry plus density: 13,515 parameters.

The six density features per atom are mean, standard deviation, RMS, mean
absolute value, maximum absolute value, and signed sum divided by the square
root of the coefficient count. Normalization is fitted on the training parents
inside each fold.

## Structure imposed by construction

The head predicts atom-pair and atom-diagonal `3 x 3` blocks from invariant
features and covariant tensor bases. The assembled matrix is symmetrized and
projected into the molecular internal subspace. Unit tests verify:

- atom-permutation covariance;
- rotation covariance;
- exact Hessian symmetry;
- translational and rotational null modes at the reference geometry;
- the geometry-only ablation ignores density values.

All four tests pass. The vectorized implementation reproduces the original
smoke metrics and reduces the two-step smoke wall time from about 6.37 s to
1.80 s on GPU1.

## Formal five-fold result

| Variant | Fit median relF | Held median relF | Held P90 relF | Held max relF | Held better than zero |
|---|---:|---:|---:|---:|---:|
| Geometry only | 0.225775 | 0.249269 | 0.283948 | 0.324290 | 20/20 |
| Geometry plus density | 0.225658 | 0.247198 | 0.288309 | 0.319053 | 20/20 |

Density changes the held median by only `0.8309%`, below the preregistered
`5%` improvement gate. On paired held parents it wins 11 cases and loses 9;
the median absolute relative-Frobenius change is only `0.000348`.

The density-aware head passes the held-median, held-P90, zero-Hessian,
symmetry, and rigid-mode gates. It fails:

- the capacity gate: fit median `0.225658 > 0.15`;
- the density-value gate: improvement `0.8309% < 5%`.

The overall preregistered result is therefore **fail**. The formal run took
`948.19 s` for all ten fold/variant models.

Numerical constraints are satisfied to machine precision:

- maximum symmetry error: exactly `0.0` in the recorded matrices;
- maximum held external-mode leakage: `6.58e-16`.

## Interpretation

The structured output parameterization is technically viable and stable. It
learns a useful geometry/chemistry Hessian prior with a small parameter count
and no catastrophic held-parent tail.

The present density representation does not demonstrate density-response
value. Compressing all auxiliary density coefficients to six invariant
statistics per atom removes angular channels, inter-atomic density transfer,
and the conditioning information carried by the density curvature operator.
The near-identical fit and held results of both ablations show that this pilot
is representation-limited rather than conventionally overfit.

This head must not be described as a physical OFDFT Hessian or integrated as
the final force/Hessian output. An independently predicted Hessian is not
guaranteed to be the second derivative of the scalar total OFDFT energy.

## Decision and next experiment

Do not access validation or Test100 with this v1 head. Do not expand the same
six-statistic density input or run optimizer sweeps.

The next response-aware pilot should separate:

```text
H_relaxed = H_fixed - C^T K^{-1} C
```

and use clean teachers for the fixed block and response correction. A useful
low-rank parameterization is:

```text
H_relaxed = P [H_fixed - B^T B] P
```

where `P` is the internal projector and `B` represents learned density-response
modes. Density inputs should retain low-order equivariant density moments or
projected response modes rather than scalar coefficient summaries. Before
that experiment:

1. generate or validate clean fixed-density and relaxed Hessian pairs on the
   same train parents;
2. save projected `E_cR` directions and a stable constrained `E_cc` solve, or
   an equivalent response correction teacher;
3. enrich the covariant block basis until the train-parent median is below
   `0.15`;
4. keep a matched geometry-only Hessian prior and a scalar-energy/force-secant
   model as controls.

IR intensities require a separate conservative dipole-response target; a
Hessian head alone can supply frequencies and modes, not IR intensities.

## Artifacts

Remote formal artifact:

`/home/shenwei01/xzh_node02_20260724/artifacts/structured_density_hessian_head_pilot_v1_20260729`

Formal summary SHA256:

`f4769f2ac5a90016be023e7e4e3a818f1a4e6ab47fef5111f0fec060afec1f73`

Per-parent metrics SHA256:

`da13c77fd0d4be34c0ba5693f2063a3509ead230d827a37f63267a6399709911`
