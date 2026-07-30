# QM9 EGFH10 scratch pilot (2026-07-30)

## Scope

This pilot tests whether the unmodified Structures25 Graphformer can be trained
from random initialization with four simultaneous conservative objectives:

- **E**: `e_kin_plus_xc`.
- **G**: the particle-number-projected difference between
  `dE_model/dc` and `gradient_label`.
- **F**: `-dE_model/dR` versus the complete PBE force label.
- **H**: the normalized force secant from a symmetric displaced-geometry pair,
  compared with the corresponding complete-PBE force secant.

The H objective is a directional force-Jacobian/curvature constraint. It does
not materialize or supervise a full Cartesian Hessian.

## Frozen protocol

- Ten fixed QM9 parents, with one center and one deterministic symmetric
  displacement pair per parent (30 geometries total).
- PBE/6-31G(2df,p), grid level 3, SCF tolerance `1e-9`.
- `perturbation_std=0.01`, `max_displacement=0.05`, translation removed.
- Full 18.7M-parameter Graphformer; no source checkpoint, `ckpt_path=null`,
  `weight_ckpt_path=null`.
- Train-only pilot; no validation or test100 access.
- Batch size 4, paired endpoints kept in the same batch, 20 optimizer steps.
- Loss weights: E/G/F/H = 0.1/0.8/1.0/0.01.

## Data and run integrity

- Kohn-Sham checkpoints: 30/30.
- Raw labels: 30/30.
- Transformed labels: 30/30.
- Force smoke check: 30 files, 30 force labels, zero failures.
- Maximum reference force norm observed by the smoke check:
  `0.05651970159565817`.
- Split SHA-256:
  `fcc8c788a74419be374abebec167dbf03c8377353d87d32a2eaab204c14bfa2d`.
- Dataset manifest SHA-256:
  `f0396ceeb9ad7f291909c44ea66b042c8f2340e2c248c332804f7fc924175e46`.
- Final checkpoint SHA-256:
  `ab7070d1739f57684a4955ebab9dcce903073b039721fdb391774b8989f71c1e`.
- Lightning stopped normally because `max_steps=20` was reached.
- Final checkpoint size: approximately 215 MB.

## Observed losses

The table compares the mean of steps 0-9 with the mean of steps 10-19.
The values are the configured unreduced component losses before application of
the scalar E/G/F/H weights.

| Component | Steps 0-9 | Steps 10-19 | Relative change |
|---|---:|---:|---:|
| Total weighted loss | 0.992028 | 0.318357 | -67.9% |
| E | 9.395445 | 2.669953 | -71.6% |
| G | 0.049425 | 0.049415 | -0.02% |
| F | 0.006298 | 0.005292 | -16.0% |
| H | 0.664498 | 0.653750 | -1.62% |

All 20 logged values were finite. The H number is the normalized, softly
capped force-secant loss and must not be interpreted as a physical Hessian MAE.

## Gradient-scale audit

The means below are parameter-gradient norms of each already-weighted loss
component during steps 10-19.

| Component | Weighted gradient norm |
|---|---:|
| E | 16.236985 |
| G | 0.001272 |
| F | 0.106969 |
| H | 0.023967 |

E therefore dominated the update by approximately 152x over F, 677x over H,
and 12,762x over G. Pairwise gradient cosines were mostly near zero; F and H
were mildly aligned (mean cosine about 0.22 during steps 10-19). The immediate
failure mode is gradient-scale imbalance, not strong objective conflict.

## Performance

After warm-up, training sustained about 3.4 optimizer steps/s on one GPU.
Peak logged GPU memory was about 859 MB. Full Hessian construction was not
used. PBE data generation dominated wall time; the 20-step optimization itself
took only about ten seconds.

## Conclusion and next experiment

The conservative EGFH path is technically feasible and stable, but the current
fixed weights do not give G or H enough influence to test learnability. This
pilot is not evidence that H has been learned: only E decreased strongly, F
improved modestly, and H was almost flat.

Before increasing molecule count or optimizer steps:

1. keep the scalar-energy Graphformer and conservative force/secant
   construction unchanged;
2. introduce measured gradient balancing or per-target normalization so E
   cannot dominate by hundreds to thousands of times;
3. repeat this same 10-parent, 20-step protocol as an A/B comparison;
4. then add a fixed held-out paired set and multiple independent displacement
   directions per molecule before making any Hessian/frequency claim.
