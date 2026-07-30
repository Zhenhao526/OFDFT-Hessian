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

## Step-100 continuation

The same scratch trajectory was resumed from the complete step-20 Lightning
checkpoint, including optimizer state, and continued to global step 100. The
step-20 checkpoint was retained unchanged for comparison.

- Source step-20 checkpoint SHA-256:
  `ab7070d1739f57684a4955ebab9dcce903073b039721fdb391774b8989f71c1e`.
- Final step-100 checkpoint SHA-256:
  `a275c5068ce0447917572872c181f7d5db697e88cc6ab2d505a7b921f6cc4824`.
- Additional optimizer steps: 80.
- Same 10 parents, 30 geometries, loss definitions, weights, seed, and paired
  batching.
- No validation or Test100 access.

The table compares steps 20-59 with steps 60-99. Values are the configured
unreduced loss components before applying their scalar weights.

| Component | Steps 20-59 | Steps 60-99 | Relative change |
|---|---:|---:|---:|
| Total weighted loss | 0.213970 | 0.148118 | -30.8% |
| E | 1.636408 | 0.984100 | -39.9% |
| G | 0.049352 | 0.049377 | +0.05% |
| F | 0.004489 | 0.004035 | -10.1% |
| H | 0.635783 | 0.617153 | -2.93% |

Longer optimization therefore continued to fit E and modestly improved F, but
did not materially fit G or the conservative-force secant H target.

## Same-molecule strict Hessian and frequency comparison

To test whether the small H-loss change corresponded to useful curvature, the
step-20 and step-100 checkpoints were evaluated on the same training parent,
`0016298`, with an identical strict protocol:

- construct the complete `45 x 45` Cartesian total-OFDFT Hessian;
- use centered force differences at `1e-4 Bohr`;
- independently optimize the density at all 90 displaced points;
- require a final projected density-gradient norm below `1e-8`;
- compare with the analytic PBE Hessian at the identical center geometry;
- symmetrize, mass weight, remove six translation/rotation modes, and match
  vibrational modes by maximum absolute eigenvector overlap.

All 90 step-100 displacement points met the strict criterion. The maximum
final projected density-gradient norm was `9.92e-9`.

| Metric | Step 20 | Step 100 | Relative change |
|---|---:|---:|---:|
| Relative Frobenius | 23.9632 | 24.0841 | +0.50% |
| Frequency MAE (cm^-1) | 4940.45 | 4931.52 | -0.18% |
| Frequency RMSE (cm^-1) | 6454.94 | 6442.33 | -0.20% |
| Frequency max error (cm^-1) | 17997.48 | 17975.46 | -0.12% |
| Mean mode overlap | 0.57963 | 0.57947 | -0.03% |
| Model imaginary modes | 25 | 25 | unchanged |

The PBE reference has two imaginary modes under the same projection and
threshold convention.

## Updated conclusion

Extending this configuration from 20 to 100 optimizer steps does **not**
produce a meaningful Hessian or vibrational improvement. The tiny frequency
changes are at the sub-percent level, Relative Frobenius becomes slightly
worse, the mode overlap does not improve, and the imaginary-mode count is
unchanged. Because the density optimization and force-derived Hessian are
strictly converged, this negative result reflects the learned curvature rather
than a failed evaluation.

The next experiment should not simply add more steps with the same weights.
It should first change the optimization geometry of the multi-objective
problem: normalize or balance component gradients, substantially increase the
effective G/H influence, and supervise multiple independent displacement
directions per parent. The same one-parent strict Hessian comparison is a
suitable cheap gate before any larger evaluation.

## Released-article QM9 weight warm-start

A subsequent test replaced random initialization with the released
Structures25 QM9 model while keeping the EGFH10 dataset, loss weights, batch
construction, seed, learning rate, and 100-step budget fixed. Only model
weights were loaded; optimizer, scheduler, epoch, and global-step state were
fresh.

```text
source:
/home/shenwei01/xzh_node02_20260724/runtime_parent/_runtime/models/train/runs/trained-on-qm9/checkpoints/last.ckpt

source SHA-256:
9759da26660c619de9c3bbf4c2dc164343ee90e08e22b3fcdcc9682dacb9bd09

weight_ckpt_path:
/home/shenwei01/xzh_node02_20260724/runs/qm9_graphformer_egfh10_article_warmstart_s100_v1/article_weight_adapter/article_qm9_current_compat.ckpt
```

The released checkpoint contains eight obsolete TensorFrames `odd_tensor`
buffers that are no longer registered by the current Graphformer. Strict
loading initially failed on exactly those eight unexpected keys. A
hash-recorded compatibility copy removed only those non-trainable buffers;
all remaining state was then loaded with strict `load_state_dict`. No
trainable parameter was removed or ignored.

- Adapted weight checkpoint SHA-256:
  `aafbdb63edc0a34fa7687ca97b55b73aa2e2a3c3bf4e6ce2b255c5b3bbe347b1`.
- Adapter manifest SHA-256:
  `3e4cf7e47a90a7fd4b98e3bea9497fa3f6ab85c1ed5c4b13c4cc8f9fc269e3dc`.
- Final warm-start step-100 checkpoint SHA-256:
  `26e244691456c42f2ed1000583da690a9b2b3eefdf1ddd22d9c94ea8b7dc8b85`.

### Training comparison

The table compares the second half of each 100-step run. For scratch this is
the recorded steps 60-99 of the continued trajectory.

| Component | Scratch-100 | Article warm-start-100 | Relative warm-start level |
|---|---:|---:|---:|
| Total weighted loss | 0.148118 | 0.037027 | 25.0% |
| E | 0.984100 | 0.035390 | 3.60% |
| G | 0.049377 | 0.027398 | 55.5% |
| F | 0.004035 | 0.008424 | 208.8% |
| H | 0.617153 | 0.314562 | 51.0% |

The final warm-start logged values at step 99 were E/G/F/H =
`0.040596/0.022097/0.005245/0.198860`. Thus the released weights greatly
improve E, G, and the one-direction H objective, while F is not uniformly
better than the scratch run.

### Same-molecule Hessian and frequency comparison

The exact same `0016298` full-Hessian protocol was run for the warm-start
checkpoint.

| Metric | Scratch-100 | Article warm-start-100 | Relative change |
|---|---:|---:|---:|
| Relative Frobenius | 24.0841 | 1.25533 | -94.79% |
| Hessian element MAE (Ha/Bohr^2) | 0.554465 | 0.045389 | -91.81% |
| Hessian element RMSE (Ha/Bohr^2) | 2.044630 | 0.106572 | -94.79% |
| Frequency MAE (cm^-1) | 4931.52 | 1616.39 | -67.22% |
| Frequency RMSE (cm^-1) | 6442.33 | 1885.34 | -70.74% |
| Frequency max error (cm^-1) | 17975.46 | 3216.99 | -82.10% |
| Mean mode overlap | 0.57947 | 0.59090 | +1.97% |
| Model imaginary modes | 25 | 25 | unchanged |

The Hessian remained nearly conservative numerically:
antisymmetric/symmetric Frobenius ratio `9.97e-6` and raw symmetry maximum
error `3.09e-5 Ha/Bohr^2`.

### Density-convergence and cost caveat

This result is a strong positive screening signal but is not yet a fully
strict accepted metric:

- 77/90 displaced points finished below the `1e-8` projected-gradient gate;
- the worst final displaced gradient norm was `1.68e-8`;
- the remaining 13 points were close to, but above, the strict threshold;
- mean displaced optimization cycles were `1238.99`;
- wall time was `3222.91 s`.

For the matched scratch-100 evaluation, the corresponding mean was `104.16`
cycles and wall time `843.46 s`. The warm-start evaluation therefore required
about 11.9 times as many optimization cycles and 3.82 times the wall time.
Several individual Cartesian displacements triggered long fallback paths.

### Warm-start conclusion

Released-article weight initialization is substantially better than random
initialization for learning useful full curvature on this pilot. It should
replace scratch initialization as the leading branch. However, it is not yet
an accurate vibrational model: Relative Frobenius remains above one,
frequency MAE remains above `1600 cm^-1`, the imaginary-mode count is
unchanged, and strict density convergence is incomplete.

The next warm-start experiment should prioritize density-response stability
and coverage rather than merely adding steps: test a lower fine-tuning
learning rate, balance F/H gradient contributions, and add multiple independent
internal displacement directions per parent. The same one-parent evaluation
should be rerun until all 90 points pass the fixed `1e-8` gate before
expanding to more molecules.

## Article warm-start continuation to step 200

The article-warm-start step-100 checkpoint was subsequently resumed as a full
Lightning checkpoint, including optimizer, scheduler, epoch, and global-step
state, and continued for another 100 optimizer steps. This is the same
trajectory rather than another weight-only restart.

- Source step-100 checkpoint SHA-256:
  `26e244691456c42f2ed1000583da690a9b2b3eefdf1ddd22d9c94ea8b7dc8b85`.
- Final step-200 checkpoint SHA-256:
  `0b67be97d9793af912ba53aaef28b525669c3fab0e316da2bd7efdad0e1955fb`.
- The log records `Restored all states` from the step-100 checkpoint and
  terminates at `max_steps=200`.
- The dataset, ten parents, 30 geometries, E/G/F/H definitions, scalar
  weights, seed, and batching protocol were unchanged.
- No validation, Test100, or full-Hessian evaluation was accessed.

The table uses adjacent 50-step means. Total is the weighted training
objective; E/G/F/H are the configured component values before their scalar
weights.

| Component | Steps 50-99 | Steps 100-149 | Steps 150-199 | Change in last two windows |
|---|---:|---:|---:|---:|
| Total weighted loss | 0.037027 | 0.026952 | 0.025265 | -6.26% |
| E | 0.035390 | 0.026205 | 0.037760 | +44.09% |
| G | 0.027398 | 0.020167 | 0.016676 | -17.31% |
| F | 0.008424 | 0.006055 | 0.006264 | +3.46% |
| H | 0.314562 | 0.214318 | 0.188423 | -12.08% |

The total objective is still decreasing, but the rate has slowed: it fell
27.21% from steps 50-99 to 100-149 and only another 6.26% from 100-149 to
150-199. G and the force-secant H objective continue to improve, by 17.31%
and 12.08% respectively in the last comparison. E and F do not show the same
monotonic behavior: E rises 44.09% and F rises 3.46% in the last window.

The weighted component-gradient norms recorded at steps 100-119 average
4.07/0.752/3.30/0.750 for E/G/F/H respectively. Pairwise cosines remain small
(absolute mean at most about 0.09), so there is no evidence of a single strong
direct conflict. The current behavior is better described as a
multi-objective plateau or trade-off: the aggregate objective still improves
through G/H while E and F fluctuate or partially regress.

This justifies retaining step 200 as a useful checkpoint, but not extending
the same ten-molecule training indefinitely on training loss alone. The next
decision should use a fixed held-out force/secant set and should test lower
learning rate or explicit gradient balancing before another long
continuation.

## Training-loss convergence continuation

The article-warm-start trajectory was continued beyond step 200 with an
explicit stopping rule rather than a fixed arbitrary endpoint:

> Stop when the absolute relative changes of mean total loss are both below
> 2% for the two most recent adjacent 50-step transitions.

Each 100-step stage was resumed from a complete Lightning checkpoint at an
epoch boundary. Model weights, AdamW moments, trainer epoch, and global step
were preserved throughout. The original cosine schedule was allowed to finish
at step 1000. Because continuing a `CosineAnnealingLR` beyond its `T_max`
would increase the learning rate again, two monotone low-learning-rate tails
were then used:

1. `7e-6` from step 1000, with a new 20-epoch cosine tail;
2. `7e-7` from step 1200, with a new 20-epoch cosine tail.

For these transitions, hash-recorded checkpoint adapters changed only the
optimizer learning rate and cosine scheduler state. Model state, optimizer
moments, and trainer progress were not changed. No component definition,
scalar loss weight, example, batch policy, or seed was changed.

### Convergence decisions

| Endpoint | Two latest 50-step total-loss changes | Decision |
|---|---:|---|
| Step 300 | -14.61%, -13.17% | continue |
| Step 400 | -12.96%, -2.43% | continue |
| Step 500 | -5.39%, -8.37% | continue |
| Step 600 | -10.61%, +2.84% | continue |
| Step 700 | -18.78%, -9.11% | continue |
| Step 800 | -8.99%, -10.49% | continue |
| Step 900 | -8.40%, -5.53% | continue |
| Step 1000 | -2.58%, -2.45% | continue; narrowly above threshold |
| Step 1100 | +15.30%, -1.46% | continue; low-LR transition |
| Step 1200 | -8.53%, -7.01% | continue |
| **Step 1300** | **-1.48%, -0.69%** | **converged; stop** |

The controller stopped automatically at global step 1300; step 1400 was not
started.

### Final training-loss levels

| Component | Steps 150-199 | Steps 1200-1249 | Steps 1250-1299 | Change from steps 150-199 |
|---|---:|---:|---:|---:|
| Total weighted loss | 0.025265 | 0.005970 | 0.005929 | -76.53% |
| E | 0.037760 | 0.002928 | 0.002857 | -92.43% |
| G | 0.016676 | 0.005057 | 0.005037 | -69.79% |
| F | 0.006264 | 0.000805 | 0.000791 | -87.38% |
| H | 0.188423 | 0.082646 | 0.082228 | -56.36% |

- Final step-1300 checkpoint SHA-256:
  `7c40f4dfb442f606a278b796e973d2669108b3a306a6b1e065143ee9e3e71545`.
- Step-1200 source checkpoint SHA-256:
  `c957dc33217210a5d683402dd67f97ce41d213126a2ff94fda3bd144660d731f`.
- Ultra-low-LR adapter manifest SHA-256:
  `930b233873502b2ff62b0dd1fe7a6a3514115b0a6ceea6ac361ccba1c4af8c70`.
- No validation, Test100, or full-Hessian evaluation was accessed during this
  convergence run.

This establishes convergence only for the ten-parent training objective. It
does not establish held-out force/secant generalization, full-Hessian
accuracy, or vibrational-frequency accuracy. The converged step-1300
checkpoint should now be frozen and evaluated on a fixed held-out set before
any further optimization.
