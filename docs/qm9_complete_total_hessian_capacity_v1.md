# QM9 Complete-Total Hessian Capacity V1

Last updated: 2026-07-24 21:10 Asia/Singapore

## Scope and Frozen Rules

This experiment replaces percent-level tuning of the old direct-HVP candidates with three gated
questions:

1. Can the model fit strict complete-total relaxed curvature on 5 small stable train parents?
2. Can it generalize to held-out directions on about 20 parents?
3. Only after both gates pass, can it generalize to unseen validation parents with train100 HVP
   labels plus train800 energy/force replay?

The formal target is median Hessian relative Frobenius at most 10%, at least 80% of molecules at
most 15%, P90 at most 20%, frequency MAE about 100--200 cm-1, near-PBE imaginary-mode counts,
improved mode overlap, and antisymmetric/symmetric Frobenius below 0.5%. Test100 is frozen and has
not been read by this experiment. Force and curvature always originate from one scalar total
energy; no force head is allowed.

Protocol: `configs/audit/qm9_complete_total_hessian_capacity_v1.yaml`.

Remote root: `/scratch/xzh/models/complete_total_capacity/20260717` on `node01`.

## Analytic Relaxed-HVP V2 Refactor (2026-07-24)

The active engineering path has moved to node02 local `/home`; `/scratch` is abandoned and must
not be used as a fallback. The legacy four-direction strict relaxed-force secant remains the
independent audit oracle, but the proposed training path now uses one center density and one fresh
full-internal-space Rademacher direction per step.

Implemented:

- constrained response `G_y y_v=-G_R v`;
- `H_relaxed v=L_RR v+L_Ry y_v`;
- differentiable direct KKT solve with an implicit parameter adjoint;
- one-probe unbiased internal Hessian Frobenius loss;
- strict center-density predictor/corrector;
- PCG, MINRES, low-rank deflation, block-PCG, and direct-solve benchmarks;
- periodic complete `3N-6` basis evaluation and split timing.

PySCF directional second integral derivatives are currently central differences of complete
first-derivative integral bundles. Thus the density/model response is analytic, while the
external integral-HVP boundary remains numerical. There is no displaced density optimization in
the training HVP and no fixed/stale/detached fallback.

The focused implementation suite passes `44/44`. The node02 formal preflight stopped correctly
before model loading because the exact original-A checkpoint and frozen stable5 assets have not
yet been restored under `/home`. It did not access validation/Test100 or launch stable5/train20.
Consequently the real `0028399` error, parameter-FD, full39, `10x`, and `<60 s/step` gates remain
unmeasured.

Frozen protocol:
`configs/audit/qm9_graphformer_complete_total_analytic_relaxed_hvp_v2.yaml`.

Detailed implementation and unblock record:
`docs/qm9_graphformer_complete_total_analytic_relaxed_hvp_refactor.md`.

## Graphformer Complete-Total Relaxed-HVP Reset (2026-07-23)

The active experiment has returned to the untouched original-A Graphformer. It does not use the
historical fixed-density `kin_plus_xc` HVP trainer, a detached density response, or an auxiliary
force/Hessian head. The frozen protocol is
`configs/audit/qm9_graphformer_complete_total_relaxed_hvp_v1.yaml`. Its scalar owner is

```text
E_total(R,c;theta) =
  E_Graphformer_kin_plus_xc(R,c;theta)
  + E_Hartree(R,c) + E_external(R,c) + E_nuclear_repulsion(R).
```

The training target is a strict relaxed-force secant

```text
H_theta v = -(F_theta(R+h v)-F_theta(R-h v))/(2h),
F_theta = -d E_total(R,c_star(R;theta);theta)/dR,
```

where each displaced density is independently relaxed below projected gradient `1e-8`. A
matrix-free constrained implicit VJP supplies `dc_star/dtheta`; failure raises and cannot fall
back to a detached proxy.

Frozen source:

```text
original-A checkpoint:
/scratch/xzh/models/train/runs/qm9_hvp_curvature_v1_A_w0ep00_f1em02_seed314159_s1200/checkpoints/last.ckpt
SHA256:
e6516b04917a9dfaae3c4d960b10d77c768f3e5e784f95290ff41c1202f9d9bc
```

Geometry-only structured direction banks cover bond, angle, torsion, and random-internal
directions, remove translation/rotation, and complete the `3N-6` internal space. They are frozen
before optimization:

| Set | Parents | Directions | Train/held roles | Manifest SHA256 | Max orthogonality error |
|---|---:|---:|---:|---|---:|
| stable5 | 5 | 228 | 183/45 | `1d9d235719805b3bf298fe3baf783b1b17a262e352b4193388f0be4f9b65bd74` | `7.77e-16` |
| train20 | 20 | 906 | 724/182 | `a902bb554cf05393c14b316e091f182c19bf888317c42aec153da774f02af378` | `8.88e-16` |

Both manifests certify `validation_accessed=false` and `test100_accessed=false`.

### Real derivative verification

Job 4056 verified molecule `0028399`, bond direction 0, `h=1e-3 Bohr`, float64:

| Check | Result | Gate |
|---|---:|---:|
| trainable graph HVP vs independently re-relaxed strict FD, relative L2 | `1.679e-9` | `<=1e-5` |
| parameter gradient vs reoptimized parameter FD, best relative error | `0.02163` | `<=0.05` |
| parameter FD errors at `1e-4/3e-5/1e-5` | `0.02170/0.02163/0.02163` | stable |
| implicit solves | `3421/3468` iterations | converged |
| implicit relative residuals | `2.803e-5/2.962e-5` | `<=3e-5` |
| peak GPU / host RSS | `1.03 GiB / 5.93 GiB` | finite |

The audit selected
`node_embedding_module.shrink_gate.inner_factor[5]`; analytic and FD gradients were
`15.8401` and approximately `16.1903`. All parameter-FD densities were reoptimized, often taking
up to about 10,100 cycles. The full audit took `4335 s`; this is a one-time derivative check, not
per-step training overhead. A 16-probe Hutchinson-Jacobi preconditioner was required for practical
runtime. Eight probes with 2500 iterations failed closed at relative residual `2.994e-4`; an
unpreconditioned attempt was canceled after `1:01:54` without completion.

### Full `3N-6` baseline and symmetry

Job 4058 evaluated all 39 internal directions of `0028399`:

| Metric | original-A |
|---|---:|
| Hessian relative Frobenius | `2.72659885` |
| MAE / RMSE (Ha/Bohr2) | `0.0804137 / 0.2396083` |
| antisymmetric/symmetric Frobenius | `2.921e-5` |
| symmetry max abs (Ha/Bohr2) | `8.664e-5` |
| energy-vs-force directional-curvature MAE (Ha/Bohr2) | `3.010e-5` |
| strict points | `79/79` |
| wall time / host RSS | `1598 s / 9.55 GiB` |

This passes the numerical symmetry gate but fails the capacity target by a wide margin, providing
the registered `~2.7` starting point.

### Stale-density blocker and repair

The first 200-step smoke, job 4059, exposed an invalid block-coordinate ordering. Step 1 used
strict densities, but step 2 built its graph from pre-update density caches with maximum projected
gradient `6.29e-2`. The code only scheduled a full refresh after that invalid graph. Job 4059 was
canceled at step 2 and is not a scientific result.

`scripts/qm9_complete_total_capacity_train.py` now supports and, under
`--require-complete-total-relaxed-hvp`, requires `--strict-active-density-refresh`:

1. determine the next step's frozen direction subset;
2. after every parameter update, strictly re-relax base plus only those active `R+/-h v` points;
3. build all E/F/HVP terms from those current stationary densities;
4. recompute the actual projected gradient on the graph and raise if any value is `>=1e-8`;
5. refresh every direction only for full-Hessian evaluation.

Focused tests now reject stale/nonfinite training densities and preserve the old interval policy
only for legacy runs. The relevant suite passes `27/27`.

### One-parent loss-weight screen and active continuation

The original `lambda_H=1` job 4060 was canceled after the short-screen results showed that its HVP
term was too weak for a useful capacity test. It has no final full-Hessian checkpoint and is not an
accepted result. Five independent 20-step runs were then started from untouched original-A with
identical directions, strict density refresh, and optimizer settings:

| Job | `lambda_H` | Hessian relative Fro, step 0 -> 20 | Energy abs error, step 0 -> 20 (Ha) | Force MAE, step 0 -> 20 (Ha/Bohr) | asym/sym | Gate |
|---:|---:|---:|---:|---:|---:|---|
| 4061 | 3 | `2.72660 -> 2.74695` | `0.12042 -> 0.00158` | `0.11607 -> 0.07641` | `1.20e-5` | fail |
| 4062 | 10 | `2.72660 -> 2.36091` | `0.12042 -> 0.03666` | `0.11607 -> 0.09245` | `1.53e-5` | fail |
| 4063 | 30 | `2.72660 -> 2.16958` | `0.12042 -> 0.08452` | `0.11607 -> 0.10865` | `1.84e-5` | fail |
| 4064 | 100 | `2.72660 -> 2.15708` | `0.12042 -> 0.12071` | `0.11607 -> 0.11586` | `2.04e-5` | fail |
| 4065 | 300 | `2.72660 -> 2.18248` | `0.12042 -> 0.15107` | `0.11607 -> 0.11839` | `2.23e-5` | fail |

All five runs pass the `<1e-8` density-stationarity gate. None passes the required 5% Hessian
capacity gate. Weight 100 is the best curvature point while keeping original-A energy and force
within the 5% regression limit; weight 300 already breaks that limit.

Job 4066 continues only job 4064 from cumulative step 20 to step 200. The continuation checkpoint
contains the exact AdamW state and is bound to the original-A root checkpoint, protocol, direction
manifest, molecule order, strict implicit-response flags, original E/F gate baseline, and loss
weights:

```text
bound checkpoint:
/scratch/xzh/models/graphformer_relaxed_hvp/20260723/v1/capacity/bound/
h100_step20_bound.ckpt
SHA256:
c18d4352584379b2f3607b0057fa633305b0ee8994d705fef060d33852c0876b

active run:
/scratch/xzh/models/graphformer_relaxed_hvp/20260723/v1/capacity/
stage1_one_parent_h100_resume20_to200_stable5_all_s180_lr1e-6_4066
```

It saves every 10 cumulative steps and evaluates the complete 39-direction Hessian every 40
additional steps plus the final step. The launcher fails if resume provenance or loss weights
differ. Stable5 and both train20 arms remain closed unless `0028399` reaches Hessian relative
Frobenius `<=0.05` and original-A energy/force regress by no more than 5%.

New or materially changed files:

```text
configs/audit/qm9_graphformer_complete_total_relaxed_hvp_v1.yaml
mldft/ofdft/internal_directions.py
mldft/ofdft/complete_total_training.py
scripts/prepare_qm9_graphformer_relaxed_hvp_directions.py
scripts/bind_qm9_graphformer_capacity_checkpoint.py
scripts/qm9_graphformer_complete_total_relaxed_hvp_verify.py
scripts/qm9_complete_total_capacity_train.py
scripts/slurm_prepare_qm9_graphformer_relaxed_hvp_directions_v1.sbatch
scripts/slurm_qm9_graphformer_complete_total_relaxed_hvp_verify_v1.sbatch
scripts/slurm_qm9_graphformer_complete_total_relaxed_hvp_capacity_v1.sbatch
tests/ofdft/test_internal_directions.py
tests/ofdft/test_complete_total_training.py
tests/test_bind_qm9_graphformer_capacity_checkpoint.py
tests/test_qm9_complete_total_capacity_train.py
```

## Frozen Stage-1 Data

Source split SHA256:
`8d5fc057a0ff6af2f7d994db7f6517145c00c745416b015caa3dd9afb57f1c7a`.

Manifest:
`/scratch/xzh/models/complete_total_capacity/20260717/stage1_manifest.json`.

Manifest SHA256:
`2d838c64bee4772c518be1ac9f786c2cb84175175d291a469f15772b7c61cebc`.

| Parent | N atoms | PBE energy (Ha) | PBE force RMS (Ha/Bohr) | PBE Hessian RMS (Ha/Bohr2) |
|---|---:|---:|---:|---:|
| 0028399 | 15 | -449.80484099 | 0.005278 | 0.087878 |
| 0031108 | 16 | -470.80760534 | 0.005155 | 0.075777 |
| 0132419 | 16 | -441.68728616 | 0.004395 | 0.077575 |
| 0031012 | 18 | -438.73895042 | 0.004199 | 0.069507 |
| 0121249 | 21 | -460.91769483 | 0.003074 | 0.051386 |

This is the active stable5-v2 list. The original Stage-1 manifest contained `0050129`, but the
pre-fit audit found `||H_asym,baseline||F/||H_PBE||F=0.03678`, so the frozen v2 protocol excludes
it and uses the final existing all-four-direction-stable candidate `0121249`. All active parents
are train100-only and pass the v2 numerical stability gate. Stage 1 consumes zero validation
parents and zero Test100 records.

## Definition Audit

The formal scalar owner is

```text
E_total = E_model_kin_plus_xc + E_Hartree + E_external + E_nuclear_repulsion.
```

Strict force is the coordinate derivative of the constrained scalar Lagrangian at a density with
projected gradient below `1e-8`. Formal HVP/full Hessian acceptance remains finite difference of
strictly relaxed scalar-derived forces.

The previous direct-HVP trainer differentiated only `E_model_kin_plus_xc`; therefore its training
force/HVP was not the formal complete-total target. The new capacity trainer fixes scalar-energy
assembly, but its first block-coordinate implementation detached the optimized density from model
parameters. A strict post-update audit shows that this omission changes the HVP optimization
direction.

For diagnosis, the trainer also computes

```text
q(v) = [E_relaxed(R+h v) - 2 E_relaxed(R) + E_relaxed(R-h v)] / h^2.
```

This is the scalar contraction `v^T H v`, not vector `H v`. At a strict density stationary point,
its model-parameter gradient follows directly from the envelope theorem and does not require
backpropagating through the density optimizer. It is used to test capacity and gradient semantics,
not relabelled as HVP.

## Completed One-Step Audits

All rows use parent `0028399`, Cartesian direction 0, `h=1e-3 Bohr`, float64, strict density
refresh, one AdamW update, gradient clipping at 1.0, and the same force-weight-1 control checkpoint.

| Job | Training graph | LR | HVP grad norm | Q grad norm | Strict relative column before | after | Result |
|---:|---|---:|---:|---:|---:|---:|---|
| 3128 | E+F+Euler+10 HVP | 3e-6 | 2262.9 | n/a | 3.43132 | 3.52871 | worse |
| 3129 | pure HVP, detached density | 3e-7 | 226.3 | n/a | 3.43132 | 3.43938 | worse |
| 3130 | pure HVP, 10-step density unroll | 3e-7 | 175.7 | n/a | 3.43132 | 3.43949 | worse |
| 3132 | pure HVP, connected multiplier only | 3e-7 | 226.3 | n/a | 3.43132 | 3.43938 | worse |
| 3133 | pure relaxed-energy curvature | 3e-7 | 0 | 670.2 | 3.43132 | 3.39182 | improved |
| 3136 | pure HVP, implicit response at 1e-3 | 3e-7 | 83.2 | n/a | 3.43132 | 3.39425 | improved |
| 3141 | pure HVP, implicit response at 3e-5 | 3e-7 | 90.4 | n/a | 3.43132 | 3.39250 | improved |
| 3138 | pure relaxed-energy curvature | 3e-6 | 0 | 670.2 | 3.43132 | 3.13613 | improved |
| 3140 | E + F + Euler + relaxed-energy curvature | 3e-6 | 0 | 670.2 | 3.43132 | 3.23058 | all E/F/H improved |

For job 3133, strict energy-curvature absolute error decreased from `2.57080` to
`2.54153 Ha/Bohr2`. Energy-second-difference and force-secant directional curvatures differ by only
`0.00204 Ha/Bohr2` after the update. The complete-total force MAE also decreased from `0.11607` to
`0.11573 Ha/Bohr`, while pure-Q total-energy absolute error increased from `0.12042` to
`0.12140 Ha`; balanced replay is therefore still required.

Job 3130 used 3.0 GiB peak GPU memory and 7.0 GiB host RSS in 173 s. Job 3133 used 1.4 GiB peak
GPU memory and 7.1 GiB host RSS in 144 s. Job 3131 tests 100-step unroll and was still running when
the first snapshot was written. It completed in 20:30 with 16.6 GiB peak GPU memory; the strict
column error changed only `3.43132 -> 3.43145`. Unrolling is therefore too costly and still less
accurate than the implicit VJP.

The matrix-free implicit VJP was verified against an analytic constrained quadratic unit test.
On the real 1136-coefficient parent, unpreconditioned PCG reached relative residual `5.94e-4` in
500 iterations and `2.75e-5` in 2000 iterations. A strict `3e-5` run converged in at most 1467
iterations for each of two displaced points and reproduced the favorable update direction. An
8-probe Hutchinson-Jacobi preconditioner worsened the 500-step residual to `2.73e-3`, so it is
implemented only as an opt-in audit and disabled by default.

The 25-step one-direction pure-Q run 3134 completed in 28:24. Strict relative column error fell
monotonically at evaluation points:

| Step | Relative column Frobenius | Energy-curvature MAE (Ha/Bohr2) | Energy abs. error (Ha) | Force MAE (Ha/Bohr) |
|---:|---:|---:|---:|---:|
| 0 | 3.43132 | 2.57080 | 0.12042 | 0.11607 |
| 5 | 3.26118 | 2.44386 | 0.12540 | 0.11441 |
| 10 | 3.12282 | 2.34388 | 0.13027 | 0.11304 |
| 15 | 2.99109 | 2.25033 | 0.13498 | 0.11202 |
| 20 | 2.86014 | 2.15627 | 0.13951 | 0.11132 |
| 25 | 2.72905 | 2.05800 | 0.14388 | 0.11091 |

This proves sustained optimization but also shows why pure curvature is not an acceptable final
objective: energy drifts. Increasing the learning rate to `3e-6` gave about the same curvature
gain in one step as ten `3e-7` steps. The equal-weight E/F/Q smoke 3140 then improved all three
strict metrics in one update: energy absolute error `0.12042 -> 0.11434 Ha`, force MAE
`0.11607 -> 0.10986 Ha/Bohr`, and relative Hessian-column error `3.43132 -> 3.23058`.
Energy-vs-Q gradient cosine was `-0.112`; force-vs-Q was `+0.074`, so clipping plus replay is
currently sufficient for the one-step direction test.

The E/Q graph now uses the stationary-energy envelope derivative, while F/HVP alone receives the
implicit density parameter VJP. This avoids amplifying a `1e-10` density residual by the
`1/h^2 = 1e6` energy-curvature stencil. Balanced implicit-response smoke 3142 completed in 3:59
and improved energy error `0.12042 -> 0.11279 Ha`, force MAE
`0.11607 -> 0.10946 Ha/Bohr`, and relative column error `3.43132 -> 3.20836`.

Job 3143 completed the 10-step balanced implicit follow-up in 14:43. Strict density was refreshed
before each parameter update and strict metrics were evaluated every two steps:

| Step | Relative column Frobenius | Energy abs. error (Ha) | Force MAE (Ha/Bohr) |
|---:|---:|---:|---:|
| 0 | 3.43132 | 0.12042 | 0.11607 |
| 2 | 3.06994 | 0.10519 | 0.10335 |
| 4 | 2.79099 | 0.09019 | 0.09399 |
| 6 | 2.49164 | 0.07296 | 0.08580 |
| 8 | 2.18810 | 0.05441 | 0.07774 |
| 10 | 1.88811 | 0.03452 | 0.07015 |

Job 3144 continued the scalar-curvature trajectory from step 10 through cumulative step 30. It
reached relative column error `0.72395`, energy error `0.01559 Ha`, and force MAE
`0.04063 Ha/Bohr`. The scalar contraction was nearly fitted, but the vector HVP remained much
farther from its reference. Balanced E/F/Q/H job 3145 reached relative column error `0.65760` at
step 20 with energy error `0.00881 Ha` and force MAE `0.05067 Ha/Bohr`.

The trainer and launcher now support two independently checkpointed AdamW states for alternating
replay and HVP updates. Job 3150 verified that path and reduced relative column error
`0.65760 -> 0.60645` through step 26, while energy error and force MAE ended at `0.02191 Ha` and
`0.04789 Ha/Bohr`. This confirms that optimizer-state conflict is real, but the implicit HVP solve
cost makes alternating updates unattractive before basic capacity is established.

Pure-HVP upper-bound job 3152 reset AdamW at the step-20 checkpoint and refreshed all three
densities before every update. Its one-column strict results were:

| Step | Relative column Frobenius | Energy abs. error (Ha) | Force MAE (Ha/Bohr) |
|---:|---:|---:|---:|
| 20 | 0.65760 | 0.00881 | 0.05067 |
| 21 | 0.60779 | 0.01313 | 0.05080 |
| 23 | 0.58841 | 0.00520 | 0.05049 |
| 27 | 0.51560 | 0.00049 | 0.05151 |
| 30 | 0.48034 | 0.00813 | 0.05262 |

All cached density gradients were about `1e-10`; the remaining error is not a density-convergence
artifact. Ten updates took `29:11`, with 9.0 GiB maximum host RSS and 1.34 GiB peak allocated GPU
memory. L1 continuation 3154 resumed the same AdamW state through step 50. Relative column error
reached `0.38114`, but energy error increased to `0.09479 Ha` and force MAE to
`0.05739 Ha/Bohr`. It is a capacity upper bound, not a promotable model, because replay terms are
disabled and the energy/force tradeoff has failed.

A Frobenius-aligned mixed absolute/relative RMSE HVP loss was added as an opt-in alternative to
the existing L1 loss. Ten tensor tests pass remotely. In the controlled one-step job 3153, L2
changed the strict relative column error `0.65760 -> 0.62556`, weaker than the corresponding L1
update near `0.6078`. A longer equal-start audit changed the conclusion: from the common step-30
checkpoint, L2 reached `0.40002` at step 40 versus L1 `0.47267` at the same cumulative step. L2
also retained energy error `0.00481 Ha` and force MAE `0.05227 Ha/Bohr`. Job 3157 then resumed the
L2 optimizer through step 60 and reached relative error `0.22581`. The cost was clear: energy error
rose to `0.05664 Ha` and force MAE to `0.06750 Ha/Bohr`. Pure L2 remains the capacity upper bound,
not a candidate. Job 3162 continued the same state from step 60 to step 80. Its best strict point
was `0.16595` at step 78; step 80 rebounded to `0.17086`, with energy error `0.02795 Ha` and force
MAE `0.08884 Ha/Bohr`. The one-column 5% ceiling is therefore not reached. Job 3165 preserves the
step-80 AdamW state but reduces the HVP learning rate from `3e-6` to `1e-6` for a bounded 20-step
oscillation-versus-capacity check; L1 is no longer continued.

The Gaussian distance layer had an unconditional `x.float()` cast even after the model was moved
to float64. This quantized scalar-energy inputs while autograd differentiated through the cast as
an identity. The layer now follows model-parameter dtype, with a regression test at `1e-9 Bohr`.
On the unchanged step-30 checkpoint, strict job 3156 left Hessian relative error effectively
unchanged (`0.480373 -> 0.480357`) but reduced energy-curvature versus force-curvature closure
error from about `1.67e-3` to `1.97e-6 Ha/Bohr2`, roughly 850-fold. This is a numerical
self-consistency repair, not a claimed model-accuracy gain. The combined GBF/capacity test set is
`21/21` passing remotely.

The same autograd pass now records gradient norms and squared-norm fractions for the distance
embedding, edge MLP, node embedding, GNN stack, and energy readout. Zero-learning-rate audit 3158
at L2 step 40 measured the following unweighted parameter-gradient norms:

| Loss | Gradient norm | Important cosine |
|---|---:|---|
| total energy | 1079.55 | energy vs HVP `+0.015` |
| total relaxed force | 275.40 | force vs HVP `+0.239` |
| scalar curvature Q | 524.47 | Q vs HVP `-0.795` |
| vector HVP | 23.15 | HVP vs force `+0.239` |

For HVP, squared gradient norm is distributed mainly over node embedding `49.2%`, energy readout
`28.1%`, Gaussian distance features `15.9%`, and GNN stack `6.5%`. Energy is instead dominated by
the energy readout (`89.5%`). This supports removing redundant Q from the balanced capacity run
and using independent replay/HVP optimizer states rather than a raw weighted sum. Job 3159 starts
from the L2 step-40 checkpoint and alternates E+F+Euler replay at `1e-6` with L2-HVP updates at
`3e-6`. It is matched to 20 HVP and 20 replay updates through cumulative step 80.

A two-update tolerance control restored the same L2 optimizer at step 40. Job 3160 reduced PCG
iterations from roughly `1680` to `426` by loosening the response tolerance from `3e-5` to `1e-3`,
but its strict step-42 relative error was `0.39445` versus `0.38053` for job 3157. Wall time was
`3:20`, only a modest end-to-end saving after density relaxation and integral work. The loose
response gradient is rejected for capacity fitting; `3e-5` remains the training tolerance.

Alternating updates now also support a sparse replay schedule. Job 3161 uses the same step-40
checkpoint and learning rates as 3159, but performs three HVP updates per replay update: 30 HVP and
10 replay updates through step 80. This directly tests whether less frequent replay can preserve
energy/force without cancelling curvature learning. The scheduling helper and diagnostics bring
the capacity unit-test count to `12/12`.

The completed strict trajectories separate the replay tradeoff:

| Run | Final step | Relative column Frobenius | Energy error (Ha) | Force MAE (Ha/Bohr) |
|---|---:|---:|---:|---:|
| 1:1 replay/HVP 3159 | 80 | 0.27558 | 0.00525 | 0.03852 |
| 3:1 HVP/replay 3161 | 80 | 0.22755 | 0.01251 | 0.05471 |
| joint gradient-scale match 3164 | 60 | 0.35059 | 0.00613 | 0.03520 |

Replay prevents the pure-HVP force regression. Sparse replay buys curvature accuracy but finishes
with a small force regression relative to the common step-40 source (`0.05227 Ha/Bohr`), while
equal replay and the joint run improve force more strongly. None approaches the 5% capacity gate.
The joint run uses `lambda_E/F/rho/H = 0.1/1/0.1/12` based on measured parameter-gradient norms.

A default-off two-task PCGrad implementation groups E/F/Euler replay against HVP/Q/spectrum,
preserves ordinary summed gradients when they align, and records conflict/cosine diagnostics.
It passes `14/14` complete-total tensor tests. Real one-step smoke 3166 measured replay-vs-HVP
cosine `+0.183`, so no projection was triggered; the strict relative error changed
`0.40001 -> 0.39695`. The current one-direction bottleneck is not a PCGrad-correctable gradient
conflict, and no long PCGrad run is justified.

The one-column residual is not a simple scale or rigid-motion defect. For pure-HVP step 80, the
predicted/reference HVP cosine is `0.98566`; optimal scalar rescaling still leaves `0.16179`
relative residual, and the predicted Cartesian component sums are around `1e-6`. The largest
errors are spread across several atom coordinates. The source network is already a 768-channel,
four-layer G3D Graphformer, so current evidence points first to optimization and multitask
constraints rather than an obviously undersized network. This does not yet rule out a functional
kernel limitation.

Pure-L2 job 3165 preserved the AdamW state from step 80, lowered the learning rate to `1e-6`, and
continued to step 100. It reached `0.14712` relative column error, energy error `0.02389 Ha`, and
force MAE `0.08617 Ha/Bohr`; the capacity gate remains failed and the energy/force tradeoff remains
unacceptable. Adam history-reset probes do not remove the plateau: `beta1/beta2=0/0.99` at
`3e-6` worsened one strict step to `0.33382`, while reset `1e-6` remained around `0.151` through
step 102. Job 3170 therefore preserves the original step-100 optimizer state and continues only
the pure-HVP capacity upper bound to cumulative step 200 at `1e-6`, with strict evaluation every
five steps and redundant zero-weight gradient diagnostics disabled.

`scripts/qm9_complete_total_capacity_analysis.py` provides the read-only Stage-1 aggregation
path. It exports run summaries, loss/Hessian trajectories, and an energy-force-Hessian tradeoff
plot while rejecting any completed run whose summary does not certify zero Test100 access. It now
also decomposes saved partial/full Hessian arrays into cosine, best scale, orthogonal residual,
maximum component error, and rigid-translation sums. The residual helper passes `1/1` test.
The current 18-run artifact is:
`/scratch/xzh/models/complete_total_capacity/20260717/analysis_one_column_v1`; it contains 119 raw
residual rows, CSV/JSON tables, and a PNG tradeoff trajectory. Its freeze certificate remains
`test100_accessed=false`, `test100_evaluations_used=0`.

## Step-200 Plateau and Angular-Kernel Capacity Audit

The preserved pure-L2 optimizer was continued to cumulative step 200. Its strict one-column
relative Frobenius reached `0.09219367`; energy absolute error and complete-total force MAE had
regressed to `0.05081835 Ha` and `0.08472779 Ha/Bohr`, so this remains a non-promotable capacity
upper bound. The residual cosine is `0.995906`, optimal scale is `1.00986`, and the residual after
optimal scaling is still `0.09167`. Rigid-translation sums are below `8e-7`; the remaining error is
directional rather than scale, rigid motion, density convergence, or scalar/force inconsistency.

An opt-in adjoint PCG warm start reuses the previous solution only for an identical geometry. In a
controlled five-step continuation from step 125, it reduced total PCG iterations from about
`2945--2989` per update to `653--788` after the cold first update and reduced wall time from
`25:27` to `21:45`. Strict metrics changed only at numerical-tolerance scale. Warm start is now the
default for these bounded capacity continuations, not a change to the formal evaluator.

Lower-learning-rate continuation does not remove the plateau. Job 3194 preserved the step-200
optimizer and ran 20 updates at `3e-7`; its final strict relative column error was `0.08866575` at
step 220, only a 3.8% relative improvement from step 200, while force and energy remained poor.
The matched `1e-7` job 3195 was still completing at this snapshot. These runs support stopping
percent-level tuning of the existing direct-HVP path.

Two train-only conservative geometry-kernel audits then isolated the missing accessible capacity:

| Scalar residual | Best column relative Frobenius | Conditioning | Interpretation |
|---|---:|---:|---|
| pair RBF, 30 configurations | `0.05374092` | about `9.06e15` | misses 5% and is numerically pathological |
| three-body RBF-Legendre | `0.01360579` | `551` | passes the 1%-5% single-column capacity target |

The best pair result used 48 radial centers and `sigma=0.75 Bohr`; its coefficient norm was
`1.03e8`. The three-body fit used four radial centers, angular orders 0--3, and 1824 invariant
features. It was constrained to zero anchor energy and force, with constraint residual below
`3e-15`, coefficient norm `0.493`, and a symmetric scalar-derived correction. This is evidence for
an angular/many-body functional-kernel bottleneck, not a promoted parent-general model.

The same three-body coefficients were then loaded through the actual complete-total trainer and
strict density-relaxed evaluator in job 3201. It reproduced column relative Frobenius
`0.0136057917`, versus the offline value `0.0136057922`; projected density gradients remained below
`1.3e-10`, anchor energy/force were unchanged, wall time was `6:43`, host RSS was 7.39 GiB, and
peak GPU memory was 0.71 GiB. The test and implementation path is fully Torch and second-order
safe; five focused pair/three-body tests pass remotely.

New files and artifacts are:

```text
mldft/ml/models/components/three_body_geometry_residual.py
scripts/qm9_complete_total_geometry_residual_capacity.py
scripts/qm9_complete_total_geometry_three_body_capacity.py
scripts/slurm_qm9_complete_total_geometry_residual_scan.sbatch
scripts/slurm_qm9_complete_total_geometry_three_body_capacity.sbatch
tests/test_qm9_complete_total_geometry_three_body_capacity.py
/scratch/xzh/models/complete_total_capacity/20260717/geometry_residual_pair_scan_step200
/scratch/xzh/models/complete_total_capacity/20260717/geometry_residual_three_body_step200_v2
/scratch/xzh/models/complete_total_capacity/20260717/one_parent_0028399_s1_lr0_h1_q0_threebody_job3201
```

Job 3202 completed the strict step-200 45x45 baseline in `32:03`. All 91 density points converged
below `1.0e-10`; cycles had median 218 and only three 10k-cycle long tails. The full matrix exposed
the direction-coverage failure hidden by column 0: full relative Frobenius was `1.28832`, with
MAE/RMSE `0.04281/0.11321 Ha/Bohr2`, even though the trained first column was at `0.09219`.
Numerical self-consistency was good (`asym/sym=2.44e-5`).

The full-matrix three-body fit jointly constrained the PBE anchor energy and force. It reduced
relative Frobenius to `0.05766`, energy error to about `1e-7 Ha`, and force MAE to
`3.10e-6 Ha/Bohr`, but missed the 5% gate. A six-configuration kernel scan showed an invariant
rank of 690 and essentially identical `0.05763` error. Increasing radial/angular resolution only
improved conditioning and coefficient norm; it did not enlarge the reachable curvature subspace.
The selected stable three-body kernel uses six centers, angular order 4 and `sigma=0.5 Bohr`, with
condition number `1.26e4` and coefficient norm `3.20`.

A parity-even four-body torsion scalar residual was then fit only to the remaining curvature while
constrained to zero incremental anchor energy/force. The initial kernel crossed the gate at
`0.04679`. A bounded six-configuration stability scan selected four radial centers, torsion order
5, `sigma=0.75 Bohr`, and bond scale 1.35. Its complete one-parent metrics are:

| Metric | Step-200 baseline | Three-body | Three-body + four-body |
|---|---:|---:|---:|
| Hessian MAE (Ha/Bohr2) | 0.042814 | 0.002496 | 0.002231 |
| Hessian RMSE (Ha/Bohr2) | 0.113215 | 0.005065 | 0.004112 |
| relative Frobenius | 1.288317 | 0.057631 | **0.046797** |
| asym/sym | 2.44e-5 | 3.44e-5 | 4.58e-5 |

The selected torsion increment has coefficient norm `24.46`, design condition number `8.80e5`,
anchor energy/force residual below `2e-8`, and exact-autograd symmetry error below `9e-15`.
Its force-FD symmetry maximum is `8.04e-5 Ha/Bohr2` at the formal `h=1e-3 Bohr`; combined
`asym/sym` remains two orders of magnitude below the 0.5% gate. The composed Hessian is formally
equivalent to re-evaluating all densities because both residuals are density independent; job
3201 verified that algebraic composition against the strict evaluator on column 0 to about
`5e-10` relative precision. Redundant full-density jobs 3203/3205 were canceled after that proof.

Additional files/artifacts are:

```text
scripts/qm9_complete_total_geometry_four_body_capacity.py
scripts/slurm_qm9_complete_total_geometry_four_body_capacity.sbatch
scripts/launch_qm9_complete_total_three_body_full_scan.sh
scripts/launch_qm9_complete_total_four_body_full_scan.sh
tests/test_qm9_complete_total_geometry_four_body_capacity.py
/scratch/xzh/models/complete_total_capacity/20260717/geometry_residual_three_body_full_kernel_scan
/scratch/xzh/models/complete_total_capacity/20260717/geometry_residual_four_body_full_kernel_scan
```

The one-parent full-Hessian sub-gate is passed. Jobs 3221--3224 now produce matched strict full
baselines for the other four frozen Stage-1 train parents. No validation parent or Test100 record
has been used.

## Shared Five-Parent Capacity And Stability Audit

Jobs 3221--3224 completed the remaining original strict baselines. All displaced densities reached
about `1e-10` projected gradient. Baseline relative Frobenius values were `1.2883`, `6.5955`,
`2.1620`, `1.9738`, and `3.2132`; the large errors are therefore not density residuals.

The first shared linear three-/four-body fit (job 3232) failed because hard E/F constraints and an
unpreconditioned 1000-iteration LSQR produced coefficients with poor curvature. Cached-design
re-solves then separated numerical solve from representation capacity:

| Shared scalar solve | Median rel Fro | Max rel Fro | E/F status |
|---|---:|---:|---|
| Hessian-only, 5k LSQR | 9.10% | 15.94% | unacceptable drift |
| Hessian-only, warm +15k | 5.69% | 13.57% | unacceptable drift |
| soft E/F, scales 0.10/0.05 | 12.32% | 18.78% | E about 1e-5 Ha; F about 6e-4 Ha/Bohr |

The warm Hessian-only three-body coefficient norm was `3.55e11`, because near-zero derivative
columns were amplified. The linear result is not a stable capacity proof even where aggregate
error is small.

A shared float64 Softplus descriptor residual was added. Its analytic formula includes both the
linear descriptor-Hessian term and the nonlinear `J^T H_feature-map J` term. A focused test verifies
energy, force, and full Hessian against PyTorch autograd. On the original five parents, h128 with
`lambda_H=10` reached best per-parent relative Frobenius values `0.71%`, `1.71%`, `0.99%`, `5.08%`,
and `2.10%`; energy errors were below `6.1e-6 Ha`. This is a strong capacity result but not a gate
pass because one parent remained just above 5% and parent `0050129` violated numerical symmetry.

The stability audit changed the parent-screen definition. Normalizing baseline curl by the
inaccurate predicted symmetric Hessian can hide a large absolute nonconservative component. The
pre-fit gate is now

```text
||H_asym,baseline||F / ||H_PBE||F < 0.005.
```

The four retained original parents are between `7.8e-6` and `3.4e-5`. `0050129` is `0.03678` and
replacement `0033410` is `0.03868`; both are excluded independently of fitted-model error. The
`0033410` capacity run confirmed the consequence: h128 ended at median/max `17.58%/26.79%`, while
its final asym/sym was `3.88%`. Final replacement `0121249` passes decisively: baseline
`||H_asym||F/||H_PBE||F=8.17e-6`, relative Frobenius `2.9614`, and projected density gradient
`1.63e-9`.

The stable5-v2 dependency chain 3271--3275 completed. The final shared h128 model passes both
programmatic gates with per-parent results:

| Parent | Hessian relative Fro | Energy abs. error (Ha) | Force MAE (Ha/Bohr) | asym/sym |
|---|---:|---:|---:|---:|
| 0028399 | 0.002052 | 1.36e-5 | 1.10e-5 | 3.42e-5 |
| 0031108 | 0.003330 | 1.85e-5 | 1.32e-5 | 1.79e-5 |
| 0132419 | 0.048886 | 1.74e-5 | 2.53e-3 | 7.82e-6 |
| 0031012 | 0.010467 | 1.68e-5 | 1.04e-4 | 2.16e-5 |
| 0121249 | 0.011177 | 1.36e-5 | 4.41e-5 | 8.17e-6 |

Median/max relative Frobenius are `0.010467/0.048886`. The train-parent vibrational capacity
diagnostic gives frequency MAE/RMSE `17.47/46.86 cm-1`, mean mode overlap `0.896`, and 10 versus
11 PBE imaginary modes. These are fitted-parent upper-bound metrics only.

Stage 2 is frozen before model fitting. Its manifest contains 31 ordered train-only candidates and
24 or fewer orthonormal internal directions per parent, stratified into train and held-out bond,
angle, torsion, low-mode, and random directions. Manifest SHA256 is
`8184ae0713f915c69d6614c3e84ce8606d4ea72be0ef6c98f544f66e5f25f403`; Test100 access remains
zero. Strict baseline array 3297 and clean local-checkpoint rescues 3319/3327 completed. The first
attempt 3282 was canceled after detecting a missing protocol compatibility field; no Hessian rows
from it are used. Failed end-of-run shared-filesystem writes from 3297 are also excluded because
formal selection requires a completed `summary.json`.

Selection v2 takes the first 20 curl-stable candidates. Their untouched baseline full-Hessian
relative Frobenius has median `2.695`, mean `2.608`, P90 `3.133`, and range `1.288--3.568`, matching
the approximately 2.7 starting point in the objective. Every selected parent has exactly 17 train
and 7 held-out directions; maximum orthonormality and external-mode residuals are `6.63e-15` and
`1.62e-15`. Ten numerical exclusions are recorded in `selection_v2/selection_audit.csv`; their
density gradients remain about `1e-10`, while PBE-normalized curl ranges from `1.15e-2` to more
than `1e3`. `0080472` passes numerically but remains an unused reserve because the first-20 rule
was already satisfied.

Job 3336 completed the shared descriptor derivative matrices in
`stage2_direction_v1/shared_descriptor_design_v2` in `2:31:28` with `7.22 GiB` maximum RSS. It did
not solve coefficients against train, held-out, or full Hessian targets. Job 3337 passed the
20-parent 10-step smoke. Job 3338 then completed the zero-initialized 5000-step lambda-H=1 PCGrad
warm-up in `633.9 s` with `11.63 GiB` maximum RSS. The loss includes E/F anchors, mixed
absolute/relative dynamically sampled train-HVP error, and a train-side low-mode wrong-curvature
term; held-out directions and full Hessians are evaluation only.

The untouched baseline and warm-up metrics are:

| Run | Median train HVP rel Fro | Median held-out HVP rel Fro | Median full Hessian rel Fro | Max full Hessian rel Fro |
|---|---:|---:|---:|---:|
| strict complete-total baseline | 2.591 | 2.615 | 2.695 | 3.568 |
| h128 lambda-H=1, step 5000 | 0.279 | 0.571 | 0.403 | 0.564 |

The warm-up is a real improvement but passes none of the Stage-2 train-direction, held-out
direction, or full-Hessian gates. End-of-warm weighted HVP gradients are much smaller than the
energy gradient, and PCGrad conflicts occur only intermittently. Job 3339 therefore continues the
same scalar checkpoint with lambda-H=10, learning rate `3e-5`, and 30,000 steps. E/F replay,
dynamic direction sampling, the spectrum term, gradient clipping, and PCGrad remain enabled. The
output directory is `stage2_direction_v1/mlp_h128_h10_v2`; Test100 access remains zero.

The v1 direction-identifiability audit finds median train-direction coverage of only `37.8%` of
the internal-coordinate dimension. Even with exact HVPs on those train directions, the PBE
`Q H Q` block left unidentified has median relative Frobenius `68.6%`. The warm model's `40.3%`
full error is better than this minimum-norm completion through its shared functional prior, but
the audit shows that the original 17-direction split is a severe information bottleneck. It does
not explain the per-parent difficulty ordering: Pearson correlations are only `0.10` between
unidentified `Q H Q` and warm full error, and `-0.08` between minimum-norm oracle and actual
held-out error. Model representation must therefore be diagnosed independently of direction
coverage.

Dense-direction v2 was preregistered before fitting with seed `20260721`, at most 48 directions,
20% immutable held-out directions, and up to eight bond, angle, torsion, and low-mode directions
per parent. Manifest SHA256 is
`dfd838b375ae1165708c66cf25a25beb51a3c0c56170e44e416adf1b566134ee`. On the frozen selected 20,
train counts are 30--37 and held-out counts are 9--11. Median train coverage rises to `76.9%`,
while median/P90/max unidentified `Q H Q` relative Frobenius falls to
`41.6%/52.3%/59.3%`. Maximum orthogonality and external-mode residuals are `2.8e-11` and
`1.3e-11`. Job 3340 trains v2 with eight dynamically sampled train directions per parent from the
v1 warm checkpoint; the checkpoint has never consumed v2 held-out directions. v1 job 3339 remains
the sparse-direction control. Both execute only on node01, and Test100 remains untouched.

Sparse-v1 job 3339 is now complete and formally fails:

| Metric | warm lambda-H=1 | sparse-v1 lambda-H=10, 30k |
|---|---:|---:|
| median train HVP rel Fro | 0.2793 | 0.0223 |
| median held-out HVP rel Fro | 0.5713 | 0.3928 |
| median full Hessian rel Fro | 0.4035 | 0.2492 |
| P90 full Hessian rel Fro | 0.5387 | 0.4416 |
| fraction full Hessian <= 0.15 | 0% | 10% |
| median energy abs. error (Ha) | 1.28e-2 | 5.16e-4 |
| median force MAE (Ha/Bohr) | 2.06e-3 | 4.59e-4 |

Train-direction fitting and E/F replay both succeed, but unseen directions remain far outside the
gate. Per-kind held-out medians remain `0.323` angle, `0.345` bond, `0.420` low-mode, `0.248`
random-internal, and `0.335` torsion. This rules out a single bad direction class.

An isolated full-Hessian capacity oracle, job 3341, uses the same 20 parents and h128 descriptor
but may never be promoted or used to initialize a direction run. By step 1400 it already reaches
median/max full error `0.142/0.237`, demonstrating that partial-direction supervision is exposing
far less capacity than a complete training target. At step 30,000 it formally passes all 20
parents with median/P90/max `0.0162/0.0399/0.0495`. Median energy absolute error is
`2.03e-5 Ha`, and median force MAE is `1.30e-4 Ha/Bohr`. Capacity-only vibrational metrics are
frequency MAE/RMSE `20.48/44.16 cm-1`, mean mode overlap `0.860`, and 42 versus 33 PBE imaginary
modes. The extra nine imaginary modes show that even a <=5% matrix fit does not automatically
pass low-curvature sign fidelity.

Near-full v3 is the final direction-coverage control. Two invalid attempts are preserved and
excluded: single-pass Gram-Schmidt reached `3e-7` orthogonality error near a complete basis, and
the first double-reorthogonalized bank allowed structured directions to fill some small spaces
without random-internal directions; job 3342 was stopped after about five minutes for this reason.
The generator now uses double reorthogonalization and reserves at least 20% random-internal
directions. The accepted manifest SHA256 is
`daf5c2d34a7e83c530852c5989b07aef33fa99943828f5d2718c1627f1cddc2b`. Every selected parent has
8--12 random directions; median train coverage is `89.7%`, and median/P90/max unidentified
`Q H Q` is `24.4%/30.4%/30.4%`. Max orthogonality/external residuals are
`7.8e-16/1.7e-16`. Job 3343 is the accepted run and trains from the v1 warm checkpoint, never from
the full-Hessian oracle.

Dense-v2 job 3340 completed at best step 29,300. It passes the median full-Hessian gate but not
the full Stage-2 gate:

| Metric | sparse-v1 | dense-v2 |
|---|---:|---:|
| median train HVP rel Fro | 0.0223 | 0.0303 |
| median held-out HVP rel Fro | 0.3928 | 0.3206 |
| median full Hessian rel Fro | 0.2492 | 0.1473 |
| P80 / P90 full rel Fro | 0.3686 / 0.4416 | 0.2373 / 0.3014 |
| fraction full <= 0.10 / 0.15 / 0.20 | 0% / 10% / 40% | 15% / 50% / 70% |
| frequency MAE / RMSE (cm-1) | 213.69 / 457.53 | 72.91 / 188.17 |
| mean mode overlap | 0.605 | 0.747 |
| model / PBE imaginary modes | 111 / 33 | 59 / 33 |

Coverage improves every independent curvature and vibrational metric, but dense-v2 remains
non-promotable. Accepted v3 job 3343 completed at step 30,000:

| v3 metric | Result | Formal target |
|---|---:|---:|
| median full Hessian rel Fro | 0.0505 | <=0.10 |
| P80 / P90 / max full rel Fro | 0.1086 / 0.1130 / 0.1883 | P90 <=0.20 |
| fraction full <=0.15 | 95% | >=80% |
| median train / held-out HVP rel Fro | 0.0225 / 0.1505 | about 0.05 / 0.10--0.15 |
| fraction train <=0.05 / held-out <=0.15 | 80% / 50% | strict gate requires 100% / 100% |
| frequency MAE / RMSE (cm-1) | 31.73 / 67.12 | about 100--200 MAE |
| mean mode overlap | 0.819 | improve |
| model / PBE imaginary modes | 46 / 33 | near PBE |
| median energy abs. / force MAE | 3.33e-4 Ha / 4.18e-4 Ha/Bohr | no significant regression |

The formal full-Hessian distribution target is passed, but the preregistered all-parent unseen
direction gate is not. The five leading held-out hard cases are `0038511`, `0064547`, `0132608`,
`0031108`, and `0031012`; their errors include real bond/angle absolute HVP discrepancies and are
not explained solely by small reference norms. No train100 or Test100 work is allowed yet.

Two train-only diagnostics are running. Job 3344 compares lambda-spectrum `1.0` against the main
`0.1` under the same frozen directions, initialization, seed, and step count. Job 3345 continues
the main best checkpoint for 10k steps at LR `1e-5` with HVP objective equal to the parent mean
plus the top-20% train-parent mean. The optional tail weight defaults to zero in code, preserving
all historical runs. Held-out directions are excluded from loss, weighting, checkpoint selection,
and stopping.

Both diagnostics completed without promotion. Spectrum weight 1.0 gives full-Hessian
median/P90/max `0.05045/0.11275/0.18886`, while median energy error increases to
`9.72e-4 Ha`. The tail continuation gives train/held/full medians
`0.02109/0.15715/0.04743` and full P90/max `0.11123/0.18018`; its original held-out median is
worse than the accepted main run. The diagnostic improves frequency MAE to `29.85 cm-1` and mode
overlap to `0.826`, but it cannot override the frozen held-out and E/F rules.

An independent post-fit confirmation was then frozen with `may_not_return_to_training=true`.
After preserving one rejected exact-duplicate preparation, the accepted bank contains 480 new
directions (24 per parent, eight random per parent), no exact training-bank duplicates, maximum
training-bank overlap `0.9709`, and external-mode residual `1.39e-16`. Its manifest is
`stage2_direction_confirmation_v1/confirmation_manifest.json`, SHA256
`ce2e9d026a12c73f1bb1d14986017b89a305cdc781082bac4ceddf38b5ef909d`.

| Confirmation metric | Main v3 | Formal distribution target |
|---|---:|---:|
| parent median relative RMS | 0.05169 | <=0.10 |
| parent P80 / P90 / max | 0.11866 / 0.11971 / 0.18622 | P90 <=0.20 |
| fraction <=0.10 / 0.15 / 0.20 | 75% / 95% / 100% | at least 80% <=0.15 |
| component HVP MAE / RMSE (Ha/Bohr2) | 0.002552 / 0.005296 | report |

Per-kind median/P90 errors are angle `0.040/0.145`, bond `0.023/0.059`, low mode
`0.062/0.160`, random `0.044/0.118`, and torsion `0.057/0.211`. This bank is independent
confirmation only and cannot influence weights, checkpoint choice, or a return to Stage-2 tuning.

The numerical Stage-2 full-Hessian distribution gate and the independent confirmation distribution
gate pass on the recorded artifacts. The original all-parent direction gate remains failed: only
50% of parents satisfy the original held-out <=15% rule. A later provenance audit, documented
below, revoked authorization to advance these artifacts as one physical model.

Stage-3 readiness audit finds that the fixed train100 inventory has 100 parents and 400 audited
directions, but only 31 parents and 202 directions are v3-stable; seven parents pass all four v2
directions. The Stage-2 complete-total baseline curl gate leaves 21 usable train parents (the
selected 20 plus reserve `0080472`). The independent validation20 audit has seven v3-stable
parents. All 100 train and 20 validation PBE analytic Hessians exist, as do train800 E/F labels.
Historical direct-HVP sidecars target PBE Hessians from a fixed-density learned-energy path and
cannot serve as complete-total relaxed model-HVP labels without an explicit baseline correction.
No Test100 data or metric has been read.

### Uniform-baseline provenance correction

The Stage-2 selected baseline manifest is heterogeneous. Seven parents (`0016298`, `0028399`,
`0064547`, `0031108`, `0132419`, `0031012`, `0121249`) use
`one_parent_0028399_s50_lr1e-6_h1_q0_job3175/checkpoints/step_0000200.ckpt`, which was
capacity-fitted on one parent. The remaining 13 use the untouched original A checkpoint
`qm9_hvp_curvature_v1_A_w0ep00_f1em02_seed314159_s1200/checkpoints/last.ckpt` (two temporary
checkpoint paths are byte copies made by the array job). A residual added to these different base
models does not define one shared scalar total energy across molecules.

Consequences:

1. The one-parent/stable5 and isolated full-Hessian fits still establish representation capacity.
2. The v1/v2/v3 coverage ladder and 480-direction confirmation remain valuable diagnostics of
   direction coverage and the smooth geometry kernel.
3. Their formal generalization pass is revoked; no v3 checkpoint may seed Stage 3 until the same
   result is reproduced with one frozen baseline checkpoint.
4. Stage-3 `assets_v1` is a rejected incomplete merge. `assets_v2` correctly lists 3200 replay
   labels, 21 curl-eligible train parents and seven validation candidates, but its HVP eligibility
   is mixed-baseline and is not authoritative for training.
5. Replay job 3370 was stopped and preserved after the mismatch was found. Uniform original-A
   replay smoke 3380 completed `15/16` strict points in `6:32`; successful points have median
   wall time `22.99 s`, median `766` cycles, tensor density-gradient median `1.57e-10`, and maximum
   tensor-vs-legacy energy difference `6.82e-13 Ha`.
6. `0000023.0000000` is the only smoke failure. Its optimizer metadata gradient is `1.02e-10`,
   but the differentiable complete-total rebuild gives gradient `0.478` and energy difference
   `0.020276 Ha`. This is classified as a tensor/legacy stationarity mismatch rather than a
   density-cycle failure; it is not admitted to replay training.
   Focused jobs 3394/3395 show that the ordinary fixed-geometry rebuild also disagrees with the
   optimizer (`0.31--0.43` projected norm), while coefficient and transform round trips are
   accurate to `1.64e-14` and matrix identity errors are below `6.4e-14`. Rebuild-to-rebuild
   energies vary, consistent with a non-unique local-frame/graph branch at this symmetric
   geometry. It remains masked pending deterministic-frame work.
7. Slurm split job 3379 initially launched only task 0 because commas in `--export` were parsed as
   separate entries. The remaining six scientific tasks were not attempted. They were resubmitted
   as job 3387 with colon-delimited IDs; 3379_0 plus 3387_0--5 now recompute all seven full
   baselines from the uniform original A checkpoint. Test100 remains at zero access.
8. Uniform baseline `0016298` is complete: original-A relative Frobenius is `2.3370`, compared
   with `2.2165` in the mixed manifest, and its PBE-normalized curl ratio is `6.21e-5`. It passes
   the numerical curl gate; the other six recomputations are still running.
9. The replay evaluator now exposes an opt-in fixed-geometry rebuild audit; replay and merge tests
   pass `4/4`. Uniform original-A train800 replay array 3396 reads the frozen 3200-label CSV and
   begins with two concurrent node01 shards while the remaining Hessian baselines run. It has no
   Test100 input.
10. All seven recomputed baselines and all 28 complete array baselines pass a tensor-value
    provenance audit against frozen original A (`7/7` and `28/28`). Float32-to-float64 promotion
    is recorded separately and every promoted value is exact. Source checkpoint SHA256 is
    `e6516b04917a9dfaae3c4d960b10d77c768f3e5e784f95290ff41c1202f9d9bc`; canonical state hash is
    `5eae47caa18ba8a6e7e96c13e0e1c4418ba4106bf4d4b5385d09e75b9d880653`.
11. The authoritative uniform selection manifest SHA256 is
    `e81e00ecbf99e7d14751c079aeb081d06455ca720bbb3ca5c758e39de359f127`. Its 20 IDs and order are
    identical to old v2; 21 of 31 candidates pass numerical gates. Uniform baseline relative-Fro
    median/mean/P90/max are `2.7331/2.7910/3.1528/3.5683`, versus old mixed medians `2.6953`.
12. Job 3402 reruns the 5000-step v1 warm-up from zero with the uniform manifest. Dependent job
    3403 runs accepted near-full v3 for 30,000 steps from the uniform warm checkpoint. The frozen
    9.1-GiB descriptor design is reused only because parent IDs/order are unchanged and its arrays
    are baseline-independent scalar descriptor derivatives; new summaries capture its provenance.
13. Uniform warm-up job 3402 completed. Best step 4300 has train/held/full medians
    `0.2328/0.5188/0.3717` and full max `0.7828`. Uniform near-full v3 job 3403 then completed
    30,000 steps in `42:43`; the frozen best checkpoint is step 30,000, SHA256
    `bc0cb2e63aad323ab011a92d806ce85f44475325b79aebabaab19c989026f6ea`. Final train/held-out
    HVP medians are `0.02340/0.15668`, while full-Hessian median/mean/P80/P90/max are
    `0.05431/0.06444/0.08151/0.12125/0.16402`. Fractions at or below `10%/15%/20%` are
    `85%/90%/100%`. Median energy absolute error and force MAE are
    `4.55e-4 Ha/4.02e-4 Ha/Bohr`; maximum asymmetry-to-symmetry ratio is `5.60e-4`.
14. Frozen post-fit job 3408 completed. Its independent 480-direction confirmation passes the
    distribution gate with parent median/P80/P90/max
    `0.05247/0.09823/0.12447/0.16264` and fractions `80%/95%/100%` at or below
    `10%/15%/20%`; mean component HVP MAE/RMSE are
    `0.00248/0.00515 Ha/Bohr2`. Frequency MAE/RMSE are `29.85/63.13 cm-1`, mean mode overlap is
    `0.821`, and model/PBE imaginary-mode totals are `41/33`. The preregistered strict all-parent
    direction gate still fails: train/held-out pass fractions are only `90%/45%`, with held-out
    maximum `0.7215`. The decision is therefore two-part: full-Hessian and independent-direction
    distributions pass, but the per-parent direction-tail failure remains a mandatory Stage-3 risk.
15. Stage-3 assets are frozen in `stage3_unseen_parent_v1/assets_v3_uniform_A`. The asset and
    independent-validation capacity manifest hashes are respectively
    `4eb176bde25d06c4ad1480bbe2b19752ff7cca23d0a3a768b36d3ed55ad3c703` and
    `95c9c2a7585ed3508bc27ee9b749a8201868050aebfaf5a50bdfcfa4baff911f`. The latter contains the
    seven validation-only IDs `0000751`, `0023317`, `0027926`, `0039447`, `0056113`, `0131469`,
    and `0132890`, with label/PBE-Hessian hashes and zero Test100 access.
16. The frozen active-feature schema is complete in
    `stage3_unseen_parent_v1/active_feature_schema_uniform_v3`. It binds the step-30,000
    checkpoint to `20,328` active float64 scalar descriptor columns: `7,992/10,095` three-body
    and `12,336/12,336` four-body. Its manifest records all checkpoint, design, feature-key, and
    schema hashes, and records zero Test100 access. The exact-collinear four-body torsion path was
    made second-order safe without changing the nondegenerate formula; combined second-derivative
    tests pass `5/5`.
17. Validation original-A full-Hessian array 3415 and dependent provenance/strict merge job 3420
    are active. Replay array 3396 is resumable by task hash; it had 683 summaries at `06:03` and
    is temporarily throttled while validation uses node01. Job 3416 restores replay concurrency
    automatically after job 3415. No running task was canceled.
18. The Stage-3 replay descriptor cache and independent-parent external evaluator are implemented
    and tested but not yet used for fitting or candidate selection. The cache stores float64
    descriptor/Jacobian tensors and scalar-baseline residual E/F targets under task hashes; it
    must first pass a one-task smoke after replay merge job 3422 and schema completion. The
    external evaluator maps only frozen active keys and reports missing coverage. Neither code
    path can read Test100.
19. The seven validation original-A baselines are now complete and pass the strict merge. The
    authoritative manifest is
    `stage3_unseen_parent_v1/validation_baselines_v1_uniform_A/validation_baseline_manifest.json`,
    SHA256 `b1937e31c13e9906b94e8bdf1516e988adad974fc70dc34eb5292eeae108772d`.
    All `7/7` parents have final density-gradient norm below `1e-8`; the observed maximum is
    `1.4424e-10`. All pass the PBE-normalized curl gate, with maximum `8.526e-4`. Their untouched
    original-A Hessian relative-Frobenius errors span `2.475--3.101`, with median `2.7031`.
20. Frozen checkpoint job 3437 then evaluated the Stage-2 step-30,000 model on those seven unseen
    parents. It fails catastrophically: Hessian relative-Frobenius median/mean/P90/max are
    `303.611/11642/32703/75040`, with `0/7` parents below `20%`. Median energy absolute error is
    `32.215 Ha` and median force MAE is `13.2666 Ha/Bohr`. Two easier parents have Hessian errors
    `1.0603` and `1.3547`, while `0132890` reaches `75040.42`. This is an unseen-parent
    extrapolation failure, not a numerical convergence or nonconservativity failure.
21. Frequency postprocessing reinforces the failure: mean frequency MAE/RMSE are
    `45278.77/71787.69 cm-1`, mean mode overlap is `0.5788`, and model/PBE imaginary-mode totals
    are `117/12`. These metrics are diagnostic validation metrics, not Test100 results.
22. Component diagnostics identify the main mechanism. Unseen-parent descriptor L2 norms range
    from ordinary values near `7.5--10.3` to `1.72e3`, `5.66e3`, `4.45e4`, and `1.03e6` on the
    catastrophic parents. The maximum descriptor component reaches `5.41e5`. A linear-only
    residual still has median relative Frobenius `300.28` and maximum `14119.1`; changing only to
    `E=N f(x/N)` gives median `321.68` and maximum `75040.19`. Therefore the dominant defect is
    the active feature scaling by tiny column norms estimated from only 20 parents. Missing active
    feature coverage (`77--100%`, depending on feature family and parent) compounds the problem,
    but Softplus nonlinearity and atom count alone do not explain it. Maximum numerical
    asymmetry/symmetry remains only `4.34e-5`.
23. Two Stage-1 repair audits are active on node01 and access no validation targets during fitting:
    job 3448 tests atom-extensive scalar energy with the old column-norm scale; job 3449 tests the
    same architecture with unit physical feature scale and zero residual initialization. Each is
    a 30,000-step stable5 capacity run. A repair is not allowed to advance unless it first restores
    the original Stage-1 `<=5%` full-Hessian ceiling and preserves energy/force anchors.
24. Train800 original-A replay generation remains resumable under job array 3396. At 08:14 it had
    `1299/3200` geometry summaries. Nested BLAS/integral workers made seven simultaneous shards
    slower than two, so the array throttle is now two. The known symmetric-frame point
    `0000023.0000000` remains explicitly excluded. The old active-schema descriptor cache and
    replay fitting path are not launched because that schema contains the diagnosed pathological
    column scaling.
25. The column-normalization audit quantifies a roughly ten-order-of-magnitude range. Active
    three-body source norms range from `9.49e-7` to `93.48`; four-body norms range from
    `1.61e-3` to `1.41e4`. Pure unit scaling removes inverse amplification but also removes useful
    preconditioning of large columns. It finishes stable5 at median/max
    `8.91%/14.89%`. Atom-extensive `E=N f(x/N)` with the original column scale finishes at
    `5.42%/10.92%`. Neither passes the Stage-1 ceiling.
26. A `floored_column_norm` mode is now implemented and tested. It divides each feature by
    `max(source_column_norm, floor)`, preserving large-column preconditioning while bounding the
    amplification of unseen features. Extensive stable5 jobs 3452--3457 scan fixed floors
    `0.01/0.1/1.0`; matched non-extensive jobs 3458--3463 start only after those jobs finish.
    The scale rule is preregistered before either scan completes: among candidates that pass all
    five parents at relative Frobenius `<=5%` without material E/F regression, choose the largest
    floor. If none passes, do not inspect the seven validation targets to choose a scale; instead
    increase conservative scalar capacity or implement a local additive energy architecture.
    These jobs use stable5 train parents only and do not access validation parents or Test100.
27. Before the matched non-extensive scan completes, the qualitative E/F clause is frozen as
    numerical stable5 gates: energy absolute-error median/all-parent maximum
    `<=1e-3/2e-3 Ha`, and force-MAE median/all-parent maximum
    `<=1e-3/3e-3 Ha/Bohr`. The three extensive floored runs all fail the Hessian gate. Floors
    `0.01/0.1/1.0` finish at median/max relative Frobenius
    `4.18%/8.70%`, `4.50%/9.57%`, and `5.19%/10.35%`, respectively. Their E/F errors also do not
    satisfy all newly explicit anchors. They remain architecture diagnostics and cannot initialize
    Stage 2.
28. The matched non-extensive scan passes all preregistered gates at every tested floor. The
    stable5-only selector chooses the largest floor, `1.0`. Its Hessian relative-Frobenius
    median/max are `0.011853/0.049251`; energy absolute-error median/max are
    `6.34e-5/8.75e-5 Ha`; force-MAE median/max are
    `8.70e-5/2.534e-3 Ha/Bohr`. The selector manifest is
    `stage1_floored_stable5_v1/selection_v1/selection_manifest.json`, SHA256
    `87b27c0e11aefaf80374b349176281fc0edf1e95f0798d11b97187e8b21aa972`. It certifies that no
    validation-parent metric and no Test100 record participated in selection.
29. A Stage2-design-complete provisional schema is frozen at
    `stage3_unseen_parent_v1/active_feature_schema_full_floor1_v1`. It uses cutoff zero and
    floor `1.0`, retaining `10,095/10,095` three-body and `12,336/12,336` four-body keys for
    `22,431` features total. The schema manifest SHA256 is
    `10ab6938afe2cec659aaa0caf06eb0094e1a4e29d2d81bffaf887c0b9437e9c9`; its dimension-only,
    zero-residual step-0 seed checkpoint SHA256 is
    `3d4e5c9f902730914f6e40ab2ef2feb8a7559342ee766f304ffa279f6e9fcb76`. The seed is not a
    trained candidate and fails the scientific gate by design. A later audit showed that
    "all Stage2-design keys" is not the same as all keys observed in train800 chemistry; this
    schema is retained for diagnosis but is not replay-authoritative.
30. The frozen external-failure plot and correlation table are in
    `validation_external_eval_stage2_s30000_uniform_A_diagnostic_v2/scaling_analysis_v1`.
    Correlations of log Hessian error with log descriptor L2, descriptor max, and correction
    Hessian norm are respectively Pearson `0.956/0.962/0.995` and Spearman
    `0.929/0.964/0.964`. This postmortem is not an input to robust-floor selection.
31. Pending cache jobs 3470/3471 were canceled before execution after the remaining key-coverage
    gap was identified. No descriptor cache or model was produced. A train800-only inventory is
    now built from the hash-frozen 3200-label CSV and will be unioned with all Stage2 design keys.
    Inventory v1 is explicitly rejected because it used mismatched four-body defaults
    (`sigma=0.4`, order `3`, bond scale `1.25`). The authoritative descriptor definition is
    `sigma=0.75`, torsion order `5`, bond scale `1.35`, matching both the Stage2 design and replay
    cache evaluator. Corrected inventory v2 is diagnostic; union-preserving v3 is the candidate
    replay schema source. Neither reads validation or Test100.
32. The authoritative union-preserving v3 inventory is complete at
    `stage3_unseen_parent_v1/train800_feature_inventory_floor1_v3`. It contains `32,121` keys:
    `11,625` three-body and `20,496` four-body. Relative to the Stage2 design, train800 adds
    `1,530` and `8,160` keys respectively, while no Stage2 key is lost. Its manifest SHA256 is
    `42ab2a5393ee55598d562d299f958297faf2b582b38d2e6f1758caa2fe02ec6a`.
33. Jobs 3477/3478 created a matching dimension-only zero-residual checkpoint and bound the
    inventory to it. The resulting dimensional schema is
    `stage3_unseen_parent_v1/active_feature_schema_train800_union_floor1_v1`; manifest SHA256 is
    `5f0c58413d0437e4613b7526efda1f6ea0ad80ff668f974a25f1611ad715093b`, and checkpoint SHA256
    is `256a97f92be6edfd3e580504464747c7a0b35b9a803a712cdd5769246ef8ad81`. It records floor `1.0`
    and zero Test100 access. This zero seed is dimensional/provenance state only and was superseded
    before cache execution because it does not preserve the passed stable5 scalar.
34. Cache jobs 3479/3480 were canceled while still dependency-pending; no cache task ran. Runtime
    descriptor settings must now exactly match the schema manifest, so a mismatch such as the
    rejected v1 four-body settings fails before evaluation.
35. Jobs 3484--3486 freeze the `17,765`-feature stable5 source schema, embed its selected floor-1
    checkpoint into the `32,121`-feature train800 union by exact descriptor key, rescale weights
    by target/source feature scale, zero all new columns, and bind the result as authoritative v2.
    The expanded checkpoint SHA256 is
    `4ab80b076968d7b1f34ac83df61809de1000574f90dafadc1574e8322867f76b`; v2 schema manifest
    SHA256 is `79d05c1351fb48532aa7569af3123b115e69c631d73dc00ae667bfa4c9870e67`.
36. Job 3487 verifies physical initialization equivalence end to end on stable5. At 32,121
    features, Hessian median/max remain exactly `0.0118532/0.0492510`; all E/F/Hessian gates pass.
    Thus union expansion preserves the passed scalar rather than restarting from zero. Focused
    node01 static/unit checks now pass `17/17`.
37. Authoritative cache jobs 3489/3490 use v2 and wait for replay merge 3422. Job 3491 then runs
    fail-closed preflight and a 10-step smoke only after the cache merge succeeds. Preflight checks
    schema/inventory/checkpoint hashes, exact 32,121-dimensional state, 800 replay parents,
    HVP-parent replay inclusion, validation disjointness, and zero Test100 access. Replay reached
    `1655/3200` successes plus one explicit symmetric-frame failure at `10:29`.
38. Train800 replay is now complete. The original array plus a transient-only rescue produced
    `3199/3200` strict geometry records covering all 800 parents; 799 parents have all four
    geometries and every parent has at least one. The only rejected point is
    `0000023.sample0`, a deterministic tensor/stationarity/frame mismatch rather than an I/O
    failure. Rescue completed `15/15` transient points without recomputing valid artifacts.
39. The authoritative replay descriptor cache also completed after a seven-shard low-concurrency
    retry. Its manifest SHA256 is
    `322a4597d7afcd89bdce8b7aec7bbbceaa8c42a198c63810d857786708455489`, its success-CSV
    SHA256 is `7cfc8e735ee15387e202be46391eac136da5e4186f970309e53c4b3f24feb83a`, and it contains
    3199 records under the 32,121-feature union schema. Runtime descriptor settings now fail
    closed against that schema.
40. Hash/dimension/disjointness preflight is ready and a 10-step replay smoke passed the complete
    scalar energy/force/Hessian forward, PCGrad backward, replay loading, logging and checkpoint
    path. The smoke exposed a real optimization issue: raw curvature gradients were commonly
    `1e5--3.7e5`, versus `1e2--1e3` for replay energy/force, while negative-cosine-only PCGrad did
    not limit positively aligned domination.
41. PCGrad now has an optional GradNorm-style cap on the curvature-to-replay gradient norm. It
    records the applied scale and balanced ratio before symmetric conflict projection. Defaults
    preserve historical behavior. Focused tests pass `26/26`, including a direct norm-cap unit
    test; the Stage-2 launcher uses a cap of `1.0`.
42. Replay-aware Stage-2 pilot job 3624 completed 5000 steps with the exact configuration
    `lambda_H=1`, Adam `3e-5`, HVP warm-up 1000, replay batch 4, and gradient-ratio cap `1.0`.
    The historical output directory name contains `h10`, but `resource.time` and `summary.json`
    are authoritative and record `lambda_H=1`. Full-Hessian relative-Frobenius median/max improved
    from `2.259/54.572` at step 0 to `1.102/2.230`; training-direction median/max improved from
    `2.221/55.291` to `1.084/2.273`; held-out-direction median/max improved from
    `2.197/68.085` to `1.146/1.437`. None of 20 parents is yet below `0.15` on any of these three
    metrics. Energy/force absolute-error medians on the 20 parents are
    `6.77e-3 Ha` and `1.45e-2 Ha/Bohr`. Numerical symmetry is sound: asym/sym median/max are
    `1.51e-5/6.10e-4`.
43. Job 3624 used 20:58 wall time, 13.4 GiB maximum host RSS, and 40.4 GiB peak GPU memory. Its
    best checkpoint is step 5000, SHA256
    `7da7b481f2278b4c6732e68249d7693c28007938392518881689b5867cd183a0`; summary SHA256 is
    `6dea43793cc11f3fe9f740d2685afc168c58c432543fdf617ea736e91cbed1b9`. All Stage-1/Stage-2
    scientific gates remain false, so frozen validation parents and Test100 remain unread. A new
    `continue-pilot` profile resumes both model and Adam state for 25,000 more steps at the same
    `lambda_H=1`; it is a Stage-2 continuation, not a promoted Stage-3 candidate.
44. Same-objective continuation job 3625 completed successfully. It resumed job 3624's step-5000
    best checkpoint and Adam state, used no second warm-up, and added 25,000 steps. Selection chose
    continuation step 23,800, i.e. cumulative step 28,800, because the last step had slightly lower
    curvature error but worse fixed replay loss. The selected full/train-direction/held-out-direction
    relative-Frobenius medians are `0.5658/0.5487/0.6792`; P90 values are
    `0.8143/0.8301/0.8670`; maxima are `0.9349/0.9381/1.0418`. No parent is below `0.20`, much
    less the `0.05/0.15` gates. Relative to s5000, full/train improve on `20/20` parents and
    held-out improves on `19/20`, so the run is effective but quantitatively insufficient.
45. The same selected checkpoint has 20-parent energy-error median/max
    `3.28e-3/1.64e-2 Ha` and force-MAE median/max
    `3.88e-3/9.66e-3 Ha/Bohr`; both strict anchor gates remain false. Force improves on `20/20`
    parents relative to s5000, while energy improves on `14/20`. Numerical symmetry remains far
    inside the self-consistency gate: asym/sym median/max are `1.51e-5/6.12e-4`.
46. Held-out failures are broad rather than one bad direction family. Per-direction median errors
    are `0.666` bond, `0.759` angle, `0.741` torsion, `0.630` random-internal, and `0.587`
    low-mode; no held-out direction is below `0.15`. Full-Hessian hard parents are
    `0016142` (`0.935`), `0132608` (`0.901`), `0049017` (`0.814`), and `0038511` (`0.805`).
    `0132608` is the held-out maximum (`1.042`) and the only parent whose held-out aggregate
    regresses from s5000. The frozen direction design's minimum-norm train-direction reconstruction
    already has median oracle error `0.851`, demonstrating weak identifiability of withheld
    orthogonal response from directional labels alone; the learned descriptor model beats this
    oracle on many parents but not the hard tail.
47. Job 3625 took 1:38:37, with 13.4 GiB maximum host RSS and 40.4 GiB peak GPU memory. Best
    checkpoint SHA256 is
    `83ac62943deb4fbc05cbe01428222c68a8b213c0c1f50df5eb9648e7e75b9b79`; summary SHA256 is
    `68ec2dffeaa044428003aca9ae2d1a4cad43557b21566d67b4cbe80760a657f2`. `/usr/bin/time`
    reports `391,234,164` filesystem-input units, so replay cache locality is a performance issue,
    but not the scientific failure source. All advancement gates remain false,
    and validation/Test100 access remains exactly zero.
48. The Stage-2 Slurm wrapper now passes `--hvp-warmup-steps` independently of replay, and the
    launcher has explicit no-replay smoke/control profiles. Job 3626 verified replay absence,
    warm-up, scalar E/F/Hessian forward/backward, logging and checkpoint persistence. The first
    formal no-replay attempt, job 3627, was stopped after about 500 steps and preserved because
    its inherited curvature-to-replay norm cap was not a meaningful no-replay control: the
    near-zero primary E/F gradient made the cap scale raw HVP gradients by about `3e-4`.
49. Job 3628 repeated the smoke with the norm cap disabled. Raw HVP parameter-gradient norms were
    about `1e4--5e4`, global clipping at `10` remained finite, and no NaN or OOM occurred. Formal
    job 3629 then trained the identical floor-1 h128 scalar for 30,000 steps with `lambda_H=1`,
    1000-step HVP warm-up, no train800 replay and no curvature/replay norm cap. It completed in
    `45:08`, selected step 29,400, used 13.4 GiB maximum host RSS and 40.3 GiB peak GPU memory.
50. The no-replay selected full/train-direction/held-out-direction relative-Frobenius
    median/P90/max are respectively `0.255/0.399/0.455`, `0.240/0.379/0.423`, and
    `0.410/0.688/0.906`. Full/train/held improve on `20/20` parents relative to the replay-aware
    result; paired median no-replay/replay ratios are `0.431/0.409/0.658`. Full-Hessian fractions
    below `0.15/0.20` are only `2/20` and `8/20`, while no held-out parent reaches `0.15`.
    Energy-error median/max are `6.18e-4/3.01e-3 Ha`, force-MAE median/max are
    `1.02e-3/3.17e-3 Ha/Bohr`, and asym/sym median/max are `1.58e-5/5.64e-4`.
51. Job 3629's best checkpoint SHA256 is
    `e0bf555eac59cd6f66fcd8ff63a61be82ff39a1634524636dcd6413aa5316ca2`; summary SHA256 is
    `416262ee2fdaa5008cd9335299fead48366310b48b34982c90cedec7532f7c4a`. The no-replay control
    proves that replay conflict/optimization drag is real, because it improves every parent's
    curvature and every force while halving wall time. It also proves that removing replay is not
    sufficient: the robust global h128 scalar still fails the full, training-direction, held-out,
    and strict E/F gates. The next registered capacity test is an exact function-preserving
    h128-to-h512 widening under the same no-replay split; validation and Test100 remain unread.
52. `expand_qm9_complete_total_hidden_width.py` implements that widening. It copies the original
    128 units in order, initializes 384 new incoming rows with a recorded seed, and sets every new
    output weight to exact zero. Therefore scalar energy, force and Hessian are identical at the
    expansion point; Adam state is intentionally not reused. The h512 initial checkpoint SHA256 is
    `e5fb4c632cccc79036510872de93ae8488c9bec6db11b38367e5910420ee9d8b` and its source is job
    3629's selected h128 checkpoint SHA256
    `e0bf555eac59cd6f66fcd8ff63a61be82ff39a1634524636dcd6413aa5316ca2`.
53. Focused node01 checks pass `12/12`, including exact energy/force/Hessian equality at `1e-14`
    absolute tolerance, rejection of non-expansion/Test100-exposed checkpoints, and all existing
    analytic scalar tests. Smoke job 3630 completed 10 steps in `3:06`; step 0 exactly reproduces
    h128 full/train/held medians `0.25515/0.24021/0.40983`. It remains finite through step 10,
    reaches `0.25518/0.23878/0.40801`, uses 13.4 GiB maximum host RSS and 41.22 GiB peak GPU
    memory, and records zero Test100 access. Smoke summary SHA256 is
    `bc0be07db7c719872d070b87e28d2198044ac6ae7261ef09aaf9d98854fe73ff`.
54. Formal no-replay h512 job 3631 is active on node01 for 30,000 steps with the identical split,
    floor-1 feature representation, `lambda_H=1`, 1000-step warm-up, Adam `3e-5`, global gradient
    clip `10`, no replay and no curvature/replay norm cap. Its result is not yet available and no
    advancement decision may be made from the smoke.
55. A width attribution control is preregistered before job 3631 finishes. The h128 and h512
    arms both start from job 3629's selected step-29,400 function, discard Adam state, reset the
    same seed/direction sequence, use the same 1000-step warm-up and add exactly 30,000 steps.
    Their only model difference at step 0 is 384 dormant h512 units. Without this h128-restart
    arm, any improvement could be caused by extra optimization steps rather than width.
56. Matched h128-restart job 3632 is active on node01. The read-only comparison utility
    `qm9_complete_total_compare_geometry_mlp_runs.py` is implemented and passes focused tests
    `2/2`. It requires identical parent IDs, rejects any run without an explicit frozen-Test100
    certificate, records all input hashes, and emits raw per-parent rows, run distributions,
    pairwise wins/ratios, training trajectories, and diagnostic plots. It is an audit tool, not a
    source of validation-parent or Test100 model selection.
57. The dormant h512 arm has an initialization audit risk: at step 4000 the 384 new units have
    output L2 norm only `7.37e-4`, versus `0.348` for the inherited 128 units. A second exact
    function-preserving expansion therefore uses 192 pairs with identical incoming weight/bias
    and output weights `+1e-3/-1e-3`. Each pair's scalar E/F/Hessian contribution cancels at step
    0, while its incoming-weight gradients are nonzero and opposite, so the new capacity can break
    symmetry immediately. Focused expansion tests now pass `3/3`.
58. Paired-expansion checkpoint SHA256 is
    `2453fdff695be3ab470709f592294bbb26fa6803e8b1bb695ac1343050cfff24`; the recorded maximum
    pair output sum and incoming-weight difference are both exact zero. Smoke job 3633 completed
    10 finite steps in `2:42`, with full/train/held medians
    `0.25518/0.23878/0.40802`, 13.4 GiB host RSS, 41.22 GiB peak GPU memory and zero Test100
    access. Smoke summary SHA256 is
    `d34bfefe225ee74ceabc9e71807d0113084479dbd9ef79fd7663315514c16c86`.
59. Formal paired-h512 job 3634 is active on node01 alongside h128 job 3632 and dormant-h512 job
    3631. The three matched arms differ only in hidden width/initialization; all use the same
    source function, reset Adam, seed/direction sequence, 1000-step warm-up, LR, 30,000 steps,
    loss definition and frozen parent split. No validation or Test100 metric is available to them.
60. A depth-two structural control is implemented without constructing a descriptor-space Hessian.
    Its scalar is `linear + shallow Softplus + deep Softplus residual`; analytic E/F/Hessian use
    two coordinate-space low-rank chain-rule projections. The formula agrees with float64
    `torch.func.hessian`, and the deep checkpoint expansion preserves the shallow scalar with 128
    pairs of identical incoming weights and `+1e-3/-1e-3` outputs. Geometry-model plus expansion
    focused tests pass `13/13`.
61. The deep h128x256 initial checkpoint SHA256 is
    `37adf84df97fb1b37033fed3e61502dd5ee4a92b79d5d2df3e6004dc64cfc950`; source is the same
    job-3629 step-29,400 checkpoint. Deep smoke job 3635 completed in `2:02`, remained finite for
    10 steps, and ended at full/train/held medians `0.25518/0.23879/0.40838`. It used 13.38 GiB
    host RSS and 40.28 GiB peak GPU memory, with zero Test100 access. Its summary SHA256 is
    `d5d4e80bf5e05dfd6cd921b2fab48e6887b0cf4ece32cf5ef471d7a5e4beef10`.
62. Formal deep h128x256 job 3636 is active for the matched 30,000-step no-replay Stage-2 control.
    It was launched because h128/dormant-h512 at matched step 12,900 were still only
    `0.214/0.206` full median and `0.376/0.362` held median, far from 10--15%. This is a structural
    nonlinear-interaction test, not another HVP-weight sweep. It cannot authorize replay or access
    validation/Test100 until its final gates are known.
63. The four matched 30,000-step global controls are complete. Selected
    full/train/held relative-Frobenius medians are: h128 restart
    `0.18370/0.16555/0.33014`, dormant h512 `0.14975/0.12818/0.29778`, paired h512
    `0.18186/0.16529/0.32950`, and deep h128x256 `0.18926/0.17122/0.33883`. Only dormant h512
    passes the full-Hessian median gate; none passes the train/held distribution gates. The h512
    summary/checkpoint SHA256 values are
    `8e9417a591e5c0b00e38e6622d55a654c677c074e45ac6b1e388ae8c57a2b8b7` and
    `cc76dcee62dfc826d1a0d79995abf7a07659ce2f6fc49b2a0e207e1f9f6d7ecc`.
64. Paired parent analysis prevents an aggregate-median overclaim: dormant h512 beats h128 on
    `16/20` full, `17/20` train, and `12/20` held aggregates, but paired median absolute gains are
    only `0.00240/0.00161/0.00383`. Persistent hard held parents are `0132608`, `0038511`, and
    `0091176`; held angle/torsion medians remain `0.376/0.373`. The frozen comparison summary
    SHA256 is `23a6cea4c3fa9f0fca0a5c6ac382363c454fc74d6a7c9c05d95c041093a51f80`.
65. Vibrational postprocessing also rejects promotion. Dormant h512 has mean frequency MAE/RMSE
    `129.91/231.86 cm-1` and mean mode overlap `0.62885`, but predicts 90 imaginary modes versus
    PBE's 33. The h128 restart values are `139.55/244.76 cm-1`, overlap `0.61857`, and 95 versus
    33 imaginary modes. Vibrational summary SHA256 is
    `0e91b9790511b62e39ba4fa05b0e409c077cbc6dacd2ebbf5c51dcb0829689d4`.
66. A train-only local-additive scalar route is implemented in
    `mldft/ml/models/components/local_body_order_residual.py` and
    `scripts/qm9_complete_total_local_body_order_capacity.py`. It uses element-typed pair,
    atom-centred triplet, and optional bonded four-body torsion terms, C3 cutoff, float64, no
    self-interactions, and exact scalar-derived force/Hessian/HVP. Focused local/trainer tests pass
    `14/14`; the Stage-2 direction analyzer additionally passes its sampled-log compatibility test.
67. The first local implementation exposed an optimization defect. Independent canceling outputs
    broke their zero baseline under Adam: batch-1 job 3642 ended with energy median/max
    `6.90e-3/1.39e-2 Ha` and selected step 0; full-batch smoke 3643 failed similarly. Outputs are
    now permanently parameterized as `+a/-a`, unpaired linear/constant shortcuts are removed, and
    parent sampling is without replacement. Locked-pair job 3645 remains stable for 1000 steps but
    changes full/held medians by less than `1e-5`, so this alone does not solve capacity.
68. Reference-geometry Taylor anchoring is now optional:
    `delta E(R)=g(R)-g(R0)-grad g(R0).(R-R0)`. It exactly preserves base-geometry scalar energy
    and force while retaining `d2g/dR2`; it is a conservative local curvature correction, not yet
    a globally transferable energy functional. Full-train-direction one-step audits confirm that
    full/train/held all improve while E/F remain bitwise at the h512 source values. The selected
    step size is pair/triplet LR `1e-4`; `1e-3` worsens train/P90 tails.
69. The 20-step anchored pair+triplet run 3655 selects step 20 at
    full/train/held `0.14866/0.12797/0.29602`; held P90 improves `0.55444 -> 0.55144`, but held max
    worsens `0.89332 -> 0.89433`. E/F are unchanged. Summary/checkpoint SHA256 values are
    `fdbad0b4c7dd0b5f38740c6d2adbee4db575a85fbb41c29b4ff3be6918d26d9c` and
    `ac3c9c3627406819e0c650f6612e3916942bae8c0caa75a7b0a50acd0407dbfc`.
70. Direction-family analysis shows the anchored pair+triplet correction improves held bond/angle
    medians by `0.00227/0.00174`, but worsens held torsion/random/low-mode by
    `0.00187/0.00166/0.00093`; only 12/20 parents improve in held aggregate. The analysis summary
    SHA256 is `83138f6327756947e98de605e1bd4b834af33ac9c3b33411910cda7a21c86113`.
71. A parity-even bonded-chain torsion scalar is therefore the active structural control. Its raw
    HVP gradient is about 780 times the pair/triplet gradient, so uniform LR `1e-4` job 3656 is a
    rejected scale-mismatch run. Separate LR audit job 3657 with pair/triplet `1e-4` and torsion
    `1e-6` restores a favorable one-step result (`0.14944/0.12807/0.29760`) with unchanged E/F.
    Matched 20-step job 3658 completed at full/train/held medians
    `0.148468/0.127900/0.295843`. Relative to pair+triplet job 3655, torsion improves held median
    by only `1.75e-4` and worsens held P90/max to `0.551879/0.895991`. Its torsion-family held
    median improves by `0.00154`, but remains slightly worse than the h512 source. It is rejected
    for continuation. Summary/checkpoint/training/per-parent/resource SHA256 values are
    `677adbc57e05a011e3926d167c918f9e3fb61d09bff573b2639df47e17f3e998`,
    `a1a79c1d8a2a8c7206a61744290c27ca5e0dcf40121760218510801ca6169aca`,
    `61ee723049e025362ad9e3252399610d652e04a4f1f90d15b05a54f5c673f4b`,
    `f486aa00a7d3dffbe5f641237e5eca565ec380e16247c5d845fbe844a1df488d`, and
    `38e13a1eed84d61de6467e2520c70142801be41b44c20f9f8d2645a8f9432451`.
72. A reference-anchored linear descriptor scalar then tested whether convex optimization could
    separate representation from optimizer failure. Fixed ridge `1e-6` overfits train directions
    and gives held median `0.319409`. Train-only calibration selects ridge `1` at
    full/train/held `0.147407/0.126715/0.296278`; a wider ridge boundary selects `30` and returns
    `0.148567/0.127867/0.297531`. Neither improves the held-direction distribution. The two
    train-CV summary hashes are
    `9c0f6fbaaac5df8b9f991e8e1292533a3658ca7a768a2d78a74fc3936771af07` and
    `7c8142ecff40a5acf34164f785574b06f7a4451adb01324f8f809dc064a2d5c1`.
73. A reference-anchored internal-coordinate quadratic scalar was evaluated next. Diagonal,
    shared local-cross, and exact-context local-cross variants give full/train/held medians
    `0.148798/0.127999/0.293847`, `0.147369/0.125154/0.297781`, and
    `0.146161/0.124624/0.295735`. The diagonal variant has the best shared held median but remains
    almost twice the `0.15` gate and has no convincing tail improvement. Summary hashes are
    `9c2589af9559aa5a649ba3a9190617207063942da0660ea147f3c8ffc79eac8b`,
    `25813456869ebf203963bea85697f0d1d74731114461d12a0e066016a41b76c7`, and
    `bd282e73da1d56b85352b57d435ac82793a04b4b7b6f7e222125e2ef106bbadd`.
74. A parent-specific full-cross oracle solves each parent independently. It reaches
    full/train/held medians `0.110910/0.086689/0.274146`; `80%` of full Hessians are below `0.15`,
    but only `20%` of held-direction aggregates are below `0.15`. This is a useful in-parent
    upper bound, not a transferable model. It shows that fitting train-covered matrix components
    is possible while the unobserved response remains unidentified. Summary SHA256 is
    `e5035fc18f56a1854afb37f11b747994ee8086c22f1c5cfd224bddd8fd209caa`.
75. A shared continuous local-environment coefficient network was implemented to predict the
    diagonal and local-cross quadratic force constants. The initial zero-output layer blocked all
    hidden-layer gradients, so initialization now subtracts a frozen seeded initial function:
    the scalar E/F/Hessian correction is exactly zero at step 0 while every layer receives
    gradients. Focused float64 scalar/Hessian/HVP and initialization tests pass `6/6` on node01.
76. The corrected matched runs still fail. Job 3673 (`lr=1e-5`, 500 steps) selects step 475 at
    full/train/held `0.150308/0.128905/0.297915`; its summary SHA256 is
    `3a01e2c137eec399b55fa2b70943021578854d2d6f8e4cec42c2611be5f42929`.
    Job 3674 (`lr=3e-6`, 1500 steps) lowers train-only calibration loss
    `0.034185 -> 0.033369`, proving that the network is optimizing, but finishes at
    `0.149694/0.128291/0.298094`. Its summary/checkpoint/training SHA256 values are
    `efdbcc9ba7284dea576c539540d3e229d5940667de02f5f2b3c401f08220f035`,
    `c7e88fe99f9ca9849d093f3ad557e29b36ca7d51ac7c46e6563e682aaa94b3f1`, and
    `7c495bde29b46b1ab802dfe6ebc83b5c9feeb3837d3f41a6fb1fca5959c070af`.
77. Final frozen-direction analysis compares source h512, pair+triplet+torsion, the parent oracle,
    and the corrected shared network. The network changes held angle/bond/low/random/torsion
    medians from source `0.37632/0.24581/0.28494/0.28862/0.37320` to
    `0.37519/0.24594/0.28488/0.28978/0.37103`; this is not a stable family-wide gain. Analysis
    summary/direction-table SHA256 values are
    `376a15096f1dc0548538d5d4f9b629ac29b0f716737b61432294ff4d2150e8a4` and
    `3e7b49b7be0ef5e3d797cf99a4704112678dc9204f61a5e865972ff8ff294f41`.
    At that checkpoint all advancement gates remained false and Test100 access was exactly zero.
78. A global all-cross invariant quadratic-coefficient network was then tested after fixing the
    zero-output hidden-gradient blocker. Job 3676 optimized its calibration loss but finished at
    full/train/held `0.149663/0.127960/0.298167`; the shared local coefficient hypothesis was
    rejected. Summary/checkpoint SHA256 values are
    `843fa43b959a43ea7cbc956f669bc099adddb19b060ad52934a84ecfc31f29d1` and
    `e9e2cfcc1a4967a552fede60df7fb5b9146c410e373a222fa3e056f2beaf6831`.
79. A source-Hessian spectral scalar operator was implemented next. Its reference-anchored
    Cartesian quadratic has exact zero energy/force correction at `R0`; its full Hessian is an
    orthogonally equivariant matrix function of the complete-total source Hessian. The spectral
    basis alone improved held median `0.297778 -> 0.278358` but failed the tail gates. Even a
    read-only source-eigenvector oracle stopped at held median `0.219824`, proving that fixed source
    eigenspaces cannot reach the target.
80. Rotation-equivariant atom-pair and diagonal block operators changed the mode subspace and gave
    job 3678 full/train/held medians `0.091196/0.076854/0.165103`. This was the first shared model
    near the full-Hessian target, but held P90/max `0.283774/0.406568` still failed. Adding
    low-rank chemistry conditioning in fresh direction protocol v4 yielded full median/P90/max
    `0.077692/0.098810/0.106594` and held `0.087741/0.131306/0.146566`, but the all-parent train
    direction maximum remained `0.104691`, above `0.05`.
81. `symmetric_hvp_interpolant` supplies the exact minimum-rank symmetric matrix completion of
    observed HVP actions. Fresh protocol v5 (manifest SHA256
    `08fca6c89754ed33015f76975e5ff46c47a3fc41d83a99bc6a42edb7d22cf32b`) combines that completion
    with the shared spectral/block operator. Job 3680 reaches full median/P90/max
    `0.012431/0.022199/0.025952`, train HVP `1.55e-5/9.18e-5/5.48e-4`, and held HVP
    `0.049450/0.087562/0.117927`; all 20 labeled-parent Stage-2 gates pass. Summary/coefficient
    SHA256 values are `4539a2b45b0050ff7674c7b0e2a67aff6d48646a40d8f9adf188a74bfac26217`
    and `a7b62e5c5d3916d69fb2f49b638255e0b5b9f7e0c358b09888c4bf7fe755ae19`.
82. This v5 pass is explicitly label-conditioned: exact completion needs each parent's training
    HVP labels and cannot be applied to an unseen parent. The shared operator without completion
    is the only transferable part. A matched replay-aware source control also passes Stage 2 at
    full median/max `0.014182/0.053289` and held median/max `0.054837/0.137633`; its
    summary/coefficient hashes are `ae68b0463720b7ea0dd64fe96a11c28179b9cbc0fcb9219dc035cc3ef8a4bc73`
    and `690a93f0e111364c217fd3bfd8964d09c3b9a9f88278e0ee5762ef7dd8ec9d82`.
83. Labeled-parent vibrational postprocessing confirms the capacity ceiling but not molecule
    generalization. h512/replay v5 frequency MAE/RMSE are `8.33/21.88` and
    `10.89/28.56 cm-1`; mean mode overlap is `0.9535/0.9360`, and both predict 31 imaginary modes
    versus PBE's 33. The vibrational summary hashes are
    `c5aa41eb9f2496ecada8a6b0452d87ed5e2a517ea4f5e6e2dc9018c82b0c0f93` and
    `07f369f6eee3f7e62345029696c1821f4a1b2aee1cfeaddcb45af83d384ebeae`.
84. Before reading shared-operator validation Hessians, protocol
    `qm9_complete_total_hessian_unseen_parent_shared_operator_v1.yaml` froze two source arms,
    seven independent validation parent IDs, exact artifact hashes, no per-parent fitting or HVP
    labels, and Hessian plus 5% median/P90 E/F regression gates. Protocol SHA256 is
    `c0e2accee7452c68bc54f2fcfd81fb2dec1771be0211e2a2ab73b9e77c954743`.
85. The h512 and replay source predictions over the seven validation parents took `37:42` and
    `36:30` on node01 (one A100 plus eight CPUs), with maximum RSS `1.73/1.32 GiB`. Their source
    Hessian median/P90 relative Frobenius values are `2.225/30.633` and `1.426/38.193`. The source
    summaries have SHA256 `437ad8f0c16cd8a46220a0220fc486fd875e15123b5581cbd05903e9ea3cf13f`
    and `93cdf5adf722a1fb8b15bc1d41963a879e95cad072f31059269e0580279c22e7`.
86. Applying only the frozen shared operator, without exact completion, fails unseen-parent
    generalization. h512 gives median/P90/max `8.881/251.613/609.675`, `0/7` below `0.15`, and
    improves only `2/7` parents. Replay gives `5.140/1515.609/3780.136`, `0/7` below `0.15`, and
    improves `3/7`. Their summary hashes are
    `afa6bbeff1a338fe781db2d34440742d998836eede815da83a1dc1b9bfcb45f5` and
    `c6d5902539a8ea566438edcf6e825c053c4e292ff45c742126b5e86aebd172ce`.
87. Replay preserves the useful E/F path: validation force median/P90 improve from original-A
    `0.1025/0.1683` to `0.0323/0.0679 Ha/Bohr`, and energy median improves
    `0.1156 -> 0.0696 Ha`; energy P90 is `0.2019 Ha` versus the 5%-allowed `0.1987 Ha`, so one E/F
    gate still fails. h512 fails both energy gates and the force P90 gate. The anchored Hessian
    operator itself changes neither source energy nor source force at `R0`.
88. Vibrational validation also rejects both arms. h512 source/shared frequency MAE are
    `1480.6/5768.6 cm-1`; replay source/shared are `1390.6/6706.0 cm-1`. Shared h512 predicts 108
    imaginary modes and shared replay 63, versus PBE's 12. Mean mode overlaps are only
    `0.5703/0.6204` for the two shared arms.
89. OOD diagnostics identify an unbounded extrapolation mechanism. Across the 14 arm-parent
    points, frozen parent-feature `max |z|` versus `log(1+||Delta H||F/||H_PBE||F)` has Spearman
    `0.899` (`p=1.24e-5`) and Pearson `0.935`. `0056113` has `max |z|=41.90/32.23` and correction
    scales `572.5/3871.2` times the PBE Frobenius norm. Three replay samples with `max |z|<2`
    improve to `0.191/0.245/0.689`, so the operator contains useful structure but is not bounded or
    chemistry-transferable. Diagnostic summary hashes are
    `3db0a08e62d82ad60484abecf4f960cf0b401dca9d4a92a9d3fadc968a3709f8` and
    `d4cc1cef9171395da0a2f09424608b96d279e2039a6979d68895652f6a090a40`.
90. The paired tables and plot are under
    `validation_spectral_block_shared_operator_comparison_v1`; summary/plot SHA256 values are
    `6a541e046844a977b041ab34ad7afe1e622c327025c560a494f66efc79c34591` and
    `27bed3be797e051886638707afdd9cb970c66aefc3420c2e670ed5402c31621b`.
    The unseen-parent gate fails, so train100 expansion is not authorized and Test100 remains
    exactly unread (`0` evaluations).
91. A replacement train-only protocol was frozen before fitting at
    `configs/audit/qm9_complete_total_hessian_parent_cv_bounded_operator_v1.yaml` (SHA256
    `b30e74c02ec244fce095f725ba34c3c637412994a58de048f025a6f3540183da`). It uses only the 20
    labeled train parents in five deterministic natoms/composition-balanced folds. Every held
    parent is excluded from feature normalization, coefficient fitting, HVP completion, and ridge
    selection. Its full PBE Hessian is used only as that fold's held-parent score. The seven
    external validation parents and Test100 remain unread by this protocol.
92. The source is now the strict complete-total original-A scalar baseline at `sample_id=0`, not
    an h512 or replay curvature-trained checkpoint. Energy, force and Hessian all come from that
    same scalar total energy. The reference-anchored quadratic correction changes source energy
    and force exactly by zero at `R0`; there is no force head. This isolates cross-parent curvature
    transfer from leakage through an already curvature-trained source.
93. The original-A held-parent source has Hessian relative-Frobenius median/P90/max
    `2.7331/3.1528/3.5683`, with `0/20` at or below `0.15`. Its frequency MAE/RMSE are
    `2921.2/3800.9 cm-1`, mean mode overlap is `0.5921`, and it predicts `809` imaginary modes
    versus PBE's `33`.
94. All seven preregistered five-fold variants completed. The table reports held-parent metrics;
    none passes the frozen median `<=0.10`, `>=80% <=0.15`, and P90 `<=0.20` gates.

    | arm | transform / conditioning / cap | ridge | median | P90 | max | <=0.15 | wins vs source |
    |---|---|---:|---:|---:|---:|---:|---:|
    | A | standard / yes / none | `1e-4` | 0.3200 | 1.0912 | 22.0585 | 5% | 95% |
    | B | tanh(2) / yes / none | `1e-4` | 0.3540 | 0.6220 | 19.8579 | 5% | 95% |
    | C | tanh(2) / yes / 1.0 | `1e-4` | 0.8901 | 0.9744 | 2.9876 | 0% | 95% |
    | D | clip(3) / yes / 1.0 | `1e-2` | 0.8785 | 1.0513 | 3.4507 | 0% | 95% |
    | E | tanh(2) / no / 1.0 | `1e-4` | 0.8666 | 0.9323 | 3.1404 | 0% | 95% |
    | F | constant / no / 1.0 | `1e-2` | 0.8645 | 0.9048 | 3.2037 | 0% | 95% |
    | G | tanh(2) / yes / 0.5 | `1e-4` | 1.7933 | 2.1572 | 2.7291 | 0% | 100% |

95. A/B establish real cross-parent signal: both improve `19/20` parents and reduce the median by
    about one order of magnitude. They still fail because `0132419` extrapolates from source
    `2.7903` to `22.0585/19.8579`. A blanket correction cap removes that explosion but also removes
    the useful correction: C--F cluster near median `0.86--0.89`. Dropping parent conditioning has
    the same floor, showing that chemistry conditioning is necessary but the current linear
    conditioning is not trustworthy out of distribution. Tightening the cap to `0.5` is worse.
96. Vibrational ranking is consistent with this diagnosis. A/B reach frequency MAE
    `282.6/276.2 cm-1`, RMSE `534.7/502.3 cm-1`, and `87/85` imaginary modes. C--F have frequency
    MAE `1068.7--1114.0 cm-1`; G has `2322.4 cm-1`. A/B are a large improvement over original-A,
    but still miss the `100--200 cm-1` target and retain about 2.6 times PBE's imaginary modes.
97. Parent-CV jobs 3707--3713 used eight CPU cores on node01 and took `0:32--4:02` wall time,
    with batch-step maximum RSS `1.30--2.71 GiB`. Vibrational jobs 3721--3728 took `4--6 s` each.
    Jobs 3714--3720 were transient 3--5 second wrapper failures caused by premature jump-shell
    variable expansion; they produced no scientific output and explicit-path resubmissions
    completed successfully. Analysis job 3729 took 7 seconds.
98. Unified artifacts are under `parent_cv_original_A_v1/analysis_v1`. The summary, variant CSV,
    per-parent CSV, and plot SHA256 values are respectively
    `409fbe315e7999adbd9d78452517d189cb2bce2317955386c2c00b11372011d4`,
    `0480314d1ccb190fa8ba7356ee681fdebc8adf9c076f2ce602c6ba92c957d82f`,
    `8acdb5ad686588603ec1a324f18edfc3d053165d88175d0fb52a5c5702b4e3d5`, and
    `72f14795d9b57ebf1700569abe8af75eb8eb1d8342e0c65c218bb2be84e7997c`.
    The machine-readable decision is `advancement_authorized=false`; validation access remains
    false and Test100 remains exactly zero.
99. A separate geometry-scalar parent-CV protocol was frozen before training at
    `configs/audit/qm9_complete_total_hessian_geometry_parent_cv_v1.yaml` (SHA256
    `5c81212f80f4c396214e748d8e3b0fc7990117a85188ac216f2b4ed4907fc479`). The five folds and all
    input hashes are fixed. G0 is a no-replay diagnostic; G1 adds train800 E/F replay but is
    explicitly allowed only if G0 first proves transferable curvature. Held-parent Hessians are
    not used for gradients, checkpoint selection, normalization, or hyperparameter selection.
100. G0 uses a zero-initialized float64 Softplus geometry-scalar MLP (`32121 -> 128 -> 128 -> 1`),
    exact scalar-derived E/F/full Hessian, 30000 steps, Hessian weight 10, E/F weights 1, PCGrad,
    and the same five 16-fit/4-held folds. All five fit sets reach the intended capacity range:
    final fit16 Hessian relative-Frobenius medians are `0.0212--0.0285` and maxima are
    `0.0384--0.0688`. This establishes adequate memorization capacity within each fold.
101. Cross-parent transfer fails broadly. The 20 held-parent median/P90/max relative Frobenius is
    `1.7305/5.2744/17.6050`; `0/20` are at or below `0.10`, `0.15`, or `0.20`. It improves
    `15/20` parents versus original-A, but even the best held values remain approximately `0.99`.
    The failure therefore is not only the `0016298` tail (`17.61`); the learned global descriptor
    response does not span the held-parent Hessian correction.
102. E/F and vibration checks reject G0 independently. Held force median improves by a factor
    `0.823` to `0.0838 Ha/Bohr`, but energy median worsens by a factor `8.83` to `0.8140 Ha`.
    Frequency MAE/RMSE improve from original-A `2921/3801` to `1251/1868 cm-1`, and imaginary
    modes drop from `809` to `300` versus PBE's `33`; mean mode overlap worsens from `0.5921` to
    `0.4911`. The frequency, overlap, energy, and all Hessian gates fail.
103. Two read-only diagnostics rule out simple post-hoc repairs. Held error correlates strongly
    with correction magnitude (Pearson `0.9885`, Spearman `0.7248`), but an oracle scalar
    amplitude fitted separately for each held parent still gives median/P90/max
    `1.605/2.337/2.569` and `0/20 <=0.15`. By contrast, spectral arms A/B have correction-target
    cosine medians around `0.994`, yet their oracle amplitudes still leave medians `0.297/0.319`.
    A strict-fold linear combination of A/B/G0 corrections also remains near median `1.56`.
    The missing object is a transferable response direction/subspace, not a global scale, cap, or
    ensemble coefficient.
104. Array job 3730 completed all five G0 folds on node01 in `33:39--34:11` each using one A100.
    Batch-step maximum RSS was `10.7--11.4 GiB`; logged peak GPU allocation was approximately
    `34.6--35.1 GiB`. Read-only analysis job 3735 took `1:05` and `0.64 GiB` RSS. No node01 project
    jobs remain active after this audit.
105. G0 artifacts are under `geometry_parent_cv_v1`. Analysis summary, held CSV, variant CSV,
    held-row JSON, plot, and vibrational-summary SHA256 values are respectively
    `5b69d06920c71b9858543b173abc6ea865f9d5f292b9206482a25b83600e93b9`,
    `24ce3ee9d37fbb95d97394d7e5d2a74ab43942d25b2916a48f3771b41ac56b76`,
    `503ff00a62f2a1fd8ea81d9862c9fc962ffa90b34a5d203f5b50d998eaa6503f`,
    `1d5697c53c1d8fc83144479d63932fc0e31c346dea0217057982d4f9ac2e7862`,
    `40e4b2e096e84c0b80ea47357f87261853bb780db98a2385ff7d60f7bb117d7e`, and
    `4d5634be055dbdabd35b3ed57847672bf866943e946d1b58bf3d16ed133cc322`.
    The machine-readable decision is again `advancement_authorized=false`, with validation and
    Test100 access false/zero.
106. The preregistered fail-closed rule now applies: G1 replay is not launched, because replay can
    protect E/F but cannot establish a response basis that G0 failed to learn. The next experiment
    must use a separately frozen local-additive or equivariant scalar response representation. It
    must first fit full Hessians on fit-only parents to `1--5%`; only then may the fixed five-fold
    parent-heldout audit run. No validation, train100 expansion, or Test100 access is authorized.
107. The replacement fit-only protocol is frozen at
    `configs/audit/qm9_complete_total_hessian_local_scalar_capacity_v1.yaml` (SHA256
    `012420b6b3a92fdef24ba3dfc151f01513dbb43e050514f35193ce10dfaa70e4`). It opens only stable5
    parents `0028399`, `0031108`, `0132419`, `0031012`, and `0121249`; the trainer records all ten
    opened label/capacity paths and certifies `unselected_parent_artifacts_opened=0`. Validation
    and Test100 remain unread.
108. The scalar definition is now explicit and conservative at each reference geometry:
    `E_local(R)=E_A-F_A*dR+0.5*dR^T sym(H_A)dR+C_theta(R)-C_theta(R0)-grad C_theta(R0)*dR`.
    The original-A FD Hessian is symmetrized before it enters this local quadratic scalar. The
    correction is invariant, anchored to zero value/gradient, and owns its force/Hessian through
    autograd. This is a reference-geometry Hessian model, not yet a global displaced-geometry
    OFDFT functional.
109. Two preregistered capacity arms compare the existing typed pair/triplet/torsion scalar with a
    new smooth invariant distance-message-passing scalar. The latter has 153,889 trainable
    parameters, 16 smooth radial functions, three Softplus message layers, no self-loop edges,
    and a frozen-initial-function subtraction. Remote focused tests pass `14/14`, including finite
    second derivatives, Hessian symmetry, H-loss parameter gradients, rigid-motion invariance, and
    all previous local-body-order regressions.
110. Node01 smoke job 3736 completed both arms for ten full-Hessian steps. Typed/message-passing
    median relative Frobenius changed from the symmetrized original-A `2.7268` to
    `2.6926/2.4111`; maxima are `3.3727/3.0546`. Both preserve energy/force to numerical precision
    (median error ratios `1.0`) and have scalar Hessian asymmetry below `3.3e-15`. Wall times are
    `71.9/60.0 s`, peak GPU allocations `1.10/1.89 GiB`, and host RSS about `1.4 GiB`.
111. Formal 5000-step fit-only job 3738 was launched on two node01 A100s under the frozen protocol.
    Its preregistered decision rule kept parent-CV unauthorized unless a formal arm had both
    stable5 median and maximum relative Frobenius at or below `0.05` while retaining E/F and
    symmetry gates. Final results are recorded in point 113.
112. A rigid-mode projection audit quantifies the cost of anchoring the correction force to zero.
    For any invariant scalar with zero correction gradient at `R0`, the closest admissible symmetric
    correction is `P * DeltaH * P`, where `P` removes translation/rotation directions. The resulting
    unavoidable stable5 relative-Frobenius floor has median/max `0.04039/0.04178`. Thus the 5%
    capacity gate is mathematically possible but has less than one percentage point of residual
    margin. The preregistered response to a failure above 5% was to jointly fit E/F/H without force
    anchoring, not add steps or width to the anchored model. Audit JSON SHA256 is
    `17b004d855649b48c8fecf242070fd657e37a00610b0762837284fbe8cd709d1`.
113. Formal job 3738 completed both 5000-step anchored arms and neither passes. L0 typed
    body-order has stable5 Hessian relative-Frobenius median/P90/max
    `0.93969/1.01040/1.05517`; L1 message passing has
    `0.94353/1.01311/1.05787`. Both preserve source E/F exactly and have asymmetry below
    `1.8e-14`, but remain about twenty times above the 5% fit-only gate. L0/L1 wall times were
    `3:15:30/2:45:18`, peak GPU allocation `2.04/1.94 GiB`, and host RSS
    `1.42/1.39 GiB`. Both selected step 5000, so this is not an early-checkpoint artifact.
114. Read-only job 3740 confirms the failure is directional, not only an amplitude error. L0/L1
    median correction-target cosines are `0.9401/0.9623`; their correction/target Frobenius ratios
    are `0.9585/0.9115`, yet even a per-parent oracle scalar amplitude leaves median relative
    Frobenius `0.9296/0.9275`. Analysis summary and plot SHA256 values are
    `d176732c8c71501456738695f744d3ad6ca26b92265313642367868dab8c1e08` and
    `eef316e83a0179337a44ab3e50846447c02ad94615bd38501cc25b8b880d2be2`.
    `parent_cv_design_authorized=false`; validation and Test100 access remain false/zero.
115. The zero-force anchor is therefore removed for the next fit-only audit. Protocol v2 defines
    one scalar
    `E_A-F_A*dR+0.5*dR^T sym(H_A)dR+C_theta(R)-C_initial(R)` and derives its energy, force, and
    Hessian by autograd. It opens the same five parents only. The first v2 smoke (job 3742) improves
    median energy/force ratios to `0.580/0.851`, but raw Hessian-task parameter gradients are only
    `1e-5--5e-4` of E/F-task gradients. PCGrad resolves directional conflict but cannot amplify
    that weak Hessian task, so the unbalanced v2 formal run is explicitly forbidden.
116. Superseding GradNorm protocol
    `configs/audit/qm9_complete_total_hessian_local_scalar_joint_capacity_v2_gradnorm.yaml`
    (SHA256 `5bed589cd57bdc697f992ee350b5040aeb89ab8b0c6cfe802370376b7cf63f11`)
    adds bounded task-norm scaling before PCGrad and changes no architecture or data. Job 3743
    verifies the path: the scale reaches its `1e4` cap, the balanced Hessian/E-F gradient-norm
    ratio rises from `0.12` at step 2 to `0.99` at step 10, all values are finite, and peak GPU
    allocation is `5.83 GiB`. The ten-step smoke is not an accuracy result; its Hessian median is
    still `2.6687` while energy/force ratios are `0.653/0.871`.
117. Focused local-scalar, second-derivative, PCGrad, and GradNorm tests pass `18/18`; both v2
    protocols, the trainer, and wrappers pass static/shell checks. The trainer records raw task
    gradient norms, GradNorm scale, pre-projection cosine, conflict flag, balanced norm ratio,
    E/F/H losses, full stable5 metrics, memory, opened artifacts, and Test100 access count.
118. Formal 3000-step GradNorm job 3744 is running on one node01 A100; dependent read-only
    analysis job 3745 is queued. The formal output root is
    `local_scalar_joint_capacity_v2_gradnorm/formal/J2_unanchored_message_passing_gradnorm`.
    It remains a stable5 fit-only capacity test. A separately frozen parent-heldout protocol is
    allowed only if median and every-parent relative Frobenius are at most `0.05`, median E/F
    ratios are at most `1.05`, and asymmetry is at most `0.005`.
119. Periodic JSONL observability was extended after 3744 started. Future runs now log full-set
    median energy error, force MAE, E/F source ratios, asymmetry, and checkpoint selection score in
    addition to sampled losses and Hessian metrics. The running process is unchanged. Updated
    trainer/test SHA256 values at that patch point were
    `492e3acaf6dbdc3f53d82510946762f982fb0d31fd8d82d063b85112c20b1310` and
    `0ec834bc7fbed0456ba41b21fa6404159f794a5ea521110a629096e54916bfa3`;
    remote focused tests pass `6/6`.
120. A fail-closed activation/Jacobian audit is prepared but not submitted. Frozen protocol
    `configs/audit/qm9_complete_total_hessian_local_scalar_activation_audit_v3.yaml` compares
    a same-seed beta-1 Softplus control against beta-5 Softplus, SiLU, and tanh with the same
    scalar, data, architecture size, optimizer, and gates. Its contingency permits smoke only after
    J2 formally fails; at most the best one or two non-control arms may then run formal. The purpose
    is to test the observed four-to-five-order initial
    Hessian-gradient deficit, not tune percentages on held parents. Protocol/wrapper SHA256 values
    are `b510aaf7a8f3cc1f320565d5141af42006e88f9fb9b30a9abc2bc548c071c715` and
    `e1f1c81cbc9b8cee94574039916c64467a47fc2eac61e78ddafb7da4260e3431`.
    All activation paths have finite scalar-derived E/F/H, symmetric Hessians, and nonzero
    finite Hessian-loss parameter gradients; remote focused tests and static checks pass `10/10`.
    The updated model/trainer SHA256 values are
    `e02300ae8915b8f4d2b235d5ce748567dd52bcbb2ef2b87bb162c70e50d92b30` and
    `77e3a5d85018d9e249bb23cb711c511d80e37b298aa3335eac34c879690360a9`.
    Smoke promotion is preregistered: all diagnostics must be finite, final joint selection score
    and median raw H/E-F gradient ratio must both beat the same-seed control, and at most the best
    two non-control arms may run formal.
121. A separate read-only gradient-conflict audit is queued as job 3746 with dependency
    `afterok:3744`. It evaluates only J2's frozen best checkpoint and stable5, reporting separate
    energy, force, and Hessian parameter-gradient norms and all three pairwise cosines globally and
    per parent. It performs no optimizer step and records opened paths plus Test100=false/zero.
    Script/wrapper/test SHA256 values are
    `6f3217598d97e66777f8f00066c3714df0a70033802863c90329eaebeeec0128`,
    `6860a136f4de3b9fcb4067f35e14f15c2ec630be830af1adcd89a54b6a931edb`, and
    `e3f626f2046712b8396f73d26e40e5bc7f5bc38a8bedc8f30669f1f982132a2e`.
    Remote unit/static checks pass `1/1`.
122. J2 formal job 3744 completed 3000 steps in `2:19:05`; analysis 3745 completed in 7 s.
    The jointly selected best is step 2500. Energy/force median source ratios are
    `0.6637/0.9508`, and asymmetry is `1.05e-14`, so all non-Hessian gates pass. Hessian
    median/P90/max relative Frobenius is `1.0363/1.1772/1.2257`; both 5% Hessian gates fail by a
    large margin. Formal summary, checkpoint, training metrics, analysis summary, and plot SHA256
    values are `9d6e4ee8497cf22e3fd3d48d085bbb0a56be6edd50441fcfff34967ed38af8b0`,
    `213ad79ddf4095ddd0e09b72b6fb90aa7e48f8967766653d359d4a2a7a828dff`,
    `a764d74b9a0683b6c9e837cdd72e3f2170aeac23adfc973fb9e0f87edc6a5496`,
    `1930cf47d24e5cd20ab6df3501646c7bd9764484e6622a042ad913ec75b2e370`, and
    `647af757187acf45c04986bc668fc8e4b41003349b7496f08350cf678eec43d3`.
123. Correction diagnostics confirm a response-direction failure. J2's median correction-target
    cosine is `0.9263`, correction/target Frobenius ratio `0.9286`, and even per-parent oracle
    scalar rescaling leaves median relative Frobenius `1.0046`. `parent_cv_design_authorized=false`;
    validation/Test100 remain false/zero. This is not an E/F forgetting failure and more J2 steps
    are not authorized.
124. Read-only job 3746 completed in 48 s. On the best checkpoint, aggregate E/F/H parameter
    gradient norms are `103690/5332/17.4`. Pairwise cosines are E-F `-0.9578`, E-H `+0.8923`, and
    F-H `-0.9675`. Force-Hessian conflict is negative on 4/5 parents and below `-0.90` on those
    four. The previous E+F combined task therefore hides a severe internal conflict while H remains
    orders of magnitude weaker. Audit JSON SHA256 is
    `eceb1a75f460fd98fc56292926ca350bf88d534d9dfa77416bba676db6613036`.
125. Contingent activation smoke array 3747 completed all four same-seed arms with finite positive
    step diagnostics. The beta-1 control has joint score `5.9663` and median raw H/E-F gradient
    ratio `1.75e-5`. J5 tanh scores `5.4603` with ratio `3.95e-4`; J3 beta-5 Softplus scores
    `5.9638` with ratio `2.59e-4`; both pass the frozen promotion rule. J4 SiLU has ratio
    `7.18e-4` but score `6.0764`, so it is rejected. Machine summary/CSV SHA256 values are
    `361e4fd32152a427d5f1f073faef6dd2055dcbb9c128fba375904e519b717207` and
    `4571b87ead8254ecb48d7ea957b02faa1bd706310f43aed94fbbeedec8c7f5c4`.
    Formal array 3751 runs only J3/J5 on node01; no other parent is opened.
126. Formal 100-step results favor J5 but are not a final decision. J5 tanh reaches Hessian
    median/max `1.0928/1.2950`, energy/force ratios `0.4157/1.0749`, and scalar asymmetry
    `2.4e-15` in 545 s. J3 beta-5 Softplus reaches `1.8907/2.4101`, E/F ratios
    `0.7383/1.7139`, and asymmetry `1.0e-15` in 659 s. Extrapolated 3000-step wall times are about
    `4.5/5.5 h`, within the 8 h limit. Both runs remain stable5-only and must complete their frozen
    3000-step profiles before the gate is decided.
127. J5 reaches step 200 in 1095 s with Hessian median/P90/max
    `1.0051/1.0829/1.1266`, energy/force source ratios `0.8574/0.6978`, and asymmetry
    `2.48e-15`; J3's latest record remains step 100. J5 is improving and currently preserves E/F,
    but it is still about twenty times above the 5% Hessian gate. No early promotion or early stop
    is made from this intermediate result.
128. A smoke-selection-aware formal merge is installed as
    `qm9_complete_total_local_scalar_activation_formal_analysis.py`. It validates the protocol
    hash, exact stable5 parent list, selected-arm set, unopened-parent count, and frozen
    validation/Test100 flags before merging. It reports Hessian and E/F curves, per-parent
    correction/oracle diagnostics, resource use, and artifact hashes; parent-CV authorization is
    emitted only when a formal arm passes every frozen gate. Static checks and the three related
    test files pass on node01. Script/test/wrapper SHA256 values are
    `51a0c83b346848f420d5cef0aec1f4f5f08cba0903506492413a3e0d8f2571e8`,
    `e29ec384ce91b6adbfe63054a2834660ede5f44da55a8208bda579a8629177f7`, and
    `5513352eb2f978a360e905f58ba0f629dfad29caa19fe741398f335382426aa3`.
    Slurm job 3753 is queued with `afterok:3751`; it will not run on a partial/failed formal array.
129. The optimizer audit confirms that J2/J3/J5 combine energy and force into one task before
    two-task PCGrad, despite the measured E-F and F-H cosines of `-0.958/-0.968`. An optional,
    default-off three-task E/F/H GradNorm plus deterministic PCGrad path is now implemented and
    covered by 29 passing related tests. Existing configurations retain the prior behavior, and
    no three-task experiment is submitted while 3751 is active. Implementation/trainer/test
    SHA256 values are
    `6aafee85a971812b6603c87370fdd8e355d23ae52ec6bb3330d7cc8b67c77bf0`,
    `61c06bd64981ded51f6954d617e519a71fb2b015faa7500e722eb853db538a0b`, and
    `61503aa48b40e91c543c6bb2419c9db8315bac7d2c40999824b24736a04fda3f`.
130. J5's step-300 record is still its best joint checkpoint: Hessian median/P90/max
    `0.8889/1.0262/1.0726`, E/F source ratios `0.7627/0.5912`, and asymmetry `2.72e-15`.
    By step 500 Hessian improves only to `0.8266/0.9529/1.0081`, while energy regresses to
    `1.854` times the source error, so the selection score is worse. J3 step 400 has
    `1.0673/1.1966/1.2676` and passing E/F ratios `0.5416/0.9369`, but its early Hessian descent
    has also slowed. Neither intermediate arm passes; both frozen 3000-step profiles continue.
131. By step 600 J5 refreshes its joint best to Hessian median/P90/max
    `0.7764/0.9266/0.9775`, E/F source ratios `0.4968/0.6802`, and asymmetry `3.15e-15`.
    J3 is `1.0162/1.1281/1.1921` with E/F `0.9333/0.7659`. Both are finite and preserve E/F at
    this checkpoint, but neither is close to the 5% gate.
132. A mechanistic successor is preregistered before the activation formal decision in
    `qm9_complete_total_hessian_local_scalar_three_task_pcgrad_v4.yaml`, SHA256
    `853a3b0c451d2973fbe2a0d9c58f77f30e530d8766c241a7979d4c778e795d62`. It changes only the
    optimizer task partition: energy, force, and Hessian receive separate force-referenced
    GradNorm scaling and deterministic three-task PCGrad. K3/K5 retain the corresponding v3
    architecture, initialization, targets, batches, learning rate, and gates. A 20-step smoke
    rule requires finite positive named task gradients and improvement over the paired v3 smoke,
    then selects at most one formal arm. The Slurm entry reads v3 formal analysis and refuses to
    run unless it certifies `parent_cv_design_authorized=false` and Test100 false/zero. No v4 job
    is submitted while 3751 is active. Smoke analysis, training wrapper, analysis wrapper, and
    test SHA256 values are
    `b23149a34069171d1c4184f30f4bf4dc9674f64af39961fd16058f59dad49b9c`,
    `d9892453b7fffcd1b7878c68865eaab590ba74c686897ef52bc76352ef6d3e25`,
    `b2f915981d86b31f24d625d0bad00b887cda61b245693eb65ae17a1b31b621ac`, and
    `577bb984426ad389bbaf88309e20f0bc5b602292da25ef0813d35eeefb116059`.
    Static checks and the related suite pass `25/25` on node01. Dependency job 3754 is queued
    after v3 merge 3753. Its gate-driven orchestration either exits if v3 passes or submits v4
    smoke, selection, at most one formal arm, and formal merge. It never submits parent-CV.
    Generic formal wrapper, formal-submit orchestrator, and contingency orchestrator SHA256 values
    are `2f61ebf1f98fde7649e4b82814d5786a03c96e6131bcda53d92060e57a5b6e3b`,
    `1c4f0c772d2b24a7926618c04180126071ee7c1ca8fd4383b9802f8bb0757a91`, and
    `f6aa4c8601e67247a03b800ff73cf60c1c50e43d050db9d16f9b8bd4132695e8`.
133. Stable5 formal reporting now includes mass-weighted vibrational postprocessing for the
    frozen original-A source and every smoke-selected formal arm. It reports frequency MAE/RMSE,
    imaginary-mode counts, and Hungarian-matched mode overlap after translation/rotation
    projection. Output now records reference manifest/hash and Test100 false/zero, and rejects
    any capacity or baseline input with a nonzero Test100 count. Node01 tests pass `8/8`.
    Evaluator/wrapper/test SHA256 values are
    `42ab89cf20e70f407808aaa7ae79a047e372c8900764cce8c1c7287fdd95afda`,
    `53541775e0626b3129568e98a193fef5c404cea30ac57f4f2fc835c24a31c662`, and
    `9f2b970c28bf1fcbdb959219318d7021d3a820a0ac450f8dc0af004ae78c4ef4`.
    Job 3755 is queued after v3 merge 3753; the v4 formal orchestrator submits the same audit
    after any selected v4 formal run.
134. Baseline-only real-data preflight 3756 completed in 6 s on exactly frozen stable5. The
    original-A mean frequency MAE/RMSE is `2203.3/3301.2 cm-1`, mean matched mode overlap is
    `0.5644`, and total model/PBE imaginary modes are `157/11`. This is the preregistered source
    comparator for formal candidates, not a candidate result. Summary/per-molecule/summary-CSV
    SHA256 values are
    `fdb2de3820fc0eb73a65949d231b7fdc959757336629b5d3f4c01c6517a9e261`,
    `90b6ae2f42535891a0502d66c517371ec2d5b5edc8d9a6839db0bdd47beaf5ee`, and
    `43673dc1bf88160678798b9ea04ce0bcb745cbfe6d9acb5b088b225b1f552382`.
135. Latest v3 records remain non-passing: J5 step 900 is Hessian median/P90/max
    `0.7112/0.8603/0.9301` with E/F `0.3072/0.7124`; J3 step 800 is
    `1.0000/1.0901/1.1425` with E/F `0.6717/0.7353`.
136. A pre-launch audit found that naive separate-task GradNorm cancels a multiplicative Hessian
    warm-up. v4 now explicitly targets `warmup * ||g_force||` for the Hessian task while energy
    and force target `||g_force||`; v3 is unchanged. Trainer/protocol/test SHA256 values are
    `61c06bd64981ded51f6954d617e519a71fb2b015faa7500e722eb853db538a0b`,
    `853a3b0c451d2973fbe2a0d9c58f77f30e530d8766c241a7979d4c778e795d62`, and
    `b0decdb61ce93c1b589f588d2b29dbb462a7c258af7b771b4b857f9270533d70`; related node01 tests
    pass `25/25`.
137. Intermediate J5 best-step correction diagnostics show why apparently good correction
    alignment is insufficient. Median correction-target cosine is `0.9742`, correction/target
    norm ratio `0.9601`, and oracle scalar amplitude `1.0103`, yet oracle-amplitude Hessian error
    remains `0.6926`. The target correction norm is median `2.7268` times the PBE Hessian norm;
    even with optimal amplitude, relative error `0.05` requires median cosine at least
    `0.999832`. All five parents remain high (`0.584--0.924` at J5 step 1000), so this is a
    shared high-precision response-direction problem rather than one outlier. J3 step 900 is also
    plateauing at median/max `0.9933/1.1137` with E/F gates intact.
138. Preflight also found that the shared trainer's protocol-stage allowlist omitted the new v4
    stage and would have rejected the automatic smoke at startup. The v4 stage is now explicitly
    accepted and covered by a real hash-checked protocol-load test. Current trainer/test SHA256
    values are `61c06bd64981ded51f6954d617e519a71fb2b015faa7500e722eb853db538a0b` and
    `b0decdb61ce93c1b589f588d2b29dbb462a7c258af7b771b4b857f9270533d70`; focused checks pass
    `7/7`.
139. A deterministic full-stable5 LBFGS ceiling runner is implemented but not submitted. It
    restores a hash-checked scalar checkpoint and supports two explicitly separated objectives:
    joint E/F/H can authorize parent-CV only by passing every existing gate; Hessian-only can
    never authorize parent-CV and exists solely to distinguish optimizer/representation capacity
    from multitask conflict. It uses full-batch closure evaluations, strong-Wolfe line search,
    best/last checkpoints, per-parent arrays, resource accounting, and Test100 false/zero
    provenance. Runner/wrapper/test SHA256 values are
    `d5438cb4fee290b1de6dd80fd6978bd980f0934eeecb82bac8e4e00be95e4cd3`,
    `2957aa8bdd896e34349e6a43c11d514642f9502a5cf0393e9d0e2e7e8393846e`, and
    `366ac3567243c8c74c2282826e47c83910bf27089825cfe77281b0d7d27167e1`; tests pass `3/3`.
    The source checkpoint and a separate run protocol must be frozen only after v4's decision.
140. A one-iteration runtime smoke 3757 used an immutable J5 step-1100 checkpoint snapshot and
    exactly frozen stable5. It completed in 72 scheduler seconds with three strong-Wolfe closure
    calls, `4769.9 MiB` peak GPU memory, and `1599.1 MiB` MaxRSS; no NaN/OOM occurred. Joint score
    moved `1.600303 -> 1.600232`, Hessian median `0.686959 -> 0.686927`, E/F source ratios ended
    `0.1767/0.7138`, and parent-CV remained false. This validates runtime only, not LBFGS efficacy
    or promotion. Snapshot/summary/metrics/checkpoint/per-parent SHA256 values are
    `ad0a8ce0c2c6d5b4f1e25cdd2bd491a798d8c0ff1fff2408645aef3af5d68a98`,
    `2cf6faf392e1fb03f42c427039d288cc3efdc1e802a1dcaf4fd7b66b5a1280f0`,
    `4fa7634f02cc699516175e7f19d8d666db2ae8e251d5a3caba395e4d96426ecd`,
    `d6c7b678d4d1fe290fe1d0c912794cf1f4169bbdef6acb5eb055cd976a1e934a`, and
    `9c7a7aec14ed280d726b353e17e31883f7bdc0bdaf35215e9db0c86d288d290d`.
    A 100-iteration run is roughly estimated at `1.1--1.4 h`, but remains forbidden before v4
    decision/freeze. J5 v3 step 1200 is still non-passing at Hessian median/max
    `0.6760/0.8963` with E/F `0.2388/0.7202`.
141. v4's paired-v3 smoke thresholds were checked against the machine-generated v3 smoke
    analysis. The original hand transcription differed by `3--5e-7`; the protocol now embeds the
    exact machine values `5.963815683324032` and `5.460310807421687`, and equality is verified
    programmatically. The superseding v4 protocol SHA256 is
    `853a3b0c451d2973fbe2a0d9c58f77f30e530d8766c241a7979d4c778e795d62`.
142. A stable5-only scalar-jet compatibility audit now separates label/definition consistency
    from representation and optimizer failure. It enforces the local rigid-invariance identities
    `H*T=0` and `H*(Omega*R)=-Omega*F`, projects the supplied force onto the invariant tangent
    space, and computes the nearest compatible symmetric Hessian both with that force fixed and
    with the invariant force free. For the PBE-minus-original-A correction, the exact-force
    relative-Frobenius floor has median/P90/max
    `0.0004212/0.0004609/0.0004776`; the free-force floor is
    `0.0003782/0.0004340/0.0004543`. Correction net-force/torque defects are only
    `7.64e-5/1.02e-5` at the median. Thus the unanchored 5% gate has over two orders of magnitude
    of mathematical margin: the current `~0.66` training floor is not caused by force/Hessian
    sign, units, rigid-motion incompatibility, or an impossible scalar jet. The audit opened only
    the frozen stable5 artifacts and records validation/Test100 false/zero. Script, test, summary,
    and per-parent CSV SHA256 values are
    `e860ef12d80eed01765af6c0ef20b3c44a32b6a7b471fed2d13c5107405bb133`,
    `01060803abbdd5e6e0bff211e825fc7ad26eba506bd30027f1595e4b15ad9777`,
    `2be7bb4dccfce012fb9a3aaed0a97341be23171d3064d5f5c48ff53b25a2a23b`, and
    `01d5515c6059ea4516c51f94eb44a04910a326c5f87ccfd624dd0f61362e27d1`;
    focused tests pass `2/2`. J5 step 1500 is still non-passing at Hessian median/P90/max
    `0.6571/0.7853/0.8673` with E/F ratios `0.3673/0.7086`; J3 step 1300 is
    `0.9674/1.0365/1.0809` with energy ratio `1.3738`.
143. A matrix-free parameter-to-Hessian Jacobian range audit is implemented for the next
    fail-closed diagnosis. It uses exact `J^T*u` from the scalar's third-order parameter graph and
    a centered float64 parameter-space `J*v`, then CGLS-projects the complete stable5 Hessian
    correction onto the local Jacobian range. It records per-parent linearized floors, nonlinear
    trust-region probes, finite-difference stability, E/F gates, resources, opened paths, and
    Test100 false/zero. Real-data runtime smoke 3761 used the immutable J5 step-1100 snapshot and
    one CGLS iteration: wall time `70.9 s`, peak GPU `4.76 GiB`, MaxRSS `1.45 GiB`; shrinking the
    parameter FD step threefold changed `J*v` by only `7.67e-8` relative with cosine
    `0.9999999999999998`. One iteration changes median Hessian error only
    `0.68696 -> 0.68528` and is not a rank conclusion. A 30-iteration audit is reserved for the
    checkpoint frozen after the v4 decision. Script/wrapper/test and clean smoke summary/metrics/
    resource SHA256 values are
    `9f58594a01daf56dcd3e2bca1dbb83e2da40fded55f8c8bba2a93c1e02fb95b7`,
    `4a86b7d7ddaa8944f74489b453ef44f29703aca193625811aa05e1bff9356be5`,
    `c23d16649ccd0bfa9f75d4a0d30918eacbedd3929892ae6072659a0a677ba2ff`,
    `bfdc7e0886f4d25cef7309dd576b935a090af7212d84ce14e4085f14a4d16c1b`,
    `d9ada41db91b1a28c4cc518157a686bf113265b627ee3b8b973441b60e06ae33`, and
    `6867ee9d7887d30341f555f66c11b984de641a30cb56007ea1c139d92a7a580d`.
    Focused Jacobian tests pass `3/3` and the combined jet/Jacobian suite passes `5/5`.
144. The post-v4 failure path is now hash-frozen and automatic, while remaining fail-closed for
    data expansion. After the v4 formal merge, a dependent job exits if v4 passes. If it fails,
    it writes an immutable diagnostic manifest binding the formal summary, unique failed arm,
    best checkpoint and step, exact stable5 parents, source hashes, 30-step Jacobian settings,
    and both 100-iteration LBFGS objectives. The LBFGS runner now requires this manifest and
    rejects source or optimizer-parameter drift; Hessian-only remains permanently unable to
    authorize parent-CV. The three diagnostics run in parallel on node01, but no parent-CV,
    validation, train100, or Test100 job is submitted. Freeze script, LBFGS runner/wrapper,
    post-v4 wrapper, updated formal orchestrator, and two test-file SHA256 values are
    `a643f20c826ec49db99fd73244bd31c5b92d31bc105b804ed6b5b9db47ad299c`,
    `e44827f6b7943dac78ec371612f1159c70bae1226d7182cd049e9490269458c7`,
    `4f346cf706b073e853ed29f6dafe7c5de5ae1cde2495267dfe04f0d20328bda0`,
    `1cc4d9e27857c1ddaa4f2e2d3088bf0e9c65e8f5c463011bb2a8a21932509378`,
    `f3b4b92acd728c09be2ff258f800bdf0ef624b027f22fb357cbe43793f7588c2`,
    `e5cc3aa7aedee3abd51b165f48aa82df623303302195cd5a4e1b5e455f5fbfa1`, and
    `95b00aebdb6fcee27f26e2a49e377e05db3e13c132b260016f00e57d3897af91`.
    The combined freeze/LBFGS/Jacobian tests pass `9/9`; all entry points pass static and shell
    checks. The already queued contingency job 3754 invokes the current nested orchestrator only
    after v3 analysis, so this frozen failure path will be used without altering active v3.
145. The three post-v4 diagnostics now have a dependent read-only merge. It revalidates every
    source path/hash, manifest hash, arm, stable5 parent, numeric setting, and Test100 flag, then
    plots CGLS residuals, joint/Hessian-only LBFGS Hessian curves, and joint E/F gates. The
    machine decision distinguishes a full joint pass, Hessian-only representation pass with
    multitask failure, linearized-Jacobian pass with nonlinear optimizer failure, and no 5%
    evidence under the frozen budget. Parent-CV authorization can be true only for a complete
    joint gate pass, and the merge records `parent_cv_submitted=false`. Analyzer/wrapper/updated
    submitter/test SHA256 values are
    `c1bee20b863011670ef005480105e0a2c3b5ae8e4984c05a3e9793a2ef0d80fa`,
    `985fdd9b57ffa9952e13b6b663e4e179b2a953fd1061e3e2169293803f4a1c53`,
    `f7036c84afd839ebe66fcd8e7007adc4b2db9c335befb77c072bee9aac0c1ea7`, and
    `10b8659bcd3bd9805e08c0fe592952818a8dc30782c4df5985f46437a586b418`.
    The integrated freeze/LBFGS/Jacobian/merge suite passes `10/10`. J5 v3 step 2000 reaches a
    new but non-passing best at Hessian median/P90/max `0.5935/0.7499/0.8340` with E/F
    `0.3163/0.7204`; J3 remains much worse.
146. The no-eligible-v4-formal branch is also covered. If v4 smoke selects no arm, the formal
    orchestrator now submits the same frozen diagnostics using the best failed v3 formal arm,
    selected by the already preregistered joint score
    `median_H + max_H + max(E_ratio-1.05,0) + max(F_ratio-1.05,0)`. The manifest records this
    source-selection rule and still binds the exact source protocol/run/checkpoint hashes. Source
    root, analysis, protocol, diagnostic root, and optional explicit arm are configurable only
    through the hash-validating freeze path. Updated freeze/post/merge-wrapper/formal-orchestrator/
    test SHA256 values are
    `37515e928bab172aa59b0bd59e65ece237eed09d870d6748ccbdb9fabbc4af72`,
    `5cd6174eaabb52a626bce94ca47799aaaca4da716840bbed854fdc302bb0d2f5`,
    `2432e271cbda952beecc0c0536f4c5bc098b23a15788ce081cc3ae3e567fa72a`,
    `c500f15da12d7345f0be1d5161fc4302a587cdc4a1dc06ff602e3046fb67f612`, and
    `d81c08149a593ee9572f56126b1f4cd917f1e3d4463fa8ffb1ce2efcef6856fd`.
    Both one-arm and multi-failed-arm freezes are covered; the integrated suite passes `11/11`.
147. Stable5 promotion is now closed over the requested vibrational targets as well as matrix and
    E/F gates. The frozen post-v4 manifest requires frequency MAE `<=200 cm-1`, absolute total
    imaginary-mode-count error `<=5`, and mean mode overlap `>=0.8` on all five parents. A new
    node01-only postprocessor evaluates original A, joint LBFGS, and Hessian-only LBFGS from their
    scalar Hessian artifacts; the final read-only merge refuses parent-CV authorization unless the
    joint candidate passes Hessian median/max `<=0.05`, E/F ratios `<=1.05`, asymmetry `<=0.005`,
    and all three vibration gates. A matrix/E/F pass with failed vibration is classified
    separately, Hessian-only remains diagnostic, and every branch records
    `parent_cv_submitted=false`, validation/Test100 false/zero. Updated freeze/analyzer/vibration
    wrapper/post wrapper/merge wrapper SHA256 values are
    `165bbfabab6400d08ce2be07d824166565897ebab30307fddb16709a834fa2b2`,
    `75f37111650c893520f02e28c97ffbaefa315d3d10a6cbb6fcb28de85ae64e0c`,
    `9b9658046d3fdd63a41f5779a41b59f8da620b4bf56964b7fc41b0687eb64c78`,
    `188520e95f1dd8b4edf85199a51b8ee0fc486c64a1458fd99b61fcb9aa43a870`, and
    `919b8abeeffd7016729eee4ab94ac71af17b3ec2f48f7660b271d769d5779b4f`.
    The expanded freeze/merge/LBFGS/Jacobian/vibration/three-task/complete-total suite passes
    `44/44`; Python static checks and all changed shell entry points pass. The original-A stable5
    reference is far from
    the new gate: frequency MAE/RMSE `2203.3/3301.2 cm-1`, mean overlap `0.5644`, and model/PBE
    imaginary-mode totals `157/11`. J5 step 2100 lowers Hessian median/P90/max to
    `0.5868/0.7385/0.8205`, but its energy ratio is `1.3407`; step 2000 remains the best joint
    checkpoint at `0.5935/0.7499/0.8340` and E/F `0.3163/0.7204`. Neither is close to promotion.
148. Formal activation audit v3 is complete and rejected. J5 tanh selects step 3000 at Hessian
    median/P90/max `0.513559/0.701311/0.777794`, E/F source ratios `0.5740/0.6885`, and
    machine-precision symmetry. J3 beta-5 Softplus selects step 2900 at
    `0.778141/0.944311/0.991500` with E/F `0.5622/0.6807`. J5 improves the original-A median by
    more than fivefold, but remains over ten times above the 5% fit gate. Its frequency MAE/RMSE
    are `537.79/786.46 cm-1`, mean mode overlap `0.5680`, and model/PBE imaginary-mode totals
    `53/11`; J3 is worse at `657.62/998.20 cm-1`, `0.5649`, and `57/11`. Formal analysis and
    vibration summary SHA256 values are
    `c7f0eb8fd06d0c4a7118874f33278aab12e2eec6b8c2383f0c3c7b2cba100595` and
    `33e8f7e15669cb4d00800ec45f15550788f45599fc81b7b4f8f107ef26a4df28`.
    Three-task v4 smoke then rejected both K3/K5 arms: their 20-step medians
    `2.7284/2.7913` do not improve step zero or their paired v3 smoke scores, so no v4 formal was
    launched. The smoke selection SHA256 is
    `7a256a9f001aff8c115df5c7d29436ea1ca020cbd7efac93b4f24da467e4ed40`.
149. The automatic no-formal diagnostic chain completed jobs 3766--3771 and identifies a local
    second-order representation/Jacobian bottleneck. From frozen J5 step 3000, 30 matrix-free
    CGLS iterations reduce the linearized Hessian median only `0.5136 -> 0.4778`; the global
    residual remains `0.9399` of its initial norm. Parameter-FD stability is excellent
    (`5.53e-8` relative, cosine within `1.3e-15` of one), so this is not differencing noise.
    Joint LBFGS makes 205 closure calls but selects iteration zero unchanged. Hessian-only LBFGS
    reaches median/P90/max `0.4394/0.6180/0.6788`, while energy ratio deteriorates to `3.5847`;
    its frequency MAE, overlap and imaginary totals remain `536.31 cm-1`, `0.5519`, and `58/11`.
    The machine diagnosis is
    `no_five_percent_capacity_evidence_with_preregistered_local_diagnostics`; parent-CV remains
    unauthorized and unsubmitted. Manifest/Jacobian/joint-LBFGS/Hessian-LBFGS/vibration/final
    analysis SHA256 values are
    `0e03b754ef2006496b89bdb0da2d616b2e54d8e832a090db87d385f8b82f0022`,
    `1be484f355f373424f67862fccd5f87ffa0b8e7c7746ce39d017f28910912d85`,
    `7f90311cf3930ce7a25c913e23f82dfef725ca3a951fe9caf6db06842b48f17c`,
    `ecace4785fd746807c3d9013a3814f7da48a37616c8230cbad29cce4e3fc9b72`,
    `7b5ccc802b723fda4b6777364b3decbfd8699fc5ab685560fd597430641cdda9`, and
    `152c3ed97a47a27c365f3579677a8c8af2b52673e8b3d9cf1ee7e078df6e2ad4`.
    Validation and Test100 remain false/zero throughout.
150. The v4 smoke analyzer previously included forward-only step-zero NaN placeholders in its
    finite-diagnostic check. Eligibility now checks finiteness only for positive update steps;
    step zero remains the required untouched Hessian baseline. A regression test verifies that
    expected step-zero NaNs are accepted while any positive-step NaN is rejected. This does not
    change the historical v4 decision because both arms independently failed the score and
    Hessian-improvement requirements.
151. The first distinct-response replacement is a local-additive explicit angular scalar. It sums
    element-resolved radial RBF features and neighbor-element-pair-resolved radial outer products
    times Legendre angular functions, then applies a smooth atom-wise scalar network. Rigid
    motions and same-element permutations leave the energy invariant; force and Hessian are exact
    scalar autograd derivatives. Focused zero-initialization, invariance, finite-Hessian, symmetry,
    and parameter-gradient tests pass. Stable5 smoke jobs 3772/3773 remain finite and use only
    `6.71 GiB`, but the 20-step Hessian median changes only `2.7268 -> 2.7169` and the maximum is
    `3.3996`; formal training is therefore not authorized. Metrics/summary SHA256 values are
    `823a6c2bad2bc5b84b57a23626903ea28e05bc933c4574663f19a74f6fc35f39` and
    `4550be4b4e8183724f8321bbaa1d4b0e19a85c7df56568eb4758526bb3a5e9b9`.
152. A column-normalized linear E/F/full-Hessian ceiling now separates angular feature-span
    capacity from neural optimization. The first implementation used nested unchunked
    `jacrev(jacrev)` and job 3774 failed before its first parent when PyTorch attempted an extra
    `72.69 GiB` standard-basis allocation. The preserved v2 implementation computes exact
    feature Hessians by feature-chunked `jacfwd(jacrev)`. A synthetic regression matches reverse-
    over-reverse values, Jacobians, and Hessians to `1e-12`; the angular/model suite passes `6/6`.
153. Chunked job 3775 completes in `1:28.21`, with `14.50 GiB` peak allocated GPU memory and
    `1.32 GiB` MaxRSS. Its 1907 active columns reduce stable5 Hessian relative-Frobenius
    median/P90/max to `0.2780/0.3028/0.3096`. Energy median absolute error is
    `1.48e-6 Ha`, force median MAE is `6.77e-4 Ha/Bohr`, scalar Hessian symmetry passes, and the
    result records validation/Test100 false/zero. Thus the optimizer is not the active blocker for
    this audit, but a three-body radial/angular linear span still misses the 5% gate on every
    decisive aggregate. The next fit-only ceiling must add a parity-even bonded four-body/torsion
    scalar; no formal angular-only training or data expansion is authorized. Summary/checkpoint
    SHA256 values are `95ca2087ac57cce357d0d2ca6848a68329d9e063a6373be11ddd07d5c7d65f24`
    and `7acb6dd994c8fc14e0276a11b948b248e98b4f93de3472969580671010dc822f`.
154. The frozen v6 parity-even bonded four-body contingency completes in job 3776. Adding 10,560
    torsion RBF/cosine-multiple columns to the 2,195 angular columns improves stable5 Hessian
    relative-Frobenius median/P90/max from `0.2780/0.3028/0.3096` to
    `0.1872/0.2096/0.2175`. Energy median absolute error is `2.17e-5 Ha`, force median MAE is
    `5.51e-4 Ha/Bohr`, and maximum scalar asymmetry is `1.21e-13`. The run takes `56:15`, peaks
    at `16.82 GiB` allocated GPU memory, and preserves validation/Test100 false/zero. It does not
    yet establish the representation floor because ridge-CGLS hit its frozen 2,000-iteration cap
    at relative normal residual `1.73e-4` while the data residual was still decreasing. A
    fail-closed v6b rerun keeps every scientific setting fixed, uses a larger exact-autograd
    feature chunk only to reduce repeated work, and extends CGLS to numerical convergence before
    deciding whether central-element-resolved angular or higher-body features are required.
    Summary/checkpoint SHA256 values are
    `8e32db5469b6e068dc652cf854d14ad59293901655697b6f6ca4d8a5d39f7b43` and
    `477803a4a0ba013f6a1e9a731f14ffe066473d525be1de44543c3b1a8fb03905`.
155. Solver-continuation job 3777 confirms that the 2,000-step truncation mattered but was not the
    full capacity blocker. At 20,000 CGLS iterations, Hessian median/P90/max improves to
    `0.1475/0.1601/0.1684`, energy median error to `6.07e-7 Ha`, and force median MAE to
    `1.98e-4 Ha/Bohr`; scalar symmetry remains at `8.0e-13` or better. The larger exact-autograd
    chunk reduces wall time from `56:15` to `22:01`, but peaks at `57.22 GiB` allocated memory.
    CGLS still does not meet its normal-residual tolerance: final relative normal residual is
    `2.55e-5`, weighted data residual is `0.02903`, and coefficient norm is `3.09e5`. Blindly
    adding more CGLS iterations is therefore replaced by a same-ridge float64 Gram/Cholesky solve;
    only after that exact regularized solution may the remaining error be assigned to the feature
    span. Summary/checkpoint SHA256 values are
    `99b89b94f1389adb82be8d41da34d99f1a56cbb38cdd863d74b6b1e1e7cc103d` and
    `fd5e818fce45ba88976dd7cd40b6fbff296b3bc00542f4f28ac20ab6f0714681`.
156. Exact same-ridge job 3778 closes the solver question. Float64 Gram/Cholesky succeeds with
    `info=0` and relative normal residual `2.73e-13`. Stable5 Hessian median/P90/max is
    `0.1361/0.1487/0.1557`; energy median absolute error is `4.43e-10 Ha`, force median MAE is
    `1.48e-4 Ha/Bohr`, and maximum scalar asymmetry is `1.29e-12`. The exact solution therefore
    proves that the remaining 13.6% median error belongs to the current feature span, not CGLS,
    density, E/F tradeoff, symmetry, or numerical differentiation. Job wall time is `39:10.68`,
    allocated GPU peak is `30.29 GiB`, and validation/Test100 remain false/zero. The next frozen
    factor is central-element resolution of all radial/angular environment features; four-body
    features, matrix weights, ridge, exact solver, and stable5 remain fixed. Summary/checkpoint
    SHA256 values are `9bb2e16b3c01e1be8e4b9ed8d56b6a97494f60d72f8258bf3861a06172bad075`
    and `42edb6a52e1fc4a2c50578ecfc6346dd331cab4251e61bf0ccdbe63b59fc5fa3`.
157. Central-element-resolved exact job 3779 completes in `1:10:45`. Resolving each radial/angular
    environment sum by central element increases the global scalar basis from 12,755 to 21,515
    columns (18,591 active) and lowers stable5 Hessian relative-Frobenius median/P90/max from
    `0.1361/0.1487/0.1557` to `0.07010/0.09704/0.09825`. Only `0132419` passes 5%, at
    `0.03423`; the other four remain between `0.06852` and `0.09825`. Energy median absolute
    error is `2.20e-10 Ha`, force median MAE is `3.65e-5 Ha/Bohr`, maximum scalar asymmetry ratio
    is `6.42e-12`, and the exact normal residual is `2.19e-13`. Therefore v7 is a real
    representation improvement but fails both frozen Hessian gates; its fail-closed vibration
    entry point refuses to run, and no direction/parent expansion is authorized. Allocated GPU
    peak is `31.75 GiB`, MaxRSS is `1.32 GiB`, and validation/Test100 remain false/zero.
    Summary/checkpoint SHA256 values are
    `14688f9158c4b11127ba7bc7ba85e96e78afc698eec605c8e691819c2333c9b0` and
    `f9c122ec545961c8de5d4e728fcc8d2650b3c1b57b8dd04673c4d3d22356cdff`.
158. A final same-family basis-resolution audit is preregistered as v8 before moving to a genuinely
    nonlinear/equivariant local scalar. It changes only `radial_size 6 -> 7` and
    `angular_order 3 -> 4`; central typing, four-body features, stable5 parents, E/F/H matrix,
    ridge, exact float64 Gram/Cholesky solve, and gates remain fixed. The resulting global basis
    has 29,115 columns. Remote focused tests pass `4/4`. Initial job 3780 used a 24-feature
    derivative chunk and was deliberately cancelled after `32:25` before completing the first
    parent: each chunk recomputes the full 3,715-feature atomic map, so that numerical setting
    would exceed its time limit. Its empty progress/resource files are preserved. The hash-bound
    rerun changes only the exact-autograd chunk to 64, writes a new v2 output directory, and keeps
    all scientific settings fixed.
159. v8 required two memory-only recoveries, both preserving failed artifacts. Job 3781 computed
    four parents and then OOMed on the 21-atom parent because historical jets/designs remained on
    GPU. CPU offload plus hash-bound per-parent jet caches in job 3782 again reached four parents,
    then exposed a within-parent chunk64 fragmentation request. Final job 3784 reads those four
    caches in `4.4 s`, uses chunk48 plus expandable CUDA segments only for the remaining parent,
    and completes the identical exact solve. Stable5 Hessian median/P90/max is
    `0.05435/0.08675/0.08833`; only `0132419` passes 5%. Exact normal residual is `1.73e-13`,
    energy median error is `1.23e-10 Ha`, force median MAE is `1.21e-5 Ha/Bohr`, and symmetry
    remains numerical. Thus same-family radial/angular basis refinement is stopped. The completed
    summary/checkpoint SHA256 values are
    `c676a08d880eadc2521771cccd4055cd3b86d50eac315c92ac03aa624206570a` and
    `b08b1122d6fde179b289d0ed0a1e33938f0e99509b709e116d307d057da17535`.
160. v9 introduces the required genuinely nonlinear local scalar response. It appends, for each
    central element, 1,536 deterministic multiscale `tanh(W x_i+b)` features of the invariant
    atomic environment to the fixed 29,115-column v8 basis. Output coefficients remain the only
    fitted quantities, so E/F/H are exact derivatives of one scalar. Remote invariance,
    finite-Hessian, symmetry, protocol-stage, and solver tests pass `9/9`. Job 3785 failed closed
    before computation on the initially unregistered protocol stage; the loader was then extended
    explicitly. Chain-submitted job 3786 is the authoritative run. A mistakenly duplicated job
    3787 was cancelled at `1:34`; deterministic cache writes were subsequently hash/finite audited.
    Job 3786 solves the 27,299 by 36,795 matrix exactly in `4:03`, with `31,718` active columns and
    relative normal residual `1.14e-13`. Stable5 Hessian median/P90/max is
    `0.01174/0.02413/0.02833`; all five parents pass 5%. Energy median error is
    `4.89e-11 Ha`, force median MAE is `4.13e-6 Ha/Bohr`, and maximum asymmetry ratio is
    `1.77e-11`. Summary/checkpoint SHA256 values are
    `e4786a1cda781df6a3a6c96e1706586c75c3fb90ea8e5a1cda07af470ff6ee1e` and
    `993014374f8eedf19d708f977a6bd019e5add43bc38e5295d07f007290dc497e`.
161. Protected vibration job 3788 closes every preregistered stable5 gate. v9 mean frequency
    MAE/RMSE is `8.19/23.40 cm-1`, mean matched mode overlap is `0.9401`, and model/PBE imaginary
    totals are `14/11`, an absolute count error of three. The matrix, E/F, symmetry, frequency,
    overlap, and imaginary gates therefore all pass on the five fitted parents. This proves a
    small-sample nonlinear local-scalar capacity ceiling, not unseen-direction or unseen-parent
    generalization. The next authorized scope is the frozen train20 direction split only;
    independent validation parents and Test100 remain unread. Vibrational summary SHA256 is
    `f449a98671f9bb073f26c0793bebadef1882121387a959cb3726f8dd588e3a71`.
162. The frozen train20 Stage-2 run is complete without reading validation or Test100. Preflight
    job 3789 validates all 20 train parents, the `17 train + 7 heldout` directions per parent, and
    a 38,571-column shared schema. Feature-jet array 3790 completes `20/20` exact float64 scalar
    feature Hessians on node01. These feature derivatives are label-free; only the 17 registered
    train-direction PBE HVPs enter the fit. Job 3810 then solves the 35,930 by 38,571 design in
    `82.2 s`, using `22.1 GiB` MaxRSS and `28.0 GiB` peak GPU memory. The normal-equation residual
    is `1.49e-13`, so the result is numerically solved rather than optimizer-limited.
163. v9 fails unseen-direction generalization decisively. Train-direction HVP relative-Frobenius
    median/max is `0.00952/0.03548`, but frozen held-direction median/P90/max is
    `1.7439/9.5859/27.3257`; full-Hessian median/P90/max is
    `1.1407/4.5739/10.1497`, with `0/20` parents below `0.20`. Energy and force are nearly
    interpolated and scalar Hessian symmetry remains numerical, so neither E/F loss nor
    nonconservative derivatives explain the failure. The held/train median ratio of `183.2`,
    together with a `1.83e10` coefficient norm, identifies severe curvature-subspace overfit.
    Summary/checkpoint SHA256 values are
    `385936bc64dee020656e3f85625bc7a0347bf5831cc3250b97e3123b424fe073` and
    `431973359ffb9e28ff5b2942cce0c0f06184fc44b955bb0c51628132aafd0ef5`.
164. A preregistered diagnostic-only ridge path in jobs 3811/3820 tests `1e-7` through `1` while
    preserving the same train/held split. Job 3819 is a preserved two-second analysis-only failure
    caused by scientific-notation directory spelling; the fixed analyzer discovers runs from their
    internal `effective_ridge` and passes three regression tests. Ridge `1e-3` is best by the frozen
    held median: it reduces held median/P90 to `0.5228/0.8794` and full-Hessian median/P90/max to
    `0.3648/0.5379/1.1593`, while shrinking normalized solution norm from `519.6` to `21.7`.
    However train HVP median/max regresses to `0.1647/0.3711`, and every ridge retains `0/20`
    full Hessians below `0.15`. Thus ill-conditioning amplifies the original failure, but ridge
    alone cannot supply the missing transferable curvature response. The path is explicitly
    non-promotable. Analysis summary/ridge CSV SHA256 values are
    `4334131fa321fa8da1b9c3250cf4995f8ace0f1f2cb74f32c9442b1c09ae2164` and
    `a3ab5d41f68106eb7d9bfd6fb174b5811029c0e6d53f8425f4d594cc41577fe5`.
165. A second one-shot diagnostic in jobs 3821/3827 holds ridge at `1e-8` and reduces each of the
    three random-feature scale blocks from width 512 to `0/16/32/64/128/256`, retaining every
    explicit angular/four-body feature. Width 16 is best by held median but reaches only held
    median/P90 `0.984/9.530` and full-Hessian median/P90/max `0.876/4.127/10.934`; the no-random
    v8 subset has full median `0.844`. Every arm has `0/20 <=0.15`. Therefore random-feature width
    alone is not the missing inductive bias. The narrower solves lower GPU peak from 28.0 GiB to
    17--22 GiB, but do not repair curvature. Analysis summary/CSV SHA256 values are
    `089fcf984507c83a386b59bbcf9edae4738523777403a7507fb650a1757e0bfa` and
    `d31f78a8bc1c44a9677cbc0fe17bae9dff848c5013e744c57ff873a8515a601a`.
166. Jobs 3828/3831 keep the original seven held directions fixed and add geometry-only,
    deterministic random internal train directions. Training counts 24, 32, and the maximal
    internal complement all improve some matrices, proving that the original 17 directions
    underconstrain curvature. Train32 gives held median `0.721` and full median `0.332`; maximal
    coverage gives full median/P90/max `0.395/2.497/12.144` and `7/20 <=0.15`. Yet train HVP
    median/max itself is `0.0632/0.0937` in the maximal arm, so the frozen feature span cannot fit
    all added directions at the 5% gate. Hard tails such as `0028399`, `0124009`, `0096630`, and
    `0015047` remain physical absolute-error failures rather than only small-denominator relative
    errors. Analysis summary/CSV SHA256 values are
    `6c1701c52a62b34a39138130934227360c1fda25dbd1177b7528c2ef4c4a2869` and
    `360e1ce87c49cf5955493842b03915f90c8972a0cf29575283fe3bb4e99a06c9`.
167. Full-Hessian fit-only job 3832 consumes every matrix element for all 20 train parents and is
    therefore an upper bound, not unseen-direction evidence. It solves a 107,498 by 38,571 design
    with `1.96e-13` normal residual in `2:40`, using 64.7 GiB MaxRSS and 28.0 GiB peak GPU memory.
    Full-Hessian relative-Frobenius median/P90/max is `0.06683/0.08732/0.10512`:
    `20/20 <=0.15` and `19/20 <=0.10`. E/F and scalar symmetry remain excellent. This proves that
    v9 has about 7% complete-matrix fit capacity on train20, while the partial-direction protocol
    lacks enough shared curvature structure to recover unseen directions. It narrowly misses the
    stricter 5% capacity target, so the next representation should increase structured shared
    response capacity rather than add another unstructured linear random-feature block. Summary,
    checkpoint, and per-parent CSV SHA256 values are
    `aa5fb1e462e20bf278ee01c0625011461e8a3acbd587c06bba1a5bdb13957cf0`,
    `9809777c786ef51c6f2d56751bddd70988bc07a69e3f170ae67951b0af8233c9`, and
    `111da28f9984c7313c1ecccef29ab371d725236ed05070501a20c5323a6bd0cb`.
168. Label-independent multi-hash CountSketch jobs 3833--3839 test whether dimensionality alone
    explains the direction-overfit gap. Six subspace sizes from 1,024 to 16,384, each with two
    signed hashes per original scalar feature, are fit only to the same registered E/F and 17
    train-direction rows. All six jobs complete in `2:21--2:39` with 12--16.6 GiB MaxRSS. The best
    arm by the already exposed held-direction median is dimension 12,288: train HVP median/max is
    `0.07699/0.10059`, held median/P90/max is `0.63630/7.40335/62.437`, and full-Hessian
    median/P90/max is `0.50547/3.65303/23.1594`, with `0/20 <=0.15`. Dimension 4,096 gives a less
    pathological but still failed full median/P90 of `0.750/0.934`; dimension 16,384 gives
    `0.675/1.988`. Energy/force remain preserved, so the failure is specifically curvature
    transfer. The random projection is rejected: generic compression destroys target-aligned
    curvature capacity and does not provide the missing inductive bias. Summary, aggregate CSV,
    and direction CSV SHA256 values are
    `bb1745c52711d47323f3277ac231549f9653d0f8ffd3716ce96a271e831c2d2c`,
    `baa88be421302001340b8beb34d0a2f543563811f4a7495c7dacc5f9d66e149c`, and
    `905db9dae7ae620c89661107e824188709736e48db1597a50059d96f2359be2c`.
    The next train20-only experiment must tie coefficients using explicit element, radial,
    angular-order, and torsion metadata. It must not be another random subspace or access
    independent validation/Test100.
169. Chemistry-aware structured scalar-subspace jobs 3840/3841 replace arbitrary hashing with
    typed DCT coefficient fields over central/neighbor elements, radial indices, angular order,
    four-body torsion keys, and frozen random-kernel projection metadata. All five arms complete
    on node01 in `97--103 s` of evaluator wall time with 1,869--13,673 parameters, 10.7--11.9 GiB
    Slurm MaxRSS, and no NaN/OOM. Capacity grows monotonically on train directions, but no arm
    passes. The smallest `a1_f1_r1` has train/held/full medians
    `1.186/1.516/1.449`. The best arm by the already exposed held median, `a3_f2_r2`, has 9,797
    parameters and train median/max `0.2225/0.3797`, held median/P90/max
    `0.6147/1.7587/6.1424`, and full-Hessian median/P90/max
    `0.4787/1.1584/2.9106`, with `0/20 <=0.15`. Increasing four-body radial modes to three lowers
    train median to `0.1357` but regresses held median/P90 to `0.9870/3.7610`. Thus a low-order
    smooth coefficient field alone underfits the labeled curvature, while added radial capacity
    reopens direction overfit. No arm is eligible for a new direction confirmation. Analysis
    summary/path/per-parent/direction CSV SHA256 values are
    `770ea6958e075b146a1704fa6ef57a603895cdaa1902b8d466d6f95ad22699ed`,
    `e5a040c9bf631cfabed1d3d4abe7b59d3157f2a5ef363f1d577188aa89f4493e`,
    `777b2a87ace4b74b36ac0e6d7de42b900941a8527b994615878e4e09a23ec6ab`, and
    `ade7211b23cf2634adb8f13ea3155e4410ad8cb41c6c278ca8c09a3ac5851172`.
    The next diagnostic audits whether the already passed stable5 v9 scalar coefficients provide
    a transferable prior on the 15 train20 parents not used in that full-Hessian fit; the five
    prior-fitted parents are marked and excluded from transfer claims.
170. Stable5 v9 coefficient-prior job 3846 closes that possibility decisively. Schema alignment
    preserves all 18,555 angular coefficients, maps 10,560 stable5 four-body keys into the 12,336
    train20 union, preserves all 7,680 random-kernel coefficients, and hashes the resulting vector
    as `66a67c40cdf50105a455741cc64b19a1537b4588e45e5f7ad1b82e1bbc6416e6`. On the five parents
    used for the original full-Hessian solve, full/held medians remain `0.01174/0.01006`. On the
    15 prior-unseen train20 parents, however, full-Hessian median/P90/max is
    `170.59/3356.55/4242.33`, held-HVP median/P90/max is
    `118.07/1612.29/1947.39`, energy median is `6.05 Ha`, and force MAE median is
    `3.19 Ha/Bohr`. The weighted Stage-2 design residual is amplified by `1457x`. The high-capacity
    v9 coefficient vector is therefore a parent-specific interpolant, not a transferable prior;
    residual fitting around it is forbidden. Summary/checkpoint/per-parent CSV SHA256 values are
    `38dae5170d5396a5673cc4f0317a1596e942451f0bc9a4f025a2248cddcb0569`,
    `01d155bf7189a4661c4fbb24fa7a576decb7d1cc93150e24d83909f5b4fa87ee`, and
    `152ba37dfb228510565522e7b40c9164ca41745492eb55a4d0e9bc1247d9b5f5`.
    This evidence redirects architecture work to a bounded equivariant scalar with shared local
    response; neither arbitrary feature coefficients nor this fitted vector may seed Stage 2.
171. A new bounded equivariant local scalar is implemented in
    `mldft/ml/models/components/local_equivariant_scalar_residual.py`. It uses scalar/vector
    message channels, typed atomic embeddings, smooth radial filters, C3 cutoff, and
    `sqrt(1+mean(v^2))` vector normalization; no zero-norm branch or force head exists. Atomic
    scalar readout is the sole energy owner, and reference Taylor anchoring preserves source E/F
    while retaining the learned Hessian. Remote rotation/translation, force-covariance,
    finite/symmetric-Hessian, and parameter-gradient checks pass with the existing capacity tests
    (`8/8`). Smoke job 3847 runs five finite steps in `1:59`: Hessian median decreases monotonically
    `2.72677 -> 2.68149`, max `3.40979 -> 3.36706`, E/F ratios remain exactly one, maximum
    asymmetry ratio is `1.15e-16`, and peak allocated GPU memory is `1.00 GiB`. Smoke
    summary/checkpoint SHA256 values are
    `1b4dadcc88307b9aa559dddbbcb439b9e8f8d20a7155901186877496cb77f9b3` and
    `e73fd0ffd4eda291e68000bd24a6397b9b43c1f62a2f69052b3930fb93fb5524`.
    This is an implementation/gradient pass, not capacity evidence. Formal 3,000-step job 3848 is
    active on one node01 A100 with a six-hour limit; no train20 direction, validation, or Test100
    data are authorized unless all five stable5 Hessians reach 5%.
172. Preregistered width/depth smoke array 3849 uses only the same stable5 labels and changes
    hidden width/depth. At 20 steps, h64/l2 (80,833 parameters) reaches Hessian
    median/P90/max `2.1483/2.5380/2.7938` in `178.9 s` with 3.06 GiB peak GPU allocation;
    h64/l3 (118,977 parameters) reaches `2.1356/2.5228/2.7774` in `266.4 s` with 4.71 GiB.
    The third interaction adds about 49% wall time for only 0.6% median improvement, so it is
    rejected before formal fitting. h64/l2 is the sole capacity expansion and formal job 3851 is
    active on a second node01 A100 while h32/l2 job 3848 continues. E2 summary/checkpoint SHA256
    values are `d6bf0b2620a87075eee067204c9a256fbedcc47b3770793c4483ebe7393b4feb` and
    `e43a8f45efd339025cce57eddd93ed064433d9172362e31bfb538508dd2348b4`; E3 values are
    `73c4a1354fc09bb959b0fa74cb6c409f5d103563e944ad82b3980167cf59cfc2` and
    `635bace5fd613d15499f43fa994d192341fb132156d8010b921aa4981b5ce049`.
173. The h32/l2 formal arm 3848 was stopped after step 350 because its median relative
    Frobenius had plateaued near `1.03--1.09` from steps 200--350; best/last checkpoints and all
    metrics through cancellation remain preserved. The h64/l2 formal arm 3851 is the active Adam
    capacity run. Through step 200 its median/P90/max improved
    `2.7268/3.1620/3.4098 -> 1.0218/1.1509/1.1552`, while E/F anchoring stayed exact and maximum
    asymmetry remained `3.1e-16`. This is still far above the all-parent 5% gate. To distinguish
    optimizer failure from representation failure, the exact h64/l2 step-100 best checkpoint was
    copied to immutable path
    `/scratch/xzh/models/complete_total_capacity/20260717/stage5_bounded_equivariant_v2/lbfgs_v1/adam_best_through_s100.ckpt`
    with SHA256 `3422fb7f891e31e016e31a8013d14975612e6dc924252cf89bcc59e15746a12a`.
    A preregistered full-stable5 joint LBFGS audit, job 3852, is active in parallel on node01 with
    30 iterations, learning rate 0.5, history 20, at most three closure evaluations per iteration,
    and the unchanged 5% gate. The manifest binds the source protocol/checkpoint hashes and
    explicitly freezes validation and Test100. No train20 or independent-parent evaluation is
    authorized while jobs 3851/3852 remain below the gate.
174. The optimizer audit and a registered one-parent decomposition now close the simple
    optimization explanation. Full-stable5 LBFGS job 3852 completes 30 iterations and 61
    closures in `1790.95 s`; median/P90/max relative Frobenius is
    `0.83084/1.00850/1.04520`, peak allocation is 8.92 GiB, and neither the capacity gate nor
    parent-CV authorization passes. Summary/checkpoint SHA256 values are
    `3746958f9729a8b3a7574aeb3598f713de60b6f177eadbef807e466dfac9c495` and
    `3fee651d396548517ff2222944a7a3bd4cdf9257f38b4db96a2961a6e2887c51`.
    Five parallel one-parent LBFGS diagnostics, job array 3853, each run 20 iterations from the
    same frozen step-100 checkpoint. Their final errors are `0.81165`, `0.93340`, `0.68913`,
    `0.73141`, and `0.50641`; all five fail 5% and explicitly report
    `parent_cv_authorized=false`. Thus neither stochastic Adam nor cross-parent gradient conflict
    is the dominant blocker: even individually, the current `l<=1` scalar/vector kernel cannot
    approach the requested ceiling under the registered optimizer audit.
175. The stable5 molecules span maximum interatomic distances of `12.11--15.09 Bohr`, compared
    with E2's 8-Bohr cutoff. E4 therefore expands the cutoff to 20 Bohr and the radial count from
    12 to 30 to preserve radial spacing. Its 20-step smoke reaches median/P90/max
    `2.03590/2.47797/2.73177` in `182.99 s`, versus E2's
    `2.14833/2.53797/2.79376`; this roughly 5% early improvement is real but not a capability
    breakthrough. Summary/checkpoint SHA256 values are
    `27bd88e1a99d877d9029f5dfbc6447bfa05daf482e4854aa9bf7af942355342e` and
    `b12baf5ca7b2e3e7e5ce56d3cd7fbf7bc949747ce2140264c6192b401ca668a6`.
    E4 formal job 3860 is active for same-step comparison. At step 400 its median/P90/max is
    `0.69714/0.84886/0.88144`, versus E2's `0.99592/1.10553/1.10705` at step 300 and
    `0.65228/0.83942/0.90115` only at step 850. The long-range arm therefore reaches comparable
    error in roughly half as many steps and has the lower tail. E2 job 3851 remains active as the
    short-range control rather than being treated as plateaued. E4's exact step-350 best was frozen
    as `lbfgs_v1/adam_best_through_s350.ckpt` (SHA256
    `228f8a3947e5c11410448494199d1163381e65fcb63513c60b5e8f093cae3f86`) for full-stable5
    LBFGS job 3866. Its iteration-5 median/max `0.75281/0.91073` does not yet improve on continuing
    Adam; the registered 30-iteration audit remains active.
176. A higher-order conservative scalar is implemented in
    `mldft/ml/models/components/local_tensor_equivariant_scalar_residual.py` using e3nn tensor
    products and explicit `l=0/1/2` channels. Energy is still an atomic scalar sum; force and
    Hessian are its derivatives, and `sqrt(1+mean(x^2))` avoids zero-norm second-derivative
    singularities. Rotation/translation invariance, force covariance, finite symmetric Hessian,
    Hessian parameter gradients, and builder binding pass the focused remote suites. Unscaled E5
    is finite but under-parameterized in function scale: five steps change median only
    `2.72677 -> 2.72666` with 10.13 GiB peak allocation. E6 adds a recorded scalar output scale
    of 1000, raising the step-1 gradient norm from `0.0010` to `1.017`; after five steps its median
    is `2.62981`, still slower and more expensive than E2. The first E6 submission 3862 repeated
    scale 1 because the builder argument was attached to the wrong architecture branch; it was
    cancelled, its logs were retained, the mapping was fixed, and a regression test now binds
    scale 37 exactly. Corrected E6 summary/checkpoint SHA256 values are
    `7d3b3027d376cb80423973a9a1bda9056541c1241bf59ac6c5689cfef95352ff` and
    `0adc1f3cd02bd7963f0654f783c7fdbb2b9b30ba6a276d4e7662c872fc37fedc`.
    Half-width E7 job 3864 completes 10 steps with 60,305 trainable parameters and
    median/P90/max `2.48409/2.92572/3.16329`. It reduces peak allocation from E6's 10.13 GiB to
    3.94 GiB but needs `335.87 s`, versus E2's `2.43628` median in `89.26 s` at step 10. It has no
    early efficiency advantage. Summary/checkpoint SHA256 values are
    `b3dd40a73b9afd52c8e6ed441da4bfd6b8e5d1ec1c3b6e5701c7124f91b158e9` and
    `66ee65a102de9f7e10ecff23da633d5593d4fd0981f1cf2e24bc3d1a2dbd5481`.
    Formal E7 job 3865 was stopped at the preregistered first comparison: step-25 median/P90/max is
    `2.00119/2.22014/2.35259` after `415.86 s`, insufficient to offset the 3--4x cost versus E2.
    Best/last/metrics SHA256 values are
    `5b66559570d6261c6de41c1f2ff4f2e2e67edf9b2c8bec28677612c5b26370ca`,
    `54c6f3070cb5ef33e2807d99cb1b6e4e1990764beca868a5e827064411f175ec`, and
    `e977419ac28e4bf0b96f07f6a0f0f15d5571eb703086b3901ab17d690dacd0b9`.
    No current tensor arm is promoted. All remain stable5-only and cannot authorize train20,
    validation, or Test100 access.
177. The registered E4/E7 optimizer diagnostics are complete and close both short alternatives.
    E4 full-stable5 joint LBFGS job 3866 runs 30 iterations and 60 closure calls from the immutable
    Adam step-350 checkpoint. It lowers Hessian median/P90/max from
    `0.73725/0.90007/0.93410` to `0.57441/0.68020/0.73621` in `1483.02 s`, with
    `9.10 GiB` peak GPU allocation and exact source E/F anchoring. This is a useful optimizer gain
    but still exceeds the all-parent 5% gate by more than an order of magnitude; summary SHA256 is
    `fd12fd34ffdd3c43a54d6f0ab9d52ee6edc65861c7a6689d13156988eccdc35e`.
    E4 one-parent LBFGS errors are `0.51314`, `0.62685`, `0.38820`, `0.53411`, and
    `0.38224`, so neither cross-parent conflict nor a single hard molecule explains the floor.
    The E7 `l<=2` one-parent audit is worse at `0.80732`, `0.93730`, `0.75055`,
    `0.78354`, and `0.95045`; all five runs explicitly keep parent-CV unauthorized. E7 is closed
    rather than extended. E2 and E4 Adam remain the only active same-budget controls. At the latest
    recorded points, E2 step 1050 has median/P90/max `0.65863/0.77224/0.83967`, while E4
    step 600 has `0.59507/0.70234/0.76245`. E4 keeps the better tail and reaches comparable
    errors in substantially fewer steps, but it has not passed stable5 and therefore cannot open
    train20, validation, train100, or Test100.
178. Because E4's registered 30-iteration LBFGS curve was still descending at its final point, a
    fail-closed extension was frozen rather than declaring a premature architecture floor. The
    active node01 job 3877 starts from the immutable Adam step-750 checkpoint
    `lbfgs_v2/adam_best_through_s750.ckpt` (SHA256
    `2e10ca9049765afaafb85a5f1cfc7c885af58045957522f661eec751e0afc36e`) and runs 100
    full-stable5 LBFGS iterations with the same learning rate, history, closure budget, scalar,
    labels, and gate as v1. The manifest and wrapper SHA256 values are
    `45bc5a276aaf35b492cfd5b554a8d67066ad326fc0d616632d5ba8f2c8c1698c` and
    `7e292818000662887fe81b130a0b64bfc52680f17f107a4c2784fd110a6580eb`;
    static checks and the LBFGS suite pass `6/6`. This extension cannot submit train20 by itself.
    Matched five-way one-parent array 3878 uses the same step-750 snapshot and 100-iteration
    settings to separate shared-parameter conflict from single-parent response capacity. Its
    manifest/wrapper SHA256 values are
    `1ae1ece83451e2843218a910106165ef965c47e726a91586aea537caffc9f085` and
    `585f282c7c3f171e5ffb1e302b039b4a88185c533d871bf84e3d21fdeeba4706`.
    One-parent outputs are permanently diagnostic and cannot authorize parent-CV even if one or
    more molecules pass 5%.
179. A distinct reference-local curvature scalar is implemented while the E4 ceilings finish.
    `LocalEquivariantQuadraticScalarResidual` predicts permutation/rotation-equivariant atom-pair
    Cartesian blocks from typed radial environments and covariant local moment frames, projects
    the assembled matrix out of rigid modes, and owns the correction through
    `0.5 dR^T DeltaH_theta(R0) dR`. Force and HVP are still exact derivatives of that scalar; no
    force head exists. This is explicitly a reference-geometry capacity model, not a global
    density-relaxed OFDFT functional. Rotation/translation covariance, rigid projection,
    atom-permutation equivariance, finite/symmetric Hessians, nonzero finite parameter gradients,
    builder binding, and existing capacity regressions pass `11/11` on node01. The
    module/trainer/two-test/config/wrapper SHA256
    values are `045740f6dac12b01d25e57fb05213c231605fc5e1f45ff8203b9870c922b49ec`,
    `05284380b5b58ee4ec4556f39e30503d282fc5c643bdff82e0169068515c1d8e`,
    `249cf76565854343d7c9b92569acdf0302c3df366c7c89508fb04f41efdc49ed`,
    `ad83cee59154ff657ba2656fe11a12f87e82bb980bd59060b2c757106ae49a26`,
    `2a358a752670d82d53bf5f8df39e6ff6b89ce82ea08bcdd8d3f81a991d0ae4e9`, and
    `7feb419eafc86998eedfe9148b7547c70cb5666235c40fe016f1f4425ebd5c97`.
    Stable5-only smoke job 3883 is queued for the first free node01 GPU; formal fitting, train20,
    validation, and Test100 remain unsubmitted.
180. E8 smoke job 3883 completes 20 full-stable5, full-batch steps in `23.19 s`. Hessian
    median/P90/max falls monotonically from `2.72677/3.16200/3.40979` to
    `1.49903/1.85060/1.96754`; E/F anchoring remains exact, scalar asymmetry is exactly zero,
    peak allocated GPU memory is only `31.6 MiB`, and MaxRSS is `1.15 GiB`. This is not yet an
    accuracy pass, but its per-wall-time descent is substantially better than E2/E4 and authorizes
    only its already frozen stable5 formal profile. Summary/checkpoint SHA256 values are
    `417b9947a02a89dde15433b1615750b6ab6937d6cd7885c705b6f3e4f969dcd7` and
    `423608d46a8b6cf3174b2add701117f4856b33ebcef2c0979fd934608c2f63d0`.
    Formal 5000-step job 3884 is active on node01; wrapper SHA256 is
    `39dddd214008eca212003cb073a4ce4db0ad4a28736342d25ccc38a73cc22c27`.
    No direction or parent-generalization data are open.
181. E4's five extended one-parent LBFGS tasks complete in `14:49--21:24`. Final registered
    relative-Frobenius errors for `0028399`, `0031108`, `0132419`, `0031012`, and
    `0121249` are respectively `0.18803`, `0.21765`, `0.09012`, `0.24459`, and
    `0.14693`; `0/5` reach 5%. Their summary SHA256 values are
    `12d870418af34ad3d9743806c803df49f8519dcb5bef11d296aeecab2443441a`,
    `d0c17fc2bfcce22fc2e05aa36bdfc1bc1fffa8131d9581ea6700542267c8883f`,
    `2c9090a69738df0df23da21c64d5dd278b58051fb0aad4902a5b90845531cf17`,
    `d4897036940d809b6b3d3d6f9cbbb4af280a0c4a92e2876680bff637de82e842`, and
    `73b3fcedcbe936d09d1a766b23b21673c1193ac2adfd3c9d93ccb995f98dbfb5`.
    This rejects cross-parent conflict as the sole E4 blocker under the frozen optimizer budget;
    the full-stable5 extension remains useful only to quantify the shared floor.
182. E8's step-500 checkpoint was frozen before opening any new data (SHA256
    `c0c1b0649965275b45218a868b76f6e51e6dc733fe24687dc3c8feb1f22d298e`) and used for
    five 100-iteration one-parent LBFGS response-span audits. They complete in under one minute
    each at errors `0.88046`, `1.04580`, `0.91909`, `1.29536`, and `0.73013`; no parent is
    near 5%. The manifest/wrapper SHA256 values are
    `941bf91d5ac56e6b9b6ff7a6df2aa9381378ad7d4a305ec9507a08caa692ce80` and
    `bb9576ad25f663b0cc7cb2362e6e49e4da3b53d855f2fbc9b1c49a631bcf7071`.
    Per-parent summary hashes are
    `fa4dc38eac1780b5a4624ffe8077143494ff5b8cf56a0654ea722c37caaa52db`,
    `c7b986d7a2d53deff157179784a346638a0c58e2933c291cc858af181fb9e4c9`,
    `d9a265c3f575dfd9fed895fb169c1486474ddc3c82a8af4ea6a5d684f3f81bb9`,
    `5e5ad5a4f892a128714e78d7b13a73d765513618f8dcf10bc72d2857826606c8`, and
    `e3f40add5d7b7dbcdd259ac39fe9a180aa263164910ae7f8fb0efbe524feef16`.
    Thus direct quadratic parameter gradients and fast runtime do not supply the missing response
    basis. The active E8 formal run remains a curve/control only unless it unexpectedly passes all
    stable5 gates; no parent-CV is opened from the one-parent diagnostics.
183. A mature higher-body scalar alternative is now isolated behind
    `LocalMACEScalarResidual`. It vendors `mace-torch==0.3.16` and `e3nn==0.4.4` only through
    `/scratch/xzh/vendor/mace_torch_0_3_16`; the project environment is not replaced. The adapter
    consumes the fixed all-pair topology, uses the MACE scalar energy only, subtracts an immutable
    same-initialization network, and applies the same reference E/F Taylor anchor. Force and full
    Hessian are outer derivatives of that one scalar; no MACE force output or independent head is
    used. Four focused tests pass on node01: finite/symmetric Hessian and nonzero third-order
    parameter gradients, rigid energy/force/Hessian covariance, exact derivative rescaling by a
    recorded scalar output factor, and atom-permutation equivariance through the all-pair graph.
    Module/capacity-script/test SHA256 values are
    `30c9025c3a5f5fe7aab63da0237a722bcd8b023a1f88374cdac319b69e242231`,
    `af47cc21bdec56a70c82bb19785cdc357805059ee8714182742c45a4d8cd09c1`, and
    `d3dd9440936cd2f416d0647ee5d37bdd5f31e717ba74f7c57aa5a6319b4bb5fe`.
    This remains a reference-local curvature-capacity audit, not yet a global density-relaxed
    OFDFT functional.
184. Three stable5-only MACE scale controls isolate the initial curvature-Jacobian problem. Job
    3891 (`scale=1`, `lr=1e-4`, one step) finishes at median/P90/max
    `2.726771/3.162001/3.409792`; summary/checkpoint hashes are
    `7ef1dc31cc695a44ff21ea7f9b49054a31a159513c11657323876b2952d3ea91` and
    `b6e853dfcc6a12f2167d726c4654a734148d7cc8b2264e5ca01403cb118d3490`.
    Job 3892 raises only Adam learning rate to `1e-3`; after 20 steps it is still
    `2.726103/3.161569/3.409291`, with summary/checkpoint hashes
    `8ea96100167da4908662000f3fa705547a5fc25145b34b7515a6ec30bbf074cd` and
    `1a7a423fea7d0d2297aac3142619e1d7d4d8a804755eab2d90d288650f6e6f07`.
    Job 3893 instead uses the tested scalar `output_scale=1000`, `lr=1e-4`; five steps reach
    `2.710127/3.152263/3.398806`, preserving exact E/F anchoring and maximum asymmetry
    `6.04e-16`. It takes `277.65 s`, peaks at 2.24 GiB allocated GPU memory and 1.97 GiB MaxRSS;
    summary/checkpoint hashes are
    `a1d9c2db9943b8479e0635b9d97467124d4af0b16769e2312cd151ba67e8e836` and
    `41b295b90166930c14bdf58ba5550d7588dff96eeaddab99a0ed55396d641138`.
    Thus learning rate alone is not the remedy, while explicit scalar-output conditioning gives
    about 25 times the total median reduction in one quarter as many steps. Failed submission
    3890 only exposed an unregistered stage name before model/data loading; it was corrected by
    adding that fail-closed stage to the loader and all artifacts remain retained.
185. The scaled MACE protocol/config/wrappers are hash frozen and formal stable5 job 3894 is
    active for 1000 steps on node01. Config SHA256 is
    `80783bd70957cebe90065fe8fad0b8bcb168fca3c9cf6ed1c344c8b9df9327fd`; smoke/formal wrapper
    hashes are `7387d0c34f0be0ce73a9596618caef356e2771ff04ffcfd9d8d7b925330e1773`
    and `1212ee8a4a576e11d57310f4e92e63d91e23932887ff58e6086f48c0a79a1b62`.
    At this submission point, active controls are E2 step 1900 at
    `0.49514/0.64082/0.72312`, E4 Adam step 1450 at `0.35598/0.43757/0.45995`, E4 LBFGS
    iteration 70 at `0.36669/0.44914/0.47884`, and E8 step 3100 at
    `0.89050/1.16136/1.26262`. None passes the 5% all-parent gate. Job 3894 can authorize only a
    newly frozen train20 protocol if every stable5 parent passes; it cannot read train20,
    validation, train100, or Test100 on its own.
186. The sole MACE capacity expansion raises hidden multiplicity from 8 to 16 and correlation
    order from 2 to 3 while holding cutoff, interactions, scalar scale, optimizer, labels, and
    five-step budget fixed. Job 3895 lowers median/P90/max from
    `2.72677/3.16200/3.40979` to `2.68201/3.11183/3.36552`, versus h8's
    `2.71013/3.15226/3.39881`. The 49,584-trainable-parameter h16 model therefore gives about
    2.7 times the h8 median reduction at similar wall time (`297.30 s`), with 3.18 GiB peak GPU
    allocation, 1.97 GiB MaxRSS, exact E/F anchoring, and maximum asymmetry `8.44e-16`.
    Summary/checkpoint SHA256 values are
    `6ea7ef7698a7912acf8871c526f77dd2c2093a7cfaf9efb8549761bc16159590` and
    `da0aa001df54745397f1cb7af19be302b4aff635485f8f8a086d90a6cad430fc`.
    Config/smoke/formal-wrapper hashes are
    `f66eb24a7a87a908c03a5636d94de971687e80d6a4e06cb6856cbb9fe57900e9`,
    `bad69ed210d3fb7b8499db57e1d6c645d23c92907b3669e71429e89971da1fda`, and
    `419bd1b0a7c146dcc2bfda1335b4dfddf56b52ad44192f165f996e6cc3a31719`.
    Formal h16 job 3896 is active for 1000 steps as the capacity candidate; h8 job 3894 remains
    the matched width/body-order control. Neither run may expose train20 or any later split before
    passing every stable5 gate.
187. The two superseded active ceilings are now final. E4 full-stable5 LBFGS job 3877 completes
    100 iterations and 200 closure calls in `5540.97 s`; median/P90/max relative Frobenius is
    `0.33474/0.39514/0.41334`, and its five per-parent values are `0.33474`, `0.41334`,
    `0.23758`, `0.36785`, and `0.30769`. This is a real continued optimizer gain from the
    30-iteration result, but `0/5` reach 5%. It peaks at 9.10 GiB allocated GPU memory and
    2.09 GiB MaxRSS. Summary/best/last SHA256 values are
    `8ac996d992ccccc785c34abd13f5c57986f8666fafbd6982458f7452f9881113`,
    `3aeb3618679a69280ef368cf11adfbbe64b2f6fd0f37258a2e08e2488dd1364e`, and
    `b63d9abb4134c7e30a2fb037b11c144efdcdd0f3027369499f419737d822e091`.
    E8 job 3884 completes 5000 steps in `4401.18 s`, selecting step 4800 at
    `0.88694/1.15781/1.25912`; its per-parent range is `0.67413--1.25912`.
    Summary/best/last hashes are
    `a059116b34f3f35306ab4a11504d03191b08caa50cf47df02024247f88513852`,
    `fbca526910e2abb662a51f135071f74ac799b6d9acd41a2d3a9a406eea6e239f`, and
    `819afd02372b8ed5096f1b047437bb163a16e0ecdd6a928d0b1c7da20fb4daf1`.
    Both preserve exact anchored E/F and scalar symmetry, but neither has adequate response span;
    both branches are closed without train20 access.
188. MACE h16 formal job 3896 reaches step 50 at median/P90/max
    `2.41501/2.85989/3.11622`, versus h8 job 3894 step 125 at
    `2.54461/3.05923/3.29103`; all reported points remain finite and symmetric. A read-only
    fail-closed merger now waits on both jobs as dependency 3897. It writes final comparison,
    per-parent and learning-curve CSVs, JSON with input hashes and threshold-crossing steps, and a
    log-scale curve plot. It sets `selected_candidate=null` and keeps all later stages frozen
    unless a completed summary already reports the full stable5 gate as passed; it cannot submit
    train20. Analysis/test/wrapper SHA256 values are
    `08eb80fa168ae32b045305a6a776614aaf94c76a28581720c5bb226160d27126`,
    `0bfcfc7524ad3534b38362b96f852a064e2d1661453a8bec1c67be9bb4c59c83`, and
    `f2381141b91d7100c9fbeb4e1f1de24e03e3d72afda7929ee28521a12e071d0c`;
    its frozen-access and fail-closed tests pass `2/2`.
189. A registered one-parent response audit now runs in parallel without opening new molecules.
    The h16 formal step-75 best checkpoint was atomically copied to
    `stage5_mace_h16_scaled_v6d/one_parent_lbfgs_v1/adam_best_snapshot.ckpt` and frozen at SHA256
    `33f5842f150d3a822fe31177f59fb7262cec781bbf217fbe2a416d79f0893229` before the
    array was submitted. The generic LBFGS audit whitelist now accepts the distinct MACE protocol
    while retaining stable5 scope, frozen validation/Test100, source hashes, fixed parent list,
    and permanent `parent_cv_authorization_allowed=false`. Its local-import fallback also avoids
    the isolated vendor's unrelated top-level `scripts` package shadowing repository modules.
    The expanded suite passes `7/7`, and direct vendor-path `--help` import succeeds.
    Script/test/manifest/wrapper SHA256 values are
    `5db3fb802e3881dc6052d255750576490c2c4edd35f2b9d3541427957caa13bb`,
    `e08b8c14fa3d63944f930bfc1c5a4dce3209c8327a53ed3ffbd966a72f05bd6c`,
    `aa246cc8a21d2fc863acb889e5832726978845a26427480e0722d15f0b2a76c2`, and
    `b7019aba4b991f63dbbcc2ec1720c428e7a8e04b983c3e11d1c8c6eead2f6762`.
    Array 3898 runs 50 LBFGS iterations per frozen stable5 parent with at most four concurrent
    GPUs. The first four iteration-0 errors are `2.31037`, `2.34392`, `2.37888`, and `3.03450`,
    exactly matching the source checkpoint. Even a 5% result remains diagnostic and cannot open
    train20; the array only separates single-parent response capacity from shared fitting conflict.
190. At iteration 10, the first four one-parent MACE LBFGS tasks reduce relative Frobenius from
    `2.31037/2.34392/2.37888/3.03450` to `1.33745/1.44191/1.37854/1.94463`.
    This verifies optimizer movement but is not yet evidence of the 5% response span. Dependency
    job 3903 will merge all five completed summaries into a
    hash-bound JSON, per-parent CSV, and log-scale bar chart. The merger rejects any source
    checkpoint mismatch, Test100 access, scope mismatch, or accidental parent-CV authorization;
    its two fail-closed tests pass. Analysis/test/wrapper SHA256 values are
    `56c6697cd6fbcec6caae2f67ac4ff1faf71e968dc93394ee8006f2d440fea19a`,
    `0368e517bbea4951bca1aa64034287f9bd8fe0d12102e89d0f76427a849781fc`, and
    `1e393ccb83b1b0276ab93a1d137b44b5702b0bbbe0ab28978ddd88da47b776bc`.
191. By iteration 20, those four one-parent errors are
    `1.11154/1.25966/0.99307/1.22096`; improvement continues but remains far from 5%.
    A single higher-capacity fallback is preregistered rather than launched unconditionally:
    h32 channels, `l<=3`, correlation 3, and three MACE interactions, with the same scalar scale,
    optimizer, stable5 labels, and gate. Dependency job 3904 waits for the one-parent merger 3903,
    verifies frozen Test100 and permanent parent-CV prohibition, exits successfully without model
    construction if h16 passes all one-parent 5% gates, and otherwise runs only a three-step
    stable5 smoke. It cannot submit formal fitting or train20. Config/wrapper SHA256 values are
    `886c68e61ee4a8b5867beed47399c434565da86bac94db553ff11fcaa9f16764` and
    `900cbceee824035d11531295d0d43782a5f4bafd363e4186a03fdc5924b7cb40`.
    Static shell/YAML checks and the frozen protocol loader pass.
192. The h16 one-parent array and fail-closed merger are final. Per-parent relative Frobenius
    errors for `0028399/0031108/0132419/0031012/0121249` are
    `1.00092/1.05740/0.84937/0.91837/0.80717`; median/P90/max are
    `0.91837/1.03481/1.05740`, and `0/5` reach 5%. Every best checkpoint is at registered
    iteration 50 after `100--102` closure calls, with per-parent wall times `1194--1648 s`,
    peak GPU allocations `1.29--3.20 GiB`, and MaxRSS `1.80--1.95 GiB`. This rejects shared-parent
    conflict as the primary failure: h16 does not span even one parent adequately under the
    registered budget. The machine decision is
    `single_parent_span_not_supported_at_registered_budget`; parent-CV remains false and Test100
    remains unread. Summary/CSV/plot SHA256 values are
    `1c64b57594cf1a6d65e0845de1cbbb6302c9a025a8eadaa6a6a6f85ca1a0a40d`,
    `b0515ac694e9aac16d5973a4d060a6b65c85ec393be8aed62a866017b7c4265d`, and
    `5ab04e0882b570c520bf3fa734f8747e29c0f44ebec40daaae1ed983b3b83bfe`.
193. Conditional h32 smoke job 3904 completed all three steps with finite scalar derivatives and
    maximum Hessian asymmetry `5.95e-16`. Median/P90/max improve only from
    `2.72677/3.16200/3.40979` to `2.71156/3.13933/3.38585`; h16 is already lower at matched
    step 3 (`2.70254/3.13089/3.38237`). The h32 model has 323,904 trainable parameters, takes
    `625.3 s` internally (`10:45.64` under `/usr/bin/time`), peaks at 23.74 GiB GPU allocation
    and 2.91 GiB MaxRSS, versus h16's 3.18 GiB early GPU peak. h32 is therefore a costly capacity
    probe, not a promoted model. Summary/best/last/per-parent SHA256 values are
    `3435f15cc429ebc97cc2f7574fc414eed57863df2df19dd26f19021dfefd4cbc`,
    `1ac92a219ed416c12653bddfdef205102c879b308e60f91465e0212e0cc2b848`,
    `645c65e6a458d10687dc2ebea011dad1226cd67b450b395d2fb8ff4bf01b9836`, and
    `59adf54f4512b641ef3277dddf735d8fc3de8b29cb7adb0b537d113868f62cdb`.
194. A final h32 single-parent response-ceiling audit is now registered from the immutable smoke
    step-3 best checkpoint. Array 3906 runs the same 50-iteration LBFGS budget independently on
    all five stable5 parents, at most four node01 GPUs concurrently; dependency 3911 performs only
    hash-bound read-only aggregation. It cannot authorize parent-CV, train20, validation, train100,
    or Test100 even if all parents pass. The manifest/array/analysis-wrapper SHA256 values are
    `567ede6ab47fbc9ac903c000eece19f58f9eabf998a077f93ca02e6e131d6030`,
    `e4a23543e009b8234309eb9095bc4d75a4b460faa41e33a30d77fcf1f217cf3d`, and
    `78e8062c565a43fe23fa0df7100a916df9d3bebf35f9c2fa5e64dacb9dc54e63`.
    JSON/shell validation, actual manifest binding, and the LBFGS/merger test suite pass `9/9`.
195. Shared stable5 Adam controls continue only to complete their frozen curves. At this update,
    h8 step 475 is `2.14674/2.66629/2.85831`, while h16 step 350 is
    `1.68937/2.08104/2.33998`; h16 remains clearly better but neither is near the 5% gate.
    No later-stage data have been opened.
196. The matrix-free parameter-to-Hessian Jacobian audit now supports current stage-tagged stable5
    protocols, isolated-vendor imports, and an explicit single-parent selection. The generic
    wrapper can pass the MACE vendor path, parent ID, and E/F anchoring without weakening its
    checkpoint parent-list or Test100 checks. Five focused tests pass, including exact CGLS
    agreement with dense least squares, rank-deficient projection floors, current/legacy stable5
    protocol recognition, and fail-closed parent selection. Script/test/generic-wrapper and two
    h16 audit-wrapper SHA256 values are
    `07d03ce3cae7401d7435c01a61b1bb39d9f348f5260d4f4f56c9c39e198e4824`,
    `1247156575a157f5890967cac6feadafdba2a71ad1e2fe9408d5e0cd12368539`,
    `aad635d56a4fd614672ecdf0d2e16f7ac32b4b7a5e397e2a8f33405d6c9c040a`,
    `edca8898bbf30bf960774087edb557145e74a2b9820b751b6f5974798c28cd9d`, and
    `12739458c15ff4aa89c79600dd971dc44bc6f901ddc8cdec000b286dc2a435f5`.
197. The first h16/`0028399` step-75 Jacobian audit used eight CGLS iterations. Its normalized
    linear residual falls from `2.31037` to `1.29999`; parameter-direction finite differences at
    relative steps `3e-6` and `1e-6` agree to relative `4.62e-9` and cosine 1.0. The unrestricted
    nonlinear probes expose why this is not a global E/F candidate: even alpha `0.01` gives a
    75.4-fold source energy-error ratio, and alpha 1 gives 428-fold energy and 4.60-fold force
    ratios. The run takes `193.4 s` internally (`3:34.41` process wall), 1.24 GiB peak GPU, and
    1.67 GiB MaxRSS. Summary/CGLS/linear/nonlinear/step hashes are
    `5678de2114b62b6e2f86bcddc401ca4ec119ea5dbe25589f0324f8d831a254a7`,
    `6d29b30e7b0afa3c335b0e7bbf97ae353e9bf565d07b876d2881e49ecb887e50`,
    `38c55d73f39ec366eeec0502c2a383a2871c55baeeeb5de29bc993f76cc16a5e`,
    `00710a469456280e38af9c7c51603d23fea67ad81042a95822fdcb46ee509d5a`, and
    `960cb967051973d3ac15324fb0c8e4a92404fb5d315d3716c156ac007c34312f`.
198. The superseding 30-iteration audit evaluates nonlinear steps with the same reference E/F
    Taylor anchor as the current curvature-capacity model. The linearized relative residual reaches
    `0.99491` from `2.31037`, almost exactly the independent h16 one-parent LBFGS final `1.00092`;
    the FD stability remains `5.31e-9` with cosine 1.0. Actual nonlinear Hessian errors for alpha
    `0.01/0.03/0.1/0.3/1.0` are `2.2914/2.2528/2.1100/1.6502/2.9529`; thus the linear direction
    does not survive a full nonlinear step, while anchored E/F remain exact. This is quantitative
    evidence for a poorly conditioned/inadequate local MACE Hessian response, not a sign/unit or
    LBFGS-only failure, but it is not a mathematical global architecture lower bound. The run takes
    `579.5 s` internally (`10:01.38` process wall), 1.24 GiB peak GPU, and 1.67 GiB MaxRSS.
    Summary/CGLS/linear/nonlinear/step hashes are
    `3dcfc764dbb6af4abf71a10ca1165429d7c1389cc17600cd513353de7916168a`,
    `390fb9c007244b514e1acf6cf90adf6cfa60b597bc8919a76c31678b61acf730`,
    `cfb295ade03673a8be30a1bbc986dac2a4c68a46e94940127e4f3112d2eb6ef7`,
    `ed3be076709a272f866fd084c0f1a9913f731e558c482683a8a504d8c94298df`, and
    `fd4fdd31de4115b3444b81d3849128bd9b3430197c1da6a5f9e272f8f8cf4f73`.
199. Because the CGLS normal-gradient is not yet at its requested tolerance after 30 iterations,
    job 3914 extends only this same-parent linear response audit to 100 iterations. It changes no
    model, label, checkpoint, nonlinear alpha grid, or data scope; its purpose is to distinguish
    slow ill-conditioned convergence from a true low-rank projection floor before selecting the
    next scalar representation. The wrapper SHA256 is
    `c7ac01ce753b8758faf678bec64d4ad94884cec482a698e28a90bc82b0d56989`.
200. The 100-iteration h16/`0028399` Jacobian audit is final. The normalized linear residual falls
    from `2.310365` to `0.797492` (relative `0.34518`), but the required parameter step is already
    `0.09256` of the base norm. Parameter-direction finite differences remain stable at relative
    difference `5.50e-9` and cosine 1.0, excluding a broken JVP/VJP implementation. The anchored
    nonlinear Hessian errors at alpha `0.01/0.03/0.1/0.3/1` are
    `2.2901/2.2496/2.1100/1.8502/13.5470`: the local linear improvement does not survive the
    finite nonlinear move. Internal wall time is `1672.7 s`, peak GPU allocation `1.24 GiB`, and
    MaxRSS `1.72 GiB`. Summary/CGLS/linear/nonlinear/step SHA256 values are
    `d5680382541052195778c515da68b20a2990cdf95b7d42e9159ed16bf6e739c9`,
    `7771cabed25ea4b073d40115ee8c2b52d00814e478878a7add6e98ebfd2df4a2`,
    `d4daa669ace3da078122adca40173be7ea99fb4a2fc03045ed7b62981ab445a2`,
    `19c1891109f2bce22e4260450a8f43988e287b0ace670d0d3506ed4f34cb1e49`, and
    `145bc6775a67544f5af8455379b9d0a5865f459793c18f0cc93cf6e9539047b5`.
    This rejects more percent-level optimizer tuning of the current nonlinear h16 model; it is
    strong conditioning/nonlinearity evidence, not a mathematical global lower bound.
201. A structurally different scalar-capacity arm is implemented. It freezes the hash-bound h16
    step-75 MACE backbone (`33f5842f...3229`), extracts each interaction's `l=0` node channels,
    applies a fixed 512-wide tanh random expansion, and trains only a zero-initialized,
    element-resolved linear scalar readout. The final MACE interaction is parsed as scalar-only,
    matching its default `keep_last_layer_irreps=false` layout. E/F/H remain derivatives of one
    scalar; there is no force/HVP/Hessian head. Only `2720/101888` parameters are trainable, and
    the source checkpoint parent IDs, arm, Test100 flag, and SHA256 are fail-closed. Remote tests
    pass `9/9` for finite second derivatives, rigid covariance/invariance, atom permutation,
    output scaling, zero initial jet, exact parameter-to-jet linearity, and trainable readout
    curvature. The five-step node01 smoke
    is finite and lowers stable5 median/P90/max relative Frobenius from
    `2.72677/3.16200/3.40979` to `2.67820/3.12026/3.37063`; E/F are unchanged by the Taylor
    anchor, asymmetry is `3.67e-17`, peak GPU allocation is `1.41 GiB`, and wall time is `161.8 s`.
    Smoke summary/checkpoint/resource hashes are `9e0115c027c14a52a05b888aeb0734c69d18d0c5e0122ec00d1cdd0aadfee54f`,
    `f53a0602f04f3eadb505ca38988f6125e7a6c4e22b0d7cccabc16b28a7a575c2`, and
    `b21c7b0433e6670cdca3960876d4cd58985a61fa9d1a66125c2b0dd571937537`.
    The expanded test file SHA256 is
    `b889ac8e6d670d84dc042a9ad97100eee4fde21029ce7f23c91d40896c64089c`.
202. Job 3916 runs the preregistered 1000-step stable5 formal trajectory without changing the
    smoke hyperparameters. Because the readout-to-Hessian map is exactly linear, job 3917 also
    runs a hash-bound, all-five-parent, 100-iteration matrix-free CGLS ceiling from the immutable
    smoke checkpoint. Its alpha-1 nonlinear probe must agree with the linear projection before
    any capacity claim. The CGLS core tests pass `5/5`. Neither job can authorize train20,
    validation, train100, or Test100; Test100 use remains zero.
203. A stable5 feature-scale audit on `0028399` finds RMS `1.7694e-3` for the h16 scalar channels
    and `5.7408e-4` for the new scalar-plus-`l>0` channel-Gram invariants. Consequently the v6f
    unscaled tanh expansion is almost linear. The separately preregistered v6g arm therefore uses
    invariant Gram/power-spectrum features, a fixed input scale of 1000, and 2048 fixed tanh
    features. This is a representation/kernel correction, not another percent-level optimizer
    tweak. Its three-step smoke improves stable5 median/P90/max from
    `2.72677/3.16200/3.40979` to `2.49285/2.76322/2.86307`, while anchored E/F remain unchanged,
    maximum asymmetry is `2.02e-16`, peak GPU allocation is `0.98 GiB`, and wall time is `121.3 s`.
    The model has `11760/110928` trainable/total parameters. Smoke summary/checkpoint/resource
    SHA256 values are `3db861bc7a39d03e07683bbce76f73867b992eee72529b47cfc7ce0918f8efb8`,
    `9c773b8d2955542a71c5834a9cc815bfd2ed11ce5b40f0316252be69a98ac8f7`, and
    `07f8bb95088a4055f6dc7958d9bf3d5635faa3ed4cd96e3cd8975f98d5aead12`.
    Job 3919 now measures the exact-linear all-stable5 CGLS ceiling; no 1000-step formal job is
    authorized until that ceiling is known. Protocol/smoke-wrapper/CGLS-wrapper hashes are
    `7f99b74c603cd1f67f81476941a88ab7bc68d61e3c6d25e89ff3de3ae4f2321f`,
    `2cddeb67eeecc1669fca7bacd630903b22f976ce8026fdc7b27494a77e376001`, and
    `7249f6c15e1e0ecb772c18a97d9c96dbdecc3fd34463c5520e15660e6400b0d4`.
204. The h32 one-parent response audit and dependency merger are final. Per-parent relative
    Frobenius errors for `0028399/0031108/0132419/0031012/0121249` are
    `0.94356/1.05070/0.75527/0.95744/0.80344`; aggregate median/P90/max is
    `0.94356/1.01340/1.05070`, with `0/5 <=0.05`. This is worse than the h16 one-parent median
    `0.91838`, despite 323,904 trainable parameters and per-task peak GPU allocations up to
    `23.99 GiB`. No h32 formal stable5 fit is authorized. Aggregate summary/per-parent hashes are
    `2b7c80c041d6bf938bb4eae2ceb51e2870efff2552ef6b57afc78368b6ae6910` and
    `9150e200e9048385464e81fe7d757d8e82ba16c46685f4800936600d23a933f1`.
205. The matched h8/h16 formal stable5 trajectories are final and both fail. At their selected
    checkpoints, h8 step 1000 has median/P90/max relative Frobenius
    `1.40140/1.64888/1.72606`, while h16 step 975 reaches
    `1.15016/1.47098/1.52560`. Their wall times are `13967.5/15460.7 s`, and peak GPU
    allocations are `2236/3178 MiB`. The h16 gain is real but remains more than twenty times the
    5% all-parent gate; exact Taylor anchoring preserves source E/F rather than fitting PBE E/F.
    Summary SHA256 values are
    `a6176ffcce0834ab29ce1e0dd206383c9400eb1817b1383ee75d7dbd88e1c07a` and
    `d26e9892ba20826db0835c148fa9b7d0ce977f65b99c5f1fee91915e809e6c14`.
206. Exact-linear v6g response audits separate optimization from feature span. On `0028399`, the
    width-2048 power readout reaches relative Frobenius `0.28578` after 100 preconditioned CGLS
    iterations, and its alpha-1 scalar evaluation agrees exactly. On all stable5 parents, the
    unpreconditioned 100-iteration solve has global residual ratio `0.31596` and per-parent
    median/P90/max `0.79290/0.88026/0.89720`. Thus the shared frozen feature span is inadequate
    even though the parameter-to-Hessian map and scalar evaluation agree to roundoff. The
    one-parent/all-parent summary hashes are
    `7ba6021ffa70c3a11bd600a80eb2bc3c40c58ac4607df5ead9a8473061c8312e` and
    `364220fe3221b9e47d8b7244da9d0f073e6b7808b1752df84ce380f2d8c10c83`.
207. A separately hash-frozen width-8192 readout does not repair the one-parent ceiling. Its
    one-step optimizer smoke selects the zero-readout step 0 checkpoint; the exact 100-iteration
    CGLS audit on `0028399` finishes at `0.30773`, slightly worse than width 2048 under the same
    iteration budget. The solve uses 42,480 trainable coefficients, 751 MiB peak GPU allocation,
    and `732.5 s`; its exact-map repeat check differs by `7.7e-16`. Smoke/audit summary hashes are
    `0c9034046ddd941c4925b860131c8a8b9c52bcb141a81393c25729fe2f420cef` and
    `43a5952dc4424e6445cc164cd46106f0f7d98392056f19c70b02d955876ea5de`.
    Width alone is therefore rejected as the next architecture change.
208. The remaining structural confound is the zero-value/zero-force Taylor anchor. The existing
    rigid-mode audit gives an unavoidable anchored floor of median/max `0.04039/0.04178`, leaving
    less than one percentage point below the requested 5% gate. The CGLS audit now optionally
    projects the full scalar jet `DeltaE/DeltaF/DeltaH`, verifies a hash-bound source protocol, and
    reports physical E/F residuals. Its force block is normalized by `sqrt(3N)`, so the objective
    is energy squared error over `0.1 Ha`, mean force-component squared error over
    `0.05 Ha/Bohr`, plus Hessian squared relative Frobenius. Focused tests pass `7/7`.
    An initial summed-force implementation, job 3925, was canceled after six iterations and is
    retained only as an invalid diagnostic. Superseding node01 job 3926 runs the preregistered
    one-parent, unanchored, joint scalar-jet ceiling for 200 CGLS iterations. It still opens only
    `0028399`; train20, validation, train100 and Test100 remain closed. Script/test/protocol/wrapper
    SHA256 values are
    `ba834ae9b5261cda7819aac6de28533a1311db97c520bdebc4098497e33f1d7f`,
    `a52d8ca4ccd474053be0b9857ae66d0befc2d49ae30f6caddc76e3f3e799d0cd`,
    `001ccbde621293fb2b370d88bd51fd43af7433be11abd957b361dacf27d79d5d`, and
    `1a2f93e9b52ce8a699dbf8f99ce9453947f8ebfbecafaf0d57a5874de2268be2`.
209. Joint-jet job 3926 is final after 200 exact-linear iterations. On `0028399`, energy absolute
    error is `8.20e-7 Ha` and force MAE is `1.342e-3 Ha/Bohr`, but Hessian relative Frobenius is
    still `0.42023`; the normalized global residual is `0.42264`, or `0.09763` of its initial
    value. The alpha-1 scalar evaluation reproduces these linear residuals and has Hessian
    asymmetry `8.5e-16`, so E/F/H all come from the same conservative scalar. However, the final
    normal-gradient norm is still `0.813`, so `0.420` is not yet a converged response-span lower
    bound. The run takes `1516.8 s` and 752 MiB peak GPU. Summary/parameter-step/metrics hashes are
    `21cb4658f7cc0ad957ba2adaac0eab63196d22113e4814e7563ba7d502bdb2e7`,
    `512556f36c7032efc39ba76d557c34996fd9e170b144bf996fbc2ba31044a8e2`, and
    `0b80c79d7c8913aa6d91a4d955910b81089643ece74bd69d60e82cf2eb612ecf`.
    The CGLS CLI now supports a hash-checked initial parameter step and reports `null`, rather than
    nonstandard JSON infinity, when the base readout norm is zero. Node01 job 3928 restarts from
    the saved step for 800 additional iterations; this is a solver-convergence audit with no
    changed data, features, or loss. Current script/protocol/wrapper hashes are
    `1c82af386520708e6b06ad04b337c78d420cc67b8920a548387204a7f407db14`,
    `38a2a9772e23c826a1f0c7f86e4593d8e451ba5c0560aba7c5a6552f5fc15cec`, and
    `7e7d2ddb9c2466b55a6494401ac74a6061516524cd2b720d06d9b0357d63cab0`.
210. A separate scalar force-secant capacity path removes the full-Hessian loss's third
    geometry/parameter derivative from training. It fits h64/l<=3/correlation-3/three-interaction
    MACE with energy, scalar-derived base force, and centered scalar-force secants at
    `R +/- 1e-3 v`; complete autograd Hessians remain evaluation gates. Secant sign/parameter
    gradients and input checks pass `3/3`. The one-parent five-step smoke is finite, takes
    `146.9 s`, and peaks at `3719 MiB`; energy error reaches `0.0461 Ha`, while five sampled
    directions predictably do not move the full Hessian (`2.73045`). The 500-step node01 pilot is
    now final: energy error is `0.004550 Ha`, force MAE improves from `0.116073` to
    `0.021662 Ha/Bohr`, but Hessian relative Frobenius changes only from `2.72677` to
    `2.71386`. It takes `1001.2 s` process wall and `3719 MiB` peak GPU allocation. Thus the
    force-secant graph is valid and force learns, but ordinary summed-loss Adam does not transfer
    the sampled curvature supervision to the full Hessian. Script/test/smoke-summary/pilot-summary/
    pilot-protocol/pilot-wrapper hashes are
    `37ab557032c814b93cb8c6ee6ca05e2ca73719a170f91fc424a8dfe7d5e4dddf`,
    `f9ac517e2f421bdc9b3728a5abb32250a5c8360c66efe8739f9cf8337387af15`,
    `edfca6b626a55f9506acf0c93cf01968dd2529dce78163a810b335f66c19920d`,
    `c233e0e202180e9b2981be670411f1a7b27982c98f2381197502952775d4aac0`,
    `8229d205bc6ab7bfc0e5992ea4446bebef37b8ab6ed975dbed48be7d9212abe9`, and
    `b34c757b12bfd1aebd6cf6da64ed79ea2610bc2498febda64754d24ee57f55f9`.
211. The hash-bound v6n step-500 gradient audit evaluates eight Cartesian directions without
    training. Median weighted parameter-gradient norms for normalized energy/force/HVP are
    `42.1838/3.01459/16.0920`. Force-HVP cosine is negative for every direction, with
    median/range `-0.2361/[-0.3096,-0.1010]`; energy-HVP median is `0.2181` and energy-force is
    `0.1101`. This quantitatively identifies both task-scale imbalance and systematic force-HVP
    conflict. The audit is finite, uses `6642 MiB`, takes `53.4 s`, and keeps validation and
    Test100 unread. Summary/per-direction/resource hashes are
    `ae1728c28e6ab0cf828c4f783ff1102825d1079e60815496d1ff371e9963dd27`,
    `3ff67c311ffd7779eea73a560db53cb863fa21b804db6c9ff3c78727fa156656`, and
    `dd42a63e76bf9b7d22ddc3a2a8066f0a2dfdc5d458447677a96945f7fa5bcd6f`.
212. A deterministic fixed-scale PCGrad implementation divides each task gradient by those frozen
    audit norms, projects only negative pairwise components, and records raw/projected cosines.
    Fail-closed initial-checkpoint metadata and gradient-audit hashes are enforced. Remote focused
    tests pass `18/18`. The v6p Adam smoke starts from the v6n checkpoint and makes all projected
    cosines nonnegative, but fresh Adam causes energy to oscillate between `0.0307` and
    `0.000957 Ha` within two steps; its selected step 2 leaves force/Hessian at
    `0.02159/2.71389`. The v6q zero-momentum SGD smoke preserves the projected direction but fixed
    step `1e-3` crosses the local descent region, reaches `0.207 Ha` energy error at step 4, and
    selects the unchanged step 0 checkpoint. Their summary hashes are
    `47180d186ccf7cf04182418b797bd3ee22b60c8bc2582061efa925ecc1dd3bdd` and
    `ec95703724ed1bdc7309ca3a8e63c5e3def58d123dc5cadceda49eac188a5feb`.
    Neither fixed-step formal run is authorized.
213. The superseding v6r mechanism uses the same scalar, data, directions, loss, and frozen
    gradient scales, but backtracks a zero-momentum SGD step until each sampled E/F/HVP task is
    nonincreasing and their sum strictly decreases. Its five-step smoke accepts all updates: four
    at `1e-4` and one at `1e-5`. Energy error falls from `4.55e-3` to `7.03e-5 Ha`, force MAE
    changes from `0.021662` to `0.021639 Ha/Bohr`, Hessian relative Frobenius remains
    `2.71385`, symmetry is at roundoff, wall time is `145.3 s`, and peak GPU allocation is
    `6487 MiB`. Smoke summary/checkpoint/resource hashes are
    `91b0c1b4803ca694155ca445e3863f7b1248f38d1f8df4a4fc81574783c88ed9`,
    `2d8ffcaf6d7a22a4482525687f4dec178c4a1309a16ba69dc6aa4f0567201b73`, and
    `f9e35004e9a5443a3f3f1075e4534191e828f2379a329ab3a4b597aff7602a02`.
    The 1000-step wrapper was launched as node01 job 3939, but the run was intentionally stopped
    at step 150 after its accepted step collapsed to `1e-8`; energy reached `9.75e-8 Ha`, force
    improved only to `0.021512 Ha/Bohr`, and Hessian worsened to `2.71518`. The strict
    nonincreasing-energy rule had become the active constraint, so extrapolating this trajectory
    to step 1000 had no information value. The step-150 checkpoint and training-metrics hashes are
    `f8145416df11b2ac1188438f6b81ac5990af82902d225a4b81054096a0177f78` and
    `a9229632ce5d230ed5ba3d98901ca2e95645dff6508cfed8c30d90052add4c17`.
214. A read-only checkpoint geometry diagnostic separates correction amplitude from direction. At
    v6r step 100, the learned Hessian correction has norm `1.6314` versus target-correction norm
    `10.7831` (`15.13%`), but their cosine is only `0.10447`; the complete Hessian relative
    Frobenius is `2.71485`. Thus the failure is not only a small correction amplitude: the learned
    correction direction is poor. The canonical step-100 checkpoint/result hashes are
    `c9d0f6789089c7e369c6ca5792b5a05629ec04dfdd611bf017ee0aa57e9bb395` and
    `b4b8e39fb2e09ffcaecbf8c93998a0ac9cdd86a6b9a47f20bad68c39ade13aa4`.
    A race made the first immutable copy's filename say step 50 even though checkpoint metadata
    says step 100; that duplicate is retained for provenance and must not be used by name.
215. The v6s successor replaces strict loss monotonicity with physical E/F budgets. Energy must
    decrease until normalized loss `4e-4` (`2e-3 Ha`) and may then move only inside that budget;
    force must remain nonincreasing until normalized loss `3.6e-3`, while the sampled HVP must
    always decrease. Its five-step smoke keeps the fifth step at `1e-4` rather than collapsing,
    with energy `6.58e-4 Ha`, force `0.021631 Ha/Bohr`, Hessian `2.71384`, scalar symmetry, and
    `6487 MiB` peak GPU. Smoke summary/checkpoint/resource hashes are
    `9fb08a3b7c0dbd335cbf0d653393f4226e756a7910d32fef7c0fae482475448c`,
    `6a3a22dff28117c0bd9c1641c6d577350818be143fd6c14456715dab519816cb`, and
    `02c567296692c847b54d7763cd35dc4261e97ef1a9be29b7f23d8b2d238f30d2`.
    Its node01 formal job 3942 was stopped at step 100 after a decisive negative trend: energy is
    `1.31e-4 Ha` and force improves to `0.020958 Ha/Bohr`, while Hessian worsens monotonically
    from `2.71386` through `2.71883` to `2.72298`. The accepted step remains `1e-4`, so this is
    not another line-search collapse; single-direction updates improve their sampled objective but
    interfere in the assembled full matrix. Step-100 checkpoint/training-metrics hashes are
    `1a00348613d11046c58fe6684683ff51866a1794fc95894c0d05cc3aca7408ed` and
    `cf4c2cdb88451e07564a1441007d6a7f1abbf87e57775ae35095cf4e370acd44`.
216. The width-8192 joint scalar-readout restart is final after 800 additional, 1000 cumulative
    exact-linear CGLS iterations. On `0028399`, alpha-1 scalar evaluation exactly matches the
    linear result: energy error `2.14e-5 Ha`, force MAE `4.686e-4 Ha/Bohr`, and Hessian relative
    Frobenius `0.22179`, with `1.39e-15` asymmetry. The normalized residual falls to `0.22232`
    (`5.14%` of the original pre-restart initial residual), but final normal-gradient norm is still
    `0.514`; therefore `0.22179` is a 1000-iteration practical limit, not a mathematical feature-
    rank lower bound. At `5572.9 s`, further CGLS is not competitive with the 5% target. Summary/
    parameter-step/metrics/resource hashes are
    `f1a13676d235a4eb7375578a3ddd0128dec7c2e73f1ee6858539f07c47ac6a2b`,
    `fd13fe250c5af1ef5e569776d35c9a1785e329710b2034dd8e0cf7a5d2980f6f`,
    `ba4c26fc39fe24c529ce71504c367f6759310c5f0217ee9a6a1eefbeda6eecbd`, and
    `d4668b48b8f4ca83ad5b839c8a8b443be7f19c56bec8e0da811b525891eaea0d`.
217. The force-secant path now supports one base E/F graph plus a dynamically sampled set of
    unique Cartesian directions whose HVP losses are averaged before balancing. A frozen audit of
    eight six-direction groups measures median E/F/HVP gradient norms
    `42.1838/3.01459/6.09932`; force-HVP cosine is negative in all groups, median/range
    `-0.3485/[-0.4965,-0.1572]`. Audit wall time is `93.2 s`, peak GPU allocation
    `27.72 GiB`, and summary/per-group/resource hashes are
    `e698a509a4283baaf172ede6848381dd1cc81f39338f9e2ea22a14e86bfccea3`,
    `4a965b9b95303d4adddd0802c5ac381ab3b9b4c118953fe1900460ee7c32d5ef`, and
    `db2976d8f3b3930fa4780282cb6e1857dba5f2dcd6b464fde00d0897f479e775`.
    The v6t smoke uses six unique directions per step and the stricter `1e-3 Ha` energy budget.
    After two steps, energy is `9.47e-5 Ha`, force MAE `0.021635 Ha/Bohr`, and Hessian improves
    slightly from `2.713856` to `2.713716`; peak GPU is `27.56 GiB`. Smoke summary/checkpoint/
    resource hashes are `a775f85203320a9c14c4a4435ce1c811ae925b38fdd87c3e06517ea685941755`,
    `4ed5779bf0b31253a3b67599b60a035fc90a996e4ae727d35a4f958176eed58c`, and
    `7db9737e47c2d332b30a73d168e12bcb51ac6b799f2909efa147ff2ff7486a2a`.
    Node01 job 3945 was stopped at step 100 after the full-matrix trajectory failed to improve:
    relative Frobenius was `2.71386/2.71435/2.71494/2.71436` at steps `0/25/50/100`, while force
    improved to `0.02081 Ha/Bohr`. Its step-100 checkpoint and metrics hashes are
    `a64b67bd25ce1b079a65dc2d7ae14506ba560b52281941846eb2168f98814865` and
    `d36905f46fb5928eb1151d0440a555f3be4b6395589e05604358231614671212`. Six-direction sampling
    therefore does not remove cross-update matrix interference.
218. The replacement v6u objective covers all 45 Cartesian basis directions on every update while
    accumulating one directional parameter gradient at a time. Its frozen full-basis audit has
    E/F/HVP gradient norms `42.18383/3.01459/4.36333`, E-HVP cosine `0.42490`, and force-HVP
    cosine `-0.45473`. Wall time is `85.59 s`, peak GPU allocation `3.89 GiB`, and summary/CSV/
    resource hashes are
    `15555d8d90841c04f67227834d9b2936f116913250d31df39170dd9b984f39ff`,
    `42d1852312a12f8dfbcbaac85a3e768c078a2893cfb0b0e62ebff369a5444847`, and
    `43579470d57bdb0010f3e16ab3c915e2fbbb8314d989267ca340835e554b4634`.
    The one-step smoke accepts the full `1e-4` step and jointly lowers energy, force, full-basis HVP
    loss, and evaluated full-Hessian error; energy/force/Hessian finish at
    `0.001803 Ha/0.021648 Ha Bohr-1/2.713781`. The Hessian change is only `-7.46e-5`, so this is
    a valid mechanism smoke, not capacity evidence. Smoke summary/best-checkpoint/resource hashes
    are `2f22205337d745f1dcd59b73839350885f91bf48cd378b0dfeaf08f2ea2ab33c`,
    `ed4ac4c85ac7af995da81f44e9ba0338e23c80da8e54098e932027c90f2923c8`, and
    `bf1537ff2749d1e4e9dcf87551fafe08e039464dd69e01f110d3fed1dd304dac`.
    Node01 job 3949 was stopped after the preregistered step-5 slope decision. Energy/force/full
    Hessian relative Frobenius at step 5 are `0.0009402 Ha`, `0.0216112 Ha Bohr-1`, and
    `2.7136175`; the five-step Hessian gain is only `2.38e-4`. Reaching the 5% capacity target at
    this slope would require tens of thousands of full-basis steps, so more v6u learning-rate or
    direction micro-tuning is rejected. Best/last checkpoint, metrics, and per-parent hashes are
    `0b09f6866807f5526f0daac76d4460de98a65aed27f0055373b01f8713422218`,
    `c947cd3cd375ffb7c7a84c74d0dd3083aca60587f1f3b19565f7d332dd8cddfe`,
    `89511ea25a6b595eef41773835d77fc24b6578b126d1085567a1a63fef2f9af5`, and
    `1e8c5aab0c25040933b3f2872a440bf2e0802bc3ae5596fd2aef65ac0f04f4f6`. The MACE full-basis
    branch is closed; stable5 and every later tier remain closed from it.
219. A distinct train-only audit asks whether the scalar Rayleigh quotient
    `q(v)=v^T H v` is numerically usable when the full vector HVP fails second-response checks.
    Protocol `qm9_complete_total_relaxed_q_train100_v1` freezes the existing 100 train parents,
    400 directions, 1200 branch tasks, every input hash, a 5% family of scalar consistency gates,
    at least three eligible directions per parent, and a preregistered readiness target of 80
    parents. It does not relabel q as a vector HVP and does not access Test100. Node01 job 3950
    completed in 13 seconds using only existing artifacts: 74/100 parents pass the density/
    electronic-branch/KKT gate, 214/400 directions pass every q gate, and only 59 parents retain
    at least three directions. Therefore `train100_q_ready=false`; even deleting every q-direction
    check post hoc could recover at most the 74 electronically eligible parents, so the 80-parent
    gate is mathematically unreachable for this frozen selection. Among the 296 directions on
    parent-gate-passing molecules, 70 fail the `h=1e-3` energy-versus-force scalar closure, 23 fail
    small-step stability, and 13 fail implicit-versus-relaxed agreement; 44/70 closure failures
    are low-frequency directions whose PBE `|q|` median is only `2.45e-3 Ha Bohr-2`, consistent
    with energy second-difference cancellation. These facts motivate a separately preregistered
    replacement-parent screen or a clearly diagnostic 59-parent Stage-2.5 run; they do not permit
    retroactive gate relaxation or formal Stage-3 training. Summary/manifest/per-direction/
    per-parent/resource hashes are
    `64dfa9c686e5c032a1687bbb1b84b09920a41a63c23ccc1199a4107deb026d0c`,
    `6160c8d132d0642b0e7bf0e0feb6161f7f0cde8c1e65dc203754ae41e0f8fab8`,
    `4cfa76abc98a5048395206e90d5a1bd06d1b2a20573aa8c4d202aff0f346a8fe`,
    `a9c6e5114ee9049c297d8db9e346b420ea38359ad29fbc9410f4e3b678225f71`, and
    `7d74ce875b43a20839dc5179c8e2ec254faca5cbffceaf4ee96b04bb9bffb5a5`.
220. The fail-closed successor is explicitly Stage 2.5, not a renamed Stage 3. A hash-only split
    first corrected the two direction-count scopes: the v1 audit has 214 eligible directions over
    all parent-gate-passing molecules, while the 59 parents with at least three directions contain
    196 directions. Job 3951 correctly failed on the original ambiguous count before producing a
    manifest. Corrected job 3952 freezes 137 training and 59 one-per-parent held-out directions by
    a model-independent SHA256 rule; held-out kinds are 23 angle bends, 17 bond stretches, 14
    random internal, and 5 low-frequency directions. Split manifest/summary/resource hashes are
    `c514c7fc3a550b0d5e19ddb0e35ad0d72908d627eb5523ff33be34625c3a5043`,
    `4eefcd24df2684949613d0dc18aa241b0c33e593e6ab3f1af561fe1352e63b18`, and
    `2bd8db3657da06cdc11c4827f681314bc5b42b411a48d8df3377928d93ef382f`.
    The corresponding local random-feature scalar uses one scalar for E/F/q, an absolute+relative
    q loss with the frozen `0.1 Ha Bohr-2` floor, and a preregistered five-value dual-ridge grid.
    It includes base E/F for all 59 parents but deliberately does not yet claim the 3200-geometry
    train800 replay; a pass can authorize only replay engineering and strict full-Hessian checks.
    Preflight job 3953 validates all parent hashes and geometries, 12--27 atoms, 42,027 global
    features, and 11,361--30,843 active local features per parent. Preflight/resource hashes are
    `91c4a3d950fb0a69d034bf2c0bf858d0f05d240d869a6616a7cc82cf5d7d02fe` and
    `c7e5180a149242159011e5086ac99fdff3e3e664779182e266edac4b464286f8`.
    During preflight review, `_load_kernel_checkpoint()` was found truncated by an earlier function
    insertion; its hash/protocol check and four-tensor float64 return are restored and covered by
    two regression tests. Thirteen local-scalar/Stage-2 tests pass. Node01 array 3954 builds 59
    exact feature jets on at most eight A100s. Default-chunk task 31 (`0059755`, 27 atoms) failed
    after 29 seconds with a deterministic GPU OOM: 67.16 GiB was resident and the next allocation
    requested 31.14 GiB. This is an implementation chunk limit, not a non-finite derivative.
    Fit 3955 and analysis 3969 were cancelled so the failed array cannot leak into a partial fit.
    A tested rescue CLI only permits reducing the feature chunk; job 3992 reruns index 31 with
    chunk 8. The only other 27-atom parent, index 47, was removed from the default array before
    execution and job 3995 also uses chunk 8. Every existing artifact is retained. A new fit is
    submitted only after all 59 hash-bound artifacts exist and both rescues pass. No job can
    access validation parents or Test100.
221. Replacement-parent membership is frozen before the Stage-2.5 model result is available.
    Metadata-only job 3972 maps each of the 26 original parent-gate failures to two unique
    candidates from the unlabelled train700 inventory. Ranking first preserves composition class,
    then minimizes natoms-bin, difficulty-bin, and atom-count distance, with a fixed SHA256
    tiebreak. The 52-parent pool has 47 exact-stratum and five preregistered nearest-stratum
    fallbacks. Manifest/summary/resource hashes are
    `a2f33838008323b66b33894d1e192c5150f244b095cfbf44c4555e61f19601da`,
    `8f2947b1653a0bde11182f0d253d181b9ec204cfdcb91b33996b945e6e41d892`, and
    `3bd183d469e14e561c01c2cbde38619ae24f203772193a9e4987372499f5656f`.
    `labels_authorized=false`: no replacement PBE Hessian or q work may start unless the frozen
    Stage-2.5 result first justifies that cost. A separate read-only post-fit analysis reports
    source/candidate q errors by role and direction kind, paired win fraction, and the largest
    required corrections; it cannot change ridge selection. Original dependency job 3969 was
    cancelled with fit 3955 after the chunk-size OOM and must be resubmitted after artifact audit.
222. The Stage-2.5 relaxed-q run is complete and fails its frozen decision. Exact jets exist for
    all 59 parents after chunk-8 rescues for indices 31, 44, and 47. The hash/shape/finite/relative-
    symmetry audit passes with manifest
    `061a2752ae603bae334bc0c4f6f6e90097e05a79142af34f2f9858681947ecfc` and maximum feature
    asymmetry Frobenius ratio `1.66e-13`. The selected `ridge=1e-4` fit interpolates train q
    (median/P90 `4.51e-5/1.34e-3`) but fails held q by orders of magnitude
    (median/P90 `3.777/16.524`, only `3/59 <=0.15`). Held low-frequency directions are worst at
    median `7.33`; only 32.2% of all held directions improve over source. Fit/analysis hashes are
    `e9d9b1c5f8e52264b07d54563127ebf4025e05d050f08b214487f84e76d275ae` and
    `11bc52cebf714dc9d5b3c64f4e9f7ff817bfbe11dea5a9d2fe02ac9ffced2e20`.
    This proves scalar-q underdetermination/direction memory, not useful unseen-direction transfer.
    Replay, replacement labels, full Hessians, validation, and Test100 remain closed.
223. Provenance review further finds a same-scalar violation in that diagnostic: the 1,200 raw
    q/HVP tasks use EGF epoch-9 checkpoint hash `722afe50...d5f1f96`, while the pointwise E/F
    arrays use A seed-314159 checkpoint hash `e6516b04...f9d9bc`. The failed held result remains
    informative, but the assembled E/F/q labels cannot support a physical one-source-scalar claim.
    Existing raw arrays retain complete branch-resolved relaxed vector HVPs, PBE HVPs, directions,
    and base densities. A separately frozen approximately 20-parent vector-HVP diagnostic can
    repair both defects without new PBE labels: select only vector-stable directions, recompute
    base E/F from the same epoch-9 scalar, and use `3N` vector constraints per direction. It must
    not reuse the exposed q-held metrics for selection.

The v6f frozen-readout implementation is contained in:

```text
mldft/ml/models/components/local_mace_scalar_residual.py
mldft/ml/models/components/local_mace_invariant_readout.py
scripts/qm9_complete_total_local_scalar_full_hessian_capacity.py
scripts/qm9_complete_total_local_scalar_jacobian_range_audit.py
configs/audit/qm9_complete_total_hessian_mace_frozen_invariant_readout_smoke_v6f.yaml
scripts/slurm_qm9_complete_total_mace_frozen_invariant_readout_smoke_v6f.sbatch
scripts/slurm_qm9_complete_total_mace_frozen_invariant_readout_formal_v6f.sbatch
scripts/slurm_qm9_complete_total_mace_frozen_invariant_readout_cgls_v6f.sbatch
configs/audit/qm9_complete_total_hessian_mace_power_readout_smoke_v6g.yaml
scripts/slurm_qm9_complete_total_mace_power_readout_smoke_v6g.sbatch
scripts/slurm_qm9_complete_total_mace_power_readout_cgls_v6g.sbatch
configs/audit/qm9_complete_total_hessian_mace_power_readout_width8192_smoke_v6h.yaml
configs/audit/qm9_complete_total_hessian_mace_power_readout_joint_jet_v6j.yaml
scripts/slurm_qm9_complete_total_mace_power_readout_width8192_one_parent_cgls_v6h.sbatch
scripts/slurm_qm9_complete_total_mace_power_readout_width8192_one_parent_joint_cgls_v6j.sbatch
configs/audit/qm9_complete_total_hessian_mace_power_readout_joint_jet_restart_v6m.yaml
scripts/slurm_qm9_complete_total_mace_power_readout_joint_cgls_restart_v6m.sbatch
scripts/qm9_complete_total_mace_force_secant_capacity.py
scripts/qm9_complete_total_mace_force_secant_gradient_audit.py
scripts/qm9_complete_total_mace_checkpoint_hessian_geometry.py
configs/audit/qm9_complete_total_hessian_mace_force_secant_h64_v6k.yaml
configs/audit/qm9_complete_total_hessian_mace_force_secant_h64_pilot_v6n.yaml
configs/audit/qm9_complete_total_hessian_mace_force_secant_h64_pcgrad_v6p.yaml
configs/audit/qm9_complete_total_hessian_mace_force_secant_h64_pcgrad_sgd_v6q.yaml
configs/audit/qm9_complete_total_hessian_mace_force_secant_h64_pcgrad_linesearch_v6r.yaml
configs/audit/qm9_complete_total_hessian_mace_force_secant_h64_pcgrad_budget_v6s.yaml
configs/audit/qm9_complete_total_hessian_mace_force_secant_multi6_v6t.yaml
configs/audit/qm9_complete_total_hessian_mace_force_secant_fullbasis_v6u.yaml
scripts/slurm_qm9_complete_total_mace_force_secant_h64_smoke_v6k.sbatch
scripts/slurm_qm9_complete_total_mace_force_secant_h64_pilot_v6n.sbatch
scripts/slurm_qm9_complete_total_mace_force_secant_gradient_audit_v6n.sbatch
scripts/slurm_qm9_complete_total_mace_force_secant_pcgrad_linesearch_smoke_v6r.sbatch
scripts/slurm_qm9_complete_total_mace_force_secant_pcgrad_linesearch_v6r.sbatch
scripts/slurm_qm9_complete_total_mace_force_secant_pcgrad_budget_smoke_v6s.sbatch
scripts/slurm_qm9_complete_total_mace_force_secant_pcgrad_budget_v6s.sbatch
scripts/slurm_qm9_complete_total_mace_checkpoint_hessian_geometry_v6r.sbatch
scripts/slurm_qm9_complete_total_mace_force_secant_multi6_gradient_audit_v6t.sbatch
scripts/slurm_qm9_complete_total_mace_force_secant_multi6_smoke_v6t.sbatch
scripts/slurm_qm9_complete_total_mace_force_secant_multi6_v6t.sbatch
scripts/slurm_qm9_complete_total_mace_force_secant_fullbasis_gradient_audit_v6u.sbatch
scripts/slurm_qm9_complete_total_mace_force_secant_fullbasis_smoke_v6u.sbatch
scripts/slurm_qm9_complete_total_mace_force_secant_fullbasis_v6u.sbatch
tests/test_qm9_complete_total_mace_force_secant_capacity.py
tests/test_qm9_complete_total_mace_force_secant_gradient_audit.py
tests/test_qm9_complete_total_mace_checkpoint_hessian_geometry.py
tests/ml/test_local_mace_scalar_residual.py
tests/test_qm9_complete_total_local_scalar_jacobian_range_audit.py
```

New code and artifacts include:

```text
scripts/qm9_complete_total_local_random_feature_stage2.py
scripts/qm9_complete_total_local_random_feature_stage2_ridge_analysis.py
scripts/qm9_complete_total_local_random_feature_stage2_width_analysis.py
scripts/qm9_complete_total_local_random_feature_stage2_direction_coverage_analysis.py
scripts/qm9_complete_total_local_random_feature_stage2_hashed_subspace_analysis.py
scripts/qm9_complete_total_structured_scalar_subspace.py
scripts/qm9_complete_total_structured_subspace_analysis.py
scripts/slurm_qm9_complete_total_local_random_feature_stage2_prepare_v10.sbatch
scripts/slurm_qm9_complete_total_local_random_feature_stage2_jets_v10.sbatch
scripts/slurm_qm9_complete_total_local_random_feature_stage2_fit_v10.sbatch
scripts/slurm_qm9_complete_total_local_random_feature_stage2_ridge_diagnostic_v1.sbatch
scripts/slurm_qm9_complete_total_local_random_feature_stage2_ridge_analysis_v1.sbatch
scripts/slurm_qm9_complete_total_local_random_feature_stage2_width_diagnostic_v1.sbatch
scripts/slurm_qm9_complete_total_local_random_feature_stage2_width_analysis_v1.sbatch
scripts/slurm_qm9_complete_total_local_random_feature_stage2_direction_coverage_v1.sbatch
scripts/slurm_qm9_complete_total_local_random_feature_stage2_direction_coverage_analysis_v1.sbatch
scripts/slurm_qm9_complete_total_local_random_feature_train20_full_capacity_v1.sbatch
scripts/slurm_qm9_complete_total_local_random_feature_stage2_hashed_subspace_v1.sbatch
scripts/slurm_qm9_complete_total_local_random_feature_stage2_hashed_subspace_analysis_v1.sbatch
scripts/slurm_qm9_complete_total_structured_subspace_v1.sbatch
scripts/slurm_qm9_complete_total_structured_subspace_analysis_v1.sbatch
scripts/slurm_qm9_complete_total_v9_prior_only_v1.sbatch
scripts/slurm_qm9_complete_total_bounded_equivariant_capacity_v1.sbatch
scripts/slurm_qm9_complete_total_bounded_equivariant_width_depth_v2.sbatch
scripts/slurm_qm9_complete_total_bounded_equivariant_h64_formal_v2.sbatch
scripts/slurm_qm9_complete_total_bounded_equivariant_lbfgs_v1.sbatch
scripts/slurm_qm9_complete_total_bounded_equivariant_one_parent_lbfgs_v1.sbatch
scripts/slurm_qm9_complete_total_bounded_equivariant_long_range_v3.sbatch
scripts/slurm_qm9_complete_total_bounded_equivariant_long_range_formal_v3.sbatch
scripts/slurm_qm9_complete_total_bounded_equivariant_long_range_lbfgs_v1.sbatch
scripts/slurm_qm9_complete_total_tensor_equivariant_capacity_v4.sbatch
scripts/slurm_qm9_complete_total_tensor_equivariant_scaled_v4b.sbatch
scripts/slurm_qm9_complete_total_tensor_equivariant_half_width_v4c.sbatch
scripts/slurm_qm9_complete_total_tensor_equivariant_half_width_formal_v4c.sbatch
configs/audit/qm9_complete_total_hessian_local_random_feature_stage2_v10.yaml
configs/audit/qm9_complete_total_hessian_local_random_feature_stage2_ridge_diagnostic_v1.yaml
configs/audit/qm9_complete_total_hessian_local_random_feature_stage2_width_diagnostic_v1.yaml
configs/audit/qm9_complete_total_hessian_local_random_feature_stage2_direction_coverage_v1.yaml
configs/audit/qm9_complete_total_hessian_local_random_feature_train20_full_capacity_v1.yaml
configs/audit/qm9_complete_total_hessian_local_random_feature_stage2_hashed_subspace_v1.yaml
configs/audit/qm9_complete_total_hessian_structured_subspace_path_v1.yaml
configs/audit/qm9_complete_total_hessian_bounded_equivariant_capacity_v1.yaml
configs/audit/qm9_complete_total_hessian_bounded_equivariant_width_depth_v2.yaml
configs/audit/qm9_complete_total_bounded_equivariant_lbfgs_v1.json
configs/audit/qm9_complete_total_bounded_equivariant_one_parent_lbfgs_v1.json
configs/audit/qm9_complete_total_hessian_bounded_equivariant_long_range_v3.yaml
configs/audit/qm9_complete_total_bounded_equivariant_long_range_lbfgs_v1.json
configs/audit/qm9_complete_total_hessian_tensor_equivariant_capacity_v4.yaml
configs/audit/qm9_complete_total_hessian_tensor_equivariant_scaled_v4b.yaml
configs/audit/qm9_complete_total_hessian_tensor_equivariant_half_width_v4c.yaml
mldft/ml/models/components/local_tensor_equivariant_scalar_residual.py
tests/ml/test_local_tensor_equivariant_scalar_residual.py
tests/test_qm9_complete_total_local_random_feature_stage2.py
tests/test_qm9_complete_total_local_random_feature_stage2_ridge_analysis.py
tests/test_qm9_complete_total_local_random_feature_stage2_width_analysis.py
tests/test_qm9_complete_total_local_random_feature_stage2_direction_coverage_analysis.py
tests/test_qm9_complete_total_local_random_feature_stage2_hashed_subspace_analysis.py
tests/test_qm9_complete_total_structured_subspace_analysis.py
tests/ml/test_local_equivariant_scalar_residual.py
scripts/qm9_complete_total_build_five_parent_baseline_manifest.py
scripts/qm9_complete_total_geometry_shared_capacity.py
scripts/qm9_complete_total_geometry_shared_resolve.py
scripts/qm9_complete_total_geometry_mlp_capacity.py
scripts/qm9_complete_total_geometry_mlp_external_eval.py
scripts/qm9_complete_total_geometry_parent_cv_analysis.py
scripts/slurm_qm9_complete_total_geometry_parent_cv.sbatch
scripts/slurm_qm9_complete_total_geometry_parent_cv_analysis.sbatch
configs/audit/qm9_complete_total_hessian_geometry_parent_cv_v1.yaml
tests/test_qm9_complete_total_geometry_parent_cv_analysis.py
mldft/ml/models/components/local_message_passing_residual.py
scripts/qm9_complete_total_local_scalar_full_hessian_capacity.py
scripts/slurm_qm9_complete_total_local_scalar_capacity.sbatch
scripts/qm9_complete_total_local_scalar_capacity_analysis.py
scripts/slurm_qm9_complete_total_local_scalar_capacity_analysis.sbatch
scripts/qm9_complete_total_local_scalar_rigid_mode_audit.py
scripts/qm9_complete_total_local_scalar_jet_compatibility_audit.py
scripts/qm9_complete_total_local_scalar_jacobian_range_audit.py
scripts/slurm_qm9_complete_total_local_scalar_jacobian_range_audit.sbatch
scripts/qm9_complete_total_freeze_local_scalar_post_v4_diagnostics.py
scripts/slurm_qm9_complete_total_local_scalar_post_v4_diagnostics.sbatch
scripts/qm9_complete_total_local_scalar_post_v4_diagnostics_analysis.py
scripts/slurm_qm9_complete_total_local_scalar_post_v4_diagnostics_analysis.sbatch
configs/audit/qm9_complete_total_hessian_local_scalar_capacity_v1.yaml
configs/audit/qm9_complete_total_hessian_local_scalar_joint_capacity_v2.yaml
configs/audit/qm9_complete_total_hessian_local_scalar_joint_capacity_v2_gradnorm.yaml
configs/audit/qm9_complete_total_hessian_local_scalar_activation_audit_v3.yaml
tests/ml/test_local_message_passing_residual.py
tests/test_qm9_complete_total_local_scalar_full_hessian_capacity.py
tests/test_qm9_complete_total_local_scalar_capacity_analysis.py
tests/test_qm9_complete_total_local_scalar_jet_compatibility_audit.py
tests/test_qm9_complete_total_local_scalar_jacobian_range_audit.py
tests/test_qm9_complete_total_freeze_local_scalar_post_v4_diagnostics.py
tests/test_qm9_complete_total_local_scalar_post_v4_diagnostics_analysis.py
scripts/slurm_qm9_complete_total_local_scalar_joint_capacity.sbatch
scripts/slurm_qm9_complete_total_local_scalar_joint_gradnorm_capacity.sbatch
scripts/slurm_qm9_complete_total_local_scalar_activation_audit.sbatch
scripts/qm9_complete_total_local_scalar_gradient_conflict_audit.py
scripts/slurm_qm9_complete_total_local_scalar_gradient_conflict_audit.sbatch
tests/test_qm9_complete_total_local_scalar_gradient_conflict_audit.py
scripts/qm9_complete_total_local_scalar_activation_smoke_analysis.py
tests/test_qm9_complete_total_local_scalar_activation_smoke_analysis.py
scripts/qm9_complete_total_capacity_candidate_select.py
scripts/qm9_complete_total_compare_geometry_mlp_runs.py
scripts/qm9_complete_total_external_scaling_analysis.py
scripts/qm9_complete_total_replay_descriptor_cache.py
scripts/qm9_complete_total_replay_descriptor_cache_merge.py
scripts/qm9_complete_total_replay_transient_rescue.py
scripts/qm9_complete_total_stage3_training_preflight.py
scripts/prepare_qm9_complete_total_train800_feature_inventory.py
scripts/bind_qm9_complete_total_feature_inventory.py
scripts/expand_qm9_complete_total_checkpoint_to_feature_inventory.py
scripts/expand_qm9_complete_total_hidden_width.py
scripts/expand_qm9_complete_total_deep_residual.py
scripts/qm9_complete_total_local_body_order_capacity.py
scripts/slurm_qm9_complete_total_local_body_order_capacity.sbatch
mldft/ml/models/components/local_body_order_residual.py
mldft/ml/models/components/local_quadratic_residual.py
mldft/ml/models/components/spectral_hessian_residual.py
scripts/qm9_complete_total_anchored_linear_direction_capacity.py
scripts/qm9_complete_total_local_quadratic_capacity.py
scripts/qm9_complete_total_local_quadratic_network_capacity.py
scripts/qm9_complete_total_spectral_operator_capacity.py
scripts/qm9_complete_total_spectral_operator_external_eval.py
scripts/qm9_complete_total_spectral_operator_validation_analysis.py
scripts/qm9_complete_total_spectral_operator_parent_cv.py
scripts/qm9_complete_total_spectral_operator_parent_cv_analysis.py
scripts/slurm_qm9_complete_total_anchored_linear_direction_capacity.sbatch
scripts/slurm_qm9_complete_total_local_quadratic_capacity.sbatch
scripts/slurm_qm9_complete_total_local_quadratic_network_capacity.sbatch
scripts/slurm_qm9_complete_total_spectral_operator_capacity.sbatch
scripts/slurm_qm9_complete_total_spectral_operator_external_eval.sbatch
scripts/slurm_qm9_complete_total_spectral_operator_validation_analysis.sbatch
scripts/slurm_qm9_complete_total_spectral_operator_parent_cv.sbatch
scripts/slurm_qm9_complete_total_spectral_operator_parent_cv_analysis.sbatch
tests/ml/test_local_quadratic_residual.py
tests/ml/test_spectral_hessian_residual.py
tests/test_qm9_complete_total_anchored_linear_direction_capacity.py
tests/test_qm9_complete_total_local_quadratic_capacity.py
tests/test_qm9_complete_total_local_quadratic_network_capacity.py
tests/test_qm9_complete_total_spectral_operator_capacity.py
tests/test_qm9_complete_total_spectral_operator_external_eval.py
tests/test_qm9_complete_total_spectral_operator_validation_analysis.py
tests/test_qm9_complete_total_spectral_operator_parent_cv.py
tests/test_qm9_complete_total_spectral_operator_parent_cv_analysis.py
configs/audit/qm9_complete_total_hessian_parent_cv_bounded_operator_v1.yaml
configs/audit/qm9_complete_total_hessian_direction_generalization_v4_block_operator.yaml
configs/audit/qm9_complete_total_hessian_direction_generalization_v5_symmetric_completion.yaml
configs/audit/qm9_complete_total_hessian_unseen_parent_shared_operator_v1.yaml
scripts/prepare_qm9_complete_total_active_feature_schema.py
scripts/launch_qm9_complete_total_atom_extensive_capacity.sh
scripts/launch_qm9_complete_total_atom_extensive_floored_capacity.sh
scripts/launch_qm9_complete_total_atom_extensive_unit_capacity.sh
scripts/launch_qm9_complete_total_floored_capacity.sh
scripts/launch_qm9_complete_total_stage3_replay_train.sh
scripts/launch_qm9_complete_total_robust_stage2_assets.sh
scripts/launch_qm9_complete_total_train800_union_assets.sh
scripts/launch_qm9_complete_total_stable5_union_assets.sh
scripts/launch_qm9_complete_total_robust_replay_cache.sh
scripts/slurm_qm9_complete_total_bind_feature_inventory.sbatch
scripts/slurm_qm9_complete_total_expand_checkpoint.sbatch
scripts/slurm_qm9_complete_total_stage3_replay_train_submit.sbatch
scripts/slurm_qm9_complete_total_stage3_active_feature_schema.sbatch
scripts/slurm_qm9_complete_total_stage3_external_diagnostic.sbatch
scripts/slurm_qm9_complete_total_stage3_external_eval.sbatch
scripts/slurm_qm9_complete_total_stage3_external_vibrational.sbatch
scripts/slurm_qm9_complete_total_stage3_replay_descriptor_cache.sbatch
scripts/slurm_qm9_complete_total_stage3_replay_descriptor_cache_merge.sbatch
scripts/slurm_qm9_complete_total_stage3_replay_merge.sbatch
scripts/slurm_qm9_complete_total_stage3_replay_rescue.sbatch
scripts/slurm_qm9_complete_total_geometry_shared_capacity.sbatch
scripts/slurm_qm9_complete_total_geometry_shared_resolve.sbatch
scripts/slurm_qm9_complete_total_geometry_mlp_capacity.sbatch
tests/test_expand_qm9_complete_total_hidden_width.py
tests/test_expand_qm9_complete_total_deep_residual.py
tests/test_qm9_complete_total_compare_geometry_mlp_runs.py
configs/audit/qm9_complete_total_hessian_capacity_v1_stable5_v2.yaml
/scratch/xzh/models/complete_total_capacity/20260717/five_parent_shared_geometry_resolve
/scratch/xzh/models/complete_total_capacity/20260717/five_parent_geometry_mlp_h128_resume20k_h10_lr1e5
/scratch/xzh/models/complete_total_capacity/20260717/five_parent_geometry_mlp_stable5_h128_h10_s30000
/scratch/xzh/models/complete_total_capacity/20260717/stage2_direction_v1/shared_descriptor_design_v2
/scratch/xzh/models/complete_total_capacity/20260717/stage2_direction_v1/mlp_h128_warm_v2
/scratch/xzh/models/complete_total_capacity/20260717/stage2_direction_v1/mlp_h128_h10_v2
/scratch/xzh/models/complete_total_capacity/20260717/stage2_direction_v1/analysis_warm_v2
/scratch/xzh/models/complete_total_capacity/20260717/stage2_direction_v2_dense/candidate_direction_manifest.json
/scratch/xzh/models/complete_total_capacity/20260717/stage2_direction_v2_dense/mlp_h128_h10
/scratch/xzh/models/complete_total_capacity/20260717/stage2_direction_v1/analysis_final_v1
/scratch/xzh/models/complete_total_capacity/20260717/stage2_capacity20_full_hessian_h128_h10
/scratch/xzh/models/complete_total_capacity/20260717/stage2_direction_v3_near_full/candidate_direction_manifest.json
/scratch/xzh/models/complete_total_capacity/20260717/stage2_direction_v3_near_full/mlp_h128_h10
/scratch/xzh/models/complete_total_capacity/20260717/stage2_coverage_ladder_analysis
/scratch/xzh/models/complete_total_capacity/20260717/stage2_coverage_ladder_analysis_v3
/scratch/xzh/models/complete_total_capacity/20260717/stage2_direction_v3_near_full/mlp_h128_h10_spec1
/scratch/xzh/models/complete_total_capacity/20260717/stage2_direction_v3_near_full/mlp_h128_h10_tail20_w1_low
/scratch/xzh/models/complete_total_capacity/20260717/stage2_direction_confirmation_v1
```

## Reproduction

Prepare the frozen manifest:

```bash
cd /scratch/xzh/code/structures25
source /scratch/xzh/env.sh
bash scripts/launch_qm9_complete_total_capacity_stage1.sh smoke
```

The successful scalar-curvature direction audit used:

```bash
SMOKE_LR=3e-7 \
SMOKE_LAMBDA_E=0 SMOKE_LAMBDA_F=0 SMOKE_LAMBDA_RHO=0 \
SMOKE_LAMBDA_H=0 SMOKE_LAMBDA_Q=1 SMOKE_LAMBDA_SPEC=0 \
SMOKE_UNROLL_STEPS=0 \
sbatch --export=ALL,SMOKE_LR,SMOKE_LAMBDA_E,SMOKE_LAMBDA_F,SMOKE_LAMBDA_RHO,SMOKE_LAMBDA_H,SMOKE_LAMBDA_Q,SMOKE_LAMBDA_SPEC,SMOKE_UNROLL_STEPS \
  scripts/slurm_qm9_complete_total_capacity_smoke.sbatch
```

The replay-aware Stage-2 pilot and exact same-objective continuation use:

```bash
bash scripts/launch_qm9_complete_total_stage3_replay_train.sh pilot 20260720
bash scripts/launch_qm9_complete_total_stage3_replay_train.sh continue-pilot 20260720
```

The completed no-replay capacity control used:

```bash
bash scripts/launch_qm9_complete_total_stage3_replay_train.sh no-replay-control 20260720
```

The function-preserving h512 smoke/formal capacity profiles use:

```bash
bash scripts/launch_qm9_complete_total_stage3_replay_train.sh no-replay-h512-smoke 20260720
bash scripts/launch_qm9_complete_total_stage3_replay_train.sh no-replay-h512-control 20260720
bash scripts/launch_qm9_complete_total_stage3_replay_train.sh no-replay-h128-restart-control 20260720
bash scripts/launch_qm9_complete_total_stage3_replay_train.sh no-replay-h512-paired-smoke 20260720
bash scripts/launch_qm9_complete_total_stage3_replay_train.sh no-replay-h512-paired-control 20260720
bash scripts/launch_qm9_complete_total_stage3_replay_train.sh no-replay-deep-h256-smoke 20260720
bash scripts/launch_qm9_complete_total_stage3_replay_train.sh no-replay-deep-h256-control 20260720
```

The completed reference-anchored local torsion control was launched with:

```bash
SOURCE_RUN_DIR=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/robust_floor1_no_replay_h512_h1_seed20260720_s30000 \
OUTPUT_DIR=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/local_body_order_torsion_anchor_purehvp_alldirs_parent20_s20_lr1e4_tlr1e6_out1e1_v1 \
STEPS=20 LOG_INTERVAL=5 PARENT_BATCH_SIZE=20 DIRECTIONS_PER_PARENT=100 \
LEARNING_RATE=1e-4 TORSION_LEARNING_RATE=1e-6 CANCELING_OUTPUT=0.1 \
HVP_WARMUP_STEPS=0 LAMBDA_ENERGY=0 LAMBDA_FORCE=0 LAMBDA_HESSIAN=1 \
LAMBDA_SPECTRUM=0 USE_PCGRAD=0 USE_GRADNORM=0 \
ANCHOR_ENERGY_FORCE_AT_PARENT=1 \
sbatch scripts/slurm_qm9_complete_total_local_body_order_capacity.sbatch
```

The final corrected shared-coefficient-network control was launched with:

```bash
SOURCE_RUN_DIR=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/robust_floor1_no_replay_h512_h1_seed20260720_s30000 \
OUTPUT_DIR=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/local_quadratic_network_initfunc_h128_fullbatch_s1500_lr3e6_v1 \
STEPS=1500 LOG_INTERVAL=50 HIDDEN_SIZE=128 PARENT_BATCH_SIZE=20 \
DIRECTIONS_PER_PARENT=100 LEARNING_RATE=3e-6 COEFFICIENT_SCALE=0.1 \
sbatch scripts/slurm_qm9_complete_total_local_quadratic_network_capacity.sbatch
```

The two Stage-2 v5 spectral/block controls use the same frozen directions and differ only in
their source run:

```bash
SOURCE_RUN_DIR=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/robust_floor1_no_replay_h512_h1_seed20260720_s30000 \
OUTPUT_DIR=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/spectral_block_parent_conditioned_symmetric_completion_v5_v1 \
DIRECTION_MANIFEST=/scratch/xzh/models/complete_total_capacity/20260717/stage2_direction_v5_symmetric_completion/candidate_direction_manifest.json \
INCLUDE_BLOCK_BASIS=1 BLOCK_PARENT_CONDITIONING=1 EXACT_TRAIN_INTERPOLANT=1 \
sbatch --export=ALL,SOURCE_RUN_DIR,OUTPUT_DIR,DIRECTION_MANIFEST,INCLUDE_BLOCK_BASIS,BLOCK_PARENT_CONDITIONING,EXACT_TRAIN_INTERPOLANT \
  scripts/slurm_qm9_complete_total_spectral_operator_capacity.sbatch

SOURCE_RUN_DIR=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/robust_replay_train_h1_seed20260720_s5000_to_s30000 \
OUTPUT_DIR=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/spectral_block_parent_conditioned_symmetric_completion_replay_source_v5_v1 \
DIRECTION_MANIFEST=/scratch/xzh/models/complete_total_capacity/20260717/stage2_direction_v5_symmetric_completion/candidate_direction_manifest.json \
INCLUDE_BLOCK_BASIS=1 BLOCK_PARENT_CONDITIONING=1 EXACT_TRAIN_INTERPOLANT=1 \
sbatch --export=ALL,SOURCE_RUN_DIR,OUTPUT_DIR,DIRECTION_MANIFEST,INCLUDE_BLOCK_BASIS,BLOCK_PARENT_CONDITIONING,EXACT_TRAIN_INTERPOLANT \
  scripts/slurm_qm9_complete_total_spectral_operator_capacity.sbatch
```

The independent-parent run first generates each frozen source Hessian and then applies only the
shared coefficients. `--protocol` is supplied by the second wrapper and exact completion is
unavailable by construction:

```bash
sbatch --export=ALL,STAGE3_EXTERNAL_ACTIVE_SCHEMA=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/active_feature_schema_train800_union_floor1_v1/active_feature_schema_manifest.json,STAGE3_EXTERNAL_CHECKPOINT=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/robust_floor1_no_replay_h512_h1_seed20260720_s30000/best.ckpt,STAGE3_EXTERNAL_OUTPUT=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/validation_external_eval_h512_source_v1 \
  scripts/slurm_qm9_complete_total_stage3_external_eval.sbatch
sbatch scripts/slurm_qm9_complete_total_spectral_operator_external_eval.sbatch

SPECTRAL_EXTERNAL_SOURCE=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/validation_external_eval_replay_source_v1 \
SPECTRAL_EXTERNAL_CANDIDATE=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/spectral_block_parent_conditioned_symmetric_completion_replay_source_v5_v1 \
SPECTRAL_EXTERNAL_OUTPUT=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/validation_spectral_block_shared_operator_replay_source_v1 \
sbatch --export=ALL,SPECTRAL_EXTERNAL_SOURCE,SPECTRAL_EXTERNAL_CANDIDATE,SPECTRAL_EXTERNAL_OUTPUT \
  scripts/slurm_qm9_complete_total_spectral_operator_external_eval.sbatch

sbatch scripts/slurm_qm9_complete_total_spectral_operator_validation_analysis.sbatch
```

The train20 parent-heldout protocol uses the original-A source and one preregistered variant per
job. For example, arm B is reproduced with:

```bash
cd /scratch/xzh/code/structures25
source /scratch/xzh/env.sh
SOURCE_FROM_BASELINE=1 \
VARIANT_ID=B_tanh2_conditioned_unbounded \
PARENT_FEATURE_TRANSFORM=tanh PARENT_FEATURE_TRANSFORM_SCALE=2 \
BLOCK_PARENT_CONDITIONING=1 MAX_CORRECTION_TO_SOURCE=inf \
OUTPUT_DIR=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/parent_cv_original_A_v1/B_tanh2_conditioned_unbounded \
sbatch --export=ALL,SOURCE_FROM_BASELINE,VARIANT_ID,PARENT_FEATURE_TRANSFORM,PARENT_FEATURE_TRANSFORM_SCALE,BLOCK_PARENT_CONDITIONING,MAX_CORRECTION_TO_SOURCE,OUTPUT_DIR \
  scripts/slurm_qm9_complete_total_spectral_operator_parent_cv.sbatch
```

All seven variant definitions and gates are frozen in the protocol YAML. After their vibrational
postprocessing is present, the complete read-only summary is reproduced with:

```bash
sbatch scripts/slurm_qm9_complete_total_spectral_operator_parent_cv_analysis.sbatch
```

The completed G0 geometry-scalar parent-heldout diagnostic is reproduced with:

```bash
cd /scratch/xzh/code/structures25
source /scratch/xzh/env.sh
sbatch scripts/slurm_qm9_complete_total_geometry_parent_cv.sbatch
sbatch --dependency=afterok:<G0_ARRAY_JOB_ID> \
  scripts/slurm_qm9_complete_total_geometry_parent_cv_analysis.sbatch
```

Do not submit `GEOMETRY_PARENT_CV_VARIANT=G1_h128_fullh_replay`: G0 failed its frozen
transferability gates, so the protocol forbids that arm.

The stable5 local-scalar capacity smoke and formal profiles use the same frozen protocol:

```bash
cd /scratch/xzh/code/structures25
source /scratch/xzh/env.sh
sbatch scripts/slurm_qm9_complete_total_local_scalar_capacity.sbatch
LOCAL_SCALAR_RUN_MODE=formal \
sbatch --export=ALL,LOCAL_SCALAR_RUN_MODE \
  scripts/slurm_qm9_complete_total_local_scalar_capacity.sbatch
```

The unanchored same-scalar smoke was diagnostic only; its formal profile is forbidden because the
raw Hessian-task gradient was too weak. The superseding bounded-GradNorm run is reproduced with:

```bash
cd /scratch/xzh/code/structures25
source /scratch/xzh/env.sh
LOCAL_SCALAR_RUN_MODE=smoke \
LOCAL_SCALAR_PROTOCOL=configs/audit/qm9_complete_total_hessian_local_scalar_joint_capacity_v2_gradnorm.yaml \
LOCAL_SCALAR_OUTPUT_ROOT=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/local_scalar_joint_capacity_v2_gradnorm/smoke \
sbatch --export=ALL,LOCAL_SCALAR_RUN_MODE,LOCAL_SCALAR_PROTOCOL,LOCAL_SCALAR_OUTPUT_ROOT \
  scripts/slurm_qm9_complete_total_local_scalar_joint_gradnorm_capacity.sbatch

LOCAL_SCALAR_RUN_MODE=formal \
LOCAL_SCALAR_PROTOCOL=configs/audit/qm9_complete_total_hessian_local_scalar_joint_capacity_v2_gradnorm.yaml \
LOCAL_SCALAR_OUTPUT_ROOT=/scratch/xzh/models/complete_total_capacity/20260717/stage3_unseen_parent_v1/local_scalar_joint_capacity_v2_gradnorm/formal \
sbatch --export=ALL,LOCAL_SCALAR_RUN_MODE,LOCAL_SCALAR_PROTOCOL,LOCAL_SCALAR_OUTPUT_ROOT \
  scripts/slurm_qm9_complete_total_local_scalar_joint_gradnorm_capacity.sbatch
```

## Current Decision

Earlier label-conditioned matrix completion passed its labeled-parent capacity and unseen-direction
checks, but it did not establish an unseen-parent OFDFT functional. The shared part failed
independent-parent transfer, so that branch remains diagnostic only.

Stage 2 now passes only in its precise labeled-parent sense. The chemistry-conditioned
spectral/block operator plus exact symmetric HVP completion reaches all full, training-direction,
and held-direction gates on 20 parents, with full-Hessian median `0.0124`. Vibrational metrics on
those same labeled parents are also strong. Exact completion consumes each parent's HVP labels,
so this proves matrix-completion capacity and unseen-direction response, not unseen-molecule
generalization.

The shared part of that operator failed the preregistered seven-parent validation. Both no-replay
h512 and replay-aware source arms have `0/7` molecules at or below `0.15`; their median relative
Frobenius errors are `8.88` and `5.14`, respectively. Frequency MAE rises to `5769/6706 cm-1`.
The replay source preserves force accuracy, so this is not an E/F replay failure. It is an
unseen-parent curvature failure dominated by source-Hessian and parent-conditioning distribution
shift. Unbounded corrections amplify the hard case `0056113` by orders of magnitude.

The required train20-only spectral/operator five-fold parent-heldout audit is complete. Starting from the clean
original-A source, A/B improve `19/20` parents and reduce median relative Frobenius from `2.733` to
`0.320/0.354`, proving that transferable curvature signal exists. They still fail the tail and
coverage gates, with only `1/20` at or below `0.15` and a `19.86--22.06` hard-case error. Simple
caps avoid the explosion but regress the median to about `0.86`; no registered arm passes.

A second, independently frozen global geometry-scalar MLP confirms that this is a representation
failure rather than insufficient fit capacity. G0 fits each 16-parent fold to `2.1--2.9%`, but its
held-parent median/P90/max are `1.73/5.27/17.61`, with `0/20 <=0.15`; energy worsens 8.83-fold.
Oracle amplitude and simple A/B/G0 combinations do not repair the held response direction.

Therefore Stage 3 remains rejected. G1 replay was not launched. The anchored local-scalar L0/L1
replacement also failed its stable5 capacity gate at about `0.94` median relative Frobenius even
after 5000 steps; no parent-CV was opened. The failure is not just correction amplitude, and the
zero-force anchor leaves only a narrow mathematical margin at the requested 5% level.

Unanchored same-scalar J2 with bounded GradNorm and PCGrad passes median E/F and symmetry gates but
fails stable5 curvature at median/max `1.036/1.226`. Its separate task gradients show near-opposite
E-F and F-H directions and a much weaker Hessian Jacobian. More J2 steps and parent-CV are rejected.

Activation v3, three-task v4, the frozen Jacobian/LBFGS diagnostics, and explicit angular/body-order
v6--v8 established the route to v9. v9 is the first scalar representation to pass the complete
stable5 matrix and vibration ceiling, but its frozen train20 result now rejects promotion:
near-exact train-direction interpolation coexists with held-direction median `1.744` and complete
Hessian median `1.141`. A broad ridge path lowers the latter only to `0.365` and loses the train
gate. Random-width reduction does no better. Raising geometric train-direction coverage improves
the median but leaves a large tail and cannot fit all added directions below 5%. In contrast, the
all-label full20 upper bound reaches full-Hessian median/P90/max `0.0668/0.0873/0.1051`, proving
that the scalar feature span has useful 20-parent matrix capacity but no adequate partial-direction
inductive bias. This is not an unconverged linear solve, E/F forgetting, asymmetry, or a ridge-only
defect. Stage 3 remains rejected; no train100 expansion, independent validation read, new HVP
inventory, or Test100 access is allowed. The next train20-only model must learn a lower-dimensional
shared equivariant curvature structure, combine absolute/relative directional loss without
near-interpolation, and then pass a newly frozen direction confirmation. Test100 use remains
exactly zero.
