# QM9 Complete-Total Relaxed-q Stage-2.5 Diagnostic

Last updated: 2026-07-23 16:32 Asia/Singapore

## Scope

This is a train-only diagnostic inside the random1000 train800 split. It asks whether a shared
conservative scalar can learn numerically stable complete-total density-relaxed directional
curvature before spending resources on replacement labels or formal train100 training.

It does not authorize unseen-parent claims, validation-parent access, or Test100. Test100 access
and evaluation counts remain zero. The model has no force or Hessian head:

```text
E_total(R) = E_source,complete-total-relaxed(R) + c^T phi(R)
F(R)       = -dE_total/dR
q(v)       = v^T (d2E_total/dR2) v
```

`q(v)` is a scalar Rayleigh quotient. It must not be reported as a vector HVP or a complete
Hessian. Formal promotion still requires strict complete-total full Hessians, frequencies,
imaginary-mode checks, mode overlap, train800 energy/force replay, and independent parents.

## Eligibility Audit

The frozen v1 audit reads only the existing 100-train-parent/400-direction/1200-branch artifacts.
All scalar consistency gates use a `0.1 Ha/Bohr^2` denominator floor and a 5% threshold.

| Quantity | Result |
|---|---:|
| Source parents | 100 |
| Parent density/branch/KKT passes | 74 |
| Eligible q directions across parent-gate passes | 214 / 400 |
| Parents with at least 3 eligible directions | 59 |
| Eligible directions on those 59 parents | 196 |
| Preregistered readiness target | 80 parents |
| Formal train100 q ready | **No** |

Even deleting every q-direction check could recover at most 74 parents, so the 80-parent gate is
unreachable for the original frozen selection. The dominant direction diagnostic is the
`h=1e-3 Bohr` scalar energy/force closure: 70 directions fail it, including 44 low-frequency
directions. Small-step stability fails 23 and implicit/relaxed agreement fails 13. This is
reported as evidence of finite-energy second-difference cancellation, not used to relax v1 after
seeing the result.

Key artifacts:

| Artifact | SHA256 |
|---|---|
| Eligibility summary | `64dfa9c686e5c032a1687bbb1b84b09920a41a63c23ccc1199a4107deb026d0c` |
| Eligibility manifest | `6160c8d132d0642b0e7bf0e0feb6161f7f0cde8c1e65dc203754ae41e0f8fab8` |
| Per-direction CSV | `4cfa76abc98a5048395206e90d5a1bd06d1b2a20573aa8c4d202aff0f346a8fe` |
| Per-parent CSV | `a9c6e5114ee9049c297d8db9e346b420ea38359ad29fbc9410f4e3b678225f71` |

## Direction Split

Corrected job 3952 freezes one held direction per eligible parent by a SHA256 ranking fixed before
candidate feature jets exist. The remaining directions are fit labels.

| Role/kind | Count |
|---|---:|
| Train total | 137 |
| Train random internal / bond / angle / low-frequency | 42 / 36 / 36 / 23 |
| Held total | 59 |
| Held random internal / bond / angle / low-frequency | 14 / 17 / 23 / 5 |

Manifest SHA256 is
`c514c7fc3a550b0d5e19ddb0e35ad0d72908d627eb5523ff33be34625c3a5043`.
Failed job 3951 is retained as an audit trail: it rejected the ambiguous assumption that all 214
eligible directions belonged to the 59 selected parents and wrote no usable manifest.

## Source Baseline

Using the frozen `0.1 Ha/Bohr^2` floor, the source complete-total relaxed q error is:

| Split | Count | Median | P90 | Max | Fraction <= 0.15 |
|---|---:|---:|---:|---:|---:|
| Train directions | 137 | 2.2948 | 2.9895 | 8251.49 | 0% |
| Held directions | 59 | 2.2924 | 3.0224 | 6.1845 | 0% |

The largest required correction is train direction `0052472/d0`:
`q_source=-3926.0146`, `q_PBE=0.47585 Ha/Bohr^2`. It passes the preregistered numerical q gates and
therefore remains in the unbounded diagnostic. It must be reported separately and cannot be
silently removed after fitting.

## Scalar Model

Protocol `qm9_complete_total_relaxed_q_stage2p5_local_scalar_v1` freezes the v9 local nonlinear
random-feature representation, float64 exact feature jets, base energy/force labels for all 59
parents, an absolute+relative q objective, and ridge grid
`{1e-12,1e-10,1e-8,1e-6,1e-4}`. The solver is a column-normalized sample-space dual Cholesky solve.
Held q labels are excluded from fitting and used only in the preregistered ridge ordering.

This diagnostic does **not** yet include all 3200 train800 replay geometries. A passing result can
authorize a separately frozen replay implementation; it cannot be promoted directly.

Preflight job 3953 passed all 59 parents:

| Property | Result |
|---|---:|
| Atom range | 12--27 |
| Global feature count | 42,027 |
| Active local feature range | 11,361--30,843 |
| Validation/Test100 accessed | No / No |

Preflight SHA256 is
`91c4a3d950fb0a69d034bf2c0bf858d0f05d240d869a6616a7cc82cf5d7d02fe`.
During review, the truncated v9 kernel-checkpoint loader was repaired and covered by regression
tests. The relevant 13-test local-scalar subset passes.

## Completed Execution

| Job | Purpose | Dependency/status |
|---|---|---|
| 3954 | Default-chunk exact E/F/H feature jets | Finished with deterministic OOM at indices 31 and 44 |
| 3992 | Index 31 chunk-8 rescue | Completed |
| 3995 | Index 47 proactive chunk-8 run | Completed |
| 4007 | Index 44 chunk-8 rescue | Completed |
| 4018 | First inventory audit | Failed closed on legacy chunk metadata; no manifest |
| 4019 | Second inventory audit | Failed closed on an inappropriate absolute symmetry gate; no manifest |
| 4020 | Relative-symmetry inventory audit | Completed, 59/59 passed |
| 4021 | Five-ridge dual fit | Completed; diagnostic gate failed |
| 4022 | Read-only paired source/candidate analysis | Completed |
| 3955 | Original five-ridge dual fit | Cancelled after task 31 failed |
| 3969 | Original read-only analysis | Cancelled with old fit dependency |

Default tasks 31 (`0059755`, 27 atoms) and 44 (`0095800`, 23 atoms) failed before writing artifacts
with deterministic CUDA OOMs. Index 47 (`0102642`, 27 atoms) was proactively removed from the
default array. All three were recomputed with chunk 8; this changes only derivative batching, not
the feature definition, labels, or protocol. Exactly 59 `.pt` and 59 JSON summaries exist with no
missing, extra, or temporary files.

The first audit exposed legacy artifacts that predated payload-level chunk metadata. The compatible
path accepts missing payload metadata only when the summary is explicitly non-override; new/rescue
artifacts require exact payload/summary agreement. The second audit found a maximum absolute
transpose difference of `1.583e-8` in a feature Hessian. A complete read-only scale audit showed
this is float64 roundoff: maximum `||H_asym||F/||H_sym||F=1.660e-13` and maximum element-scale
ratio `2.574e-13`. The final relative gate is `1e-10`, much stricter than the project's `0.5%`
numerical self-consistency requirement. Audit job 4020 passed all hashes, identities, shapes,
finite checks, global-column checks, and relative symmetry checks.

| Audit artifact | SHA256 |
|---|---|
| Feature-jet manifest | `061a2752ae603bae334bc0c4f6f6e90097e05a79142af34f2f9858681947ecfc` |
| Audit summary | `015e7e8aac11f747107af4ac2002292c1f3f264abbea51c56aedf904cd50f544` |
| Audit resource record | `b438f6ae660dba96c1b9dd36f4c48bc56e558996fbb419a99f243dd2587c523c` |

The 59 GPU tasks used `81,998.5 GPU-task seconds` in aggregate, produced
`29,951,244,216` bytes, and reached a maximum recorded allocation of `71,840 MiB`. The audit took
149.9 seconds and 2.85 GiB RSS. Fit job 4021 took 272.2 seconds, 32.6 GiB RSS, and 2.13 GiB peak
GPU allocation.

## Fit Result

The fit has 3,561 weighted rows and 42,027 columns. Every ridge nearly or exactly interpolates the
137 training q directions, but none generalizes to the one frozen direction per parent:

| Ridge | Train median / P90 | Held median / P90 | Held <=15% | E median/source | F median/source |
|---:|---:|---:|---:|---:|---:|
| `1e-12` | `4.64e-8 / 2.54e-7` | `10.137 / 49.984` | `0/59` | `9.60e-7` | `2.37e-5` |
| `1e-10` | `1.20e-9 / 7.27e-9` | `8.613 / 57.569` | `1/59` | `2.05e-8` | `2.40e-5` |
| `1e-8` | `1.47e-8 / 2.64e-7` | `5.896 / 22.410` | `3/59` | `4.18e-8` | `2.60e-5` |
| `1e-6` | `9.81e-7 / 1.05e-5` | `4.704 / 18.089` | `2/59` | `2.07e-6` | `2.79e-4` |
| `1e-4` selected | `4.51e-5 / 1.34e-3` | `3.777 / 16.524` | `3/59` | `1.29e-4` | `1.63e-2` |

For the selected arm, only 32.2% of held directions improve over source. Held medians by kind are
`4.50` angle, `3.26` bond, `7.33` low-frequency, and `1.90` random internal. All 137 training
directions improve and are below 15%. This is a matrix-underdetermination/direction-memorization
failure, not lack of optimizer convergence, E/F forgetting, or Hessian asymmetry.

| Fit/analysis artifact | SHA256 |
|---|---|
| Fit summary | `e9d9b1c5f8e52264b07d54563127ebf4025e05d050f08b214487f84e76d275ae` |
| Selected checkpoint | `6e57a0d88ef2565d621d71215cdd8955ee509ccaf8e601a21f471e2f328aa1b8` |
| Analysis summary | `11bc52cebf714dc9d5b3c64f4e9f7ff817bfbe11dea5a9d2fe02ac9ffced2e20` |
| Paired direction CSV | `2e55750f6e41d4d76d8f51ff3f5063a318e39c7de5fd865dbc71e30830e1c97d` |

## Source-Provenance Audit

Post-fit provenance review found that all 1,200 relaxed-q/HVP raw tasks use
`qm9_random1000_egf_forcew1p0_e10_20260714_202203/epoch_009.ckpt`
(`722afe50...d5f1f96`), while the Stage-2.5 base E/F arrays use
`qm9_hvp_curvature_v1_A...seed314159_s1200/last.ckpt`
(`e6516b04...f9d9bc`). Thus the fitted pointwise E/F source and relaxed-q source are different
functions. The correction itself is a scalar, but the assembled E/F/q labels do not define one
common `E_source(R)`.

This does not explain away the severe held-direction failure, but it independently prevents this
run from supporting a same-scalar physical claim. A corrected successor must bind one checkpoint
hash to base E/F and relaxed vector HVP/secant labels. Existing raw arrays contain the required
branch-resolved relaxed HVP vectors and densities; no new PBE label is needed.

## Replacement Pool

Replacement membership is frozen before candidate metrics. Metadata-only job 3972 assigns two
unique train700 candidates to each of the 26 original parent-gate failures. The 52 candidates
contain 47 exact-stratum matches and five deterministic nearest-stratum fallbacks. Every fallback
preserves composition class; the smallest CH-only hard slot necessarily crosses atom-count bin
because its exact train700 stratum is empty.

Manifest SHA256 is
`a2f33838008323b66b33894d1e192c5150f244b095cfbf44c4555e61f19601da`.
It explicitly records `labels_authorized=false`; no replacement PBE Hessian or q label is being
generated.

## Decision Gate

The selected Stage-2.5 arm must satisfy all of:

- train q median <= 5% and P90 <= 15%;
- held q median <= 15%, P90 <= 20%, and at least 80% of held directions <= 15%;
- median energy and force error no worse than 1.05 times source;
- scalar correction-Hessian asymmetry below `5e-12`.

Passing only justifies implementing train800 replay and running strict full-Hessian checks on a
newly frozen train-only subset. Failure returns to target scaling, robust influence control,
representation capacity, or complete-total response semantics. It does not justify additional
ridge tuning on the exposed held directions.

The result is **failure**. Train800 replay, replacement-parent labels, full-Hessian promotion,
validation-parent evaluation, and Test100 remain unauthorized. The next train-only experiment is
a separately frozen approximately 20-parent vector-HVP/secant diagnostic using only directions
that passed the existing vector stability audit and a single checkpoint for source E/F/HVP.
Vector supervision supplies `3N` constraints per direction and directly tests whether scalar-q
underdetermination is the dominant blocker.
