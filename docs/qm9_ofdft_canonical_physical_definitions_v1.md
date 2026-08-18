# QM9 OFDFT Canonical Physical Definitions v1

Status: normative semantics specification; implementation remediation and
focused verification completed on 2026-08-03. No v4 training or performance
evaluation result is claimed by this document.

This document fixes the physical meaning of the active QM9 OFDFT E/G/F/H
mainline. It separates labels that identify the unknown functional at the PBE
label density from observables of the model's self-consistent potential-energy
surface. New runs must record this semantic version explicitly. Historical
v1--v3 runs retain their original definitions and are not retroactively
reinterpreted.

## 1. Notation and single scalar owner

Let

- `R in R^(3N)` be nuclear Cartesian coordinates in Bohr;
- `c in R^n_c` be the independent, untransformed auxiliary-density
  coefficients used by the OFDFT solver;
- `theta` be all trainable Graphformer parameters;
- `q(R)` be the electron-number covector in the same coefficient basis; and
- `N_e` be the required electron count.

The Graphformer owns one scalar learned contribution,

```text
A_theta(R,c) = E_theta^(Ts+Exc)(R,c).
```

The complete total scalar is

```text
E_total_theta(R,c)
  = A_theta(R,c)
  + E_H(R,c)
  + E_ext(R,c)
  + E_nn(R).
```

There is no independent force or Hessian owner in this mainline. Force and
Hessian values must be derivatives of this scalar.

The electron-number constraint and constrained Lagrangian are

```text
C(R,c) = q(R)^T c - N_e = 0,

L_theta(R,c,mu) = E_total_theta(R,c) + mu C(R,c).
```

The model-self-consistent density and relaxed potential-energy surface are

```text
(c_theta*(R), mu_theta*(R)):
    L_c = 0 and C = 0,

Ebar_theta(R) = E_total_theta(R,c_theta*(R)).
```

`Ebar_theta` is the production potential-energy surface. Its first and second
nuclear derivatives own the canonical physical force and Hessian.

The current analytic response implementation supports only a
geometry-independent electron-number covector in the untransformed coefficient
basis. It must fail closed when `q_R` or its required directional derivatives
are nonzero; formulas that omit constraint-geometry terms are not valid outside
that boundary.

## 2. Reference record and common label index

For each reference geometry, one index `k_ref` must be selected once and used
for all Structures25 functional labels:

```text
c_ref       = of_labels/spatial/coeffs[k_ref]
A_ref       = of_labels/energies/e_kin_plus_xc[k_ref]
g_A_ref     = of_labels/spatial/grad_kin_plus_xc[k_ref]
```

`k_ref` must be the converged final label-density record and must have a valid
energy label. Code must not combine `coeffs[-1]` with an energy selected from a
different iteration through a boolean mask. A missing valid common index is a
data-contract failure.

The same geometry also provides independently named physical references:

```text
E_KS_ref = converged PBE total energy
F_ref    = complete PBE relaxed nuclear force
H_ref    = complete PBE relaxed Cartesian nuclear Hessian
```

The PBE method, orbital basis, density-fitting setup, grid, convergence
tolerances, geometry, coordinate units, and artifact hashes must be bound in
the run manifest. These physical references are not interchangeable with the
learned `Ts+Exc` component labels.

At `c_ref`, define a representation-matched reference total energy,

```text
E_total_ref_matched(R,c_ref)
  = A_ref
  + E_H(R,c_ref)
  + E_ext(R,c_ref)
  + E_nn(R).
```

Then the identity

```text
E_total_theta(R,c_ref) - E_total_ref_matched(R,c_ref)
  = A_theta(R,c_ref) - A_ref
```

must hold numerically. The difference

```text
delta_E_reference_closure = E_total_ref_matched - E_KS_ref
```

is a reference-representation/density-fitting closure diagnostic. It must not
be silently folded into the Structures25 component-energy loss.

## 3. Density-gradient projection and reference closure

In the untransformed coefficient coordinates, define the electron-number
tangent projector

```text
P_q = I - q q^T / (q^T q).
```

Gradients are covectors and must be transformed consistently before this
projection. In particular, model autograd, `g_A_ref`, and `q` must refer to the
same independent coefficient representation. Projecting tensors taken from
different transformed bases is forbidden.

The canonical Structures25 G residual is

```text
r_G = P_q [grad_c A_theta(R,c_ref) - g_A_ref].
```

It is not defined as the model complete-total projected gradient with a zero
target. To audit when the two expressions are equivalent, record

```text
r_ref_closure
  = P_q [g_A_ref + grad_c E_H(R,c_ref) + grad_c E_ext(R,c_ref)],

r_total_label
  = P_q grad_c E_total_theta(R,c_ref).
```

The implementation must verify

```text
r_total_label = r_G + r_ref_closure
```

to numerical tolerance. Direct Structures25 gradient matching is equivalent
to enforcing complete-total stationarity at `c_ref` only when
`r_ref_closure` is negligible under a threshold frozen before formal
training. If that closure gate fails, a formal v4 run must not silently replace
one target with the other. The allowed remedies are to regenerate a mutually
consistent reference record or introduce a separately named, versioned label
correction after an explicit scientific decision.

The model-center stationarity residual

```text
r_star = P_q grad_c E_total_theta(R,c_theta*)
```

is a solver diagnostic and fail-closed gate. It is not a G label and must never
be appended to, averaged with, or weighted as `L_G`. Consequently, the G loss
cannot depend on the number of HVP probes evaluated in the same optimizer
step.

## 4. Canonical supervised quantities and losses

The physical residual is defined independently of the optimizer loss norm.
The following normalization is the canonical v4 default; any change requires
a new protocol value and must not inherit v4 loss weights without a new
gradient-scale calibration.

### 4.1 E: learned-component energy at the label density

The E supervision identifies the unknown learned scalar:

```text
Delta_A = A_theta(R,c_ref) - A_ref                       [Hartree]

L_E = abs(Delta_A) / s_E.
```

The equivalent matched-total expression from Section 2 may be used as a
numerical identity check. Comparing the label-point model total directly with
`E_KS_ref` is a different quantity and must not be logged as this `L_E`.

The physical relaxed total-energy error is evaluated separately:

```text
Delta_Ebar = Ebar_theta(R) - E_KS_ref                    [Hartree].
```

If a future protocol trains `Delta_Ebar`, that objective must be named
`L_E_relaxed_total`; it is not silently substituted for the Structures25 E
term.

### 4.2 G: learned-component density derivative at the label density

The canonical size-normalized G loss is

```text
L_G = mean_i [(r_G[i] / s_G)^2].
```

`r_G` has the units of Hartree per untransformed density coefficient. Raw
coefficient norm, RMS, maximum absolute component, `r_ref_closure`, and
`r_total_label` must be reported separately. A loss using a sum instead of a
mean changes molecule-size weighting and therefore constitutes a different
protocol.

The reference Euler-closure gate is applied to coefficient RMS, not the raw
Euclidean norm, so its scale does not grow trivially as `sqrt(n_c)`. The
train-only preflight for v4 parent `0016298`, performed before any v4 update,
found norm `5.4559623642e-5` over 1026 coefficients, RMS
`1.7033256463e-6`, and maximum absolute component `2.4421175846e-5`. The v4
RMS gate is frozen at `5e-6`; these values are data-contract diagnostics, not
training results and not G-loss samples.

The same label record has `abs(q^T c_ref-N_e)=1.0108199222e-3` electrons.
Because `A_ref` and `g_A_ref` are labels at that exact fitted coefficient
vector, v4 must not silently renormalize `c_ref` and then pretend the labels
belong to the modified point. Instead it records and gates the label-point
residual at `2e-3` electrons. This tolerance is only for the auxiliary E/G
identification point; the production model center `c_theta*` retains the
strict constrained-solver residual gate.

### 4.3 F: complete-total relaxed force

The canonical force is

```text
F_theta*(R)
  = -d Ebar_theta(R) / dR
  = -partial_R L_theta(R,c_theta*,mu_theta*)             [Hartree/Bohr].
```

The equality uses strict constrained stationarity. During parameter training,
the derivative of a force loss must include the parameter response of
`c_theta*(theta)`; evaluating a detached self-consistent density does not give
the full gradient of a relaxed-force objective.

Let

```text
a_F = mean_i abs(F_theta*[i] - F_ref[i]),
r_F = max(sqrt(mean_i F_ref[i]^2), f_floor).
```

The canonical force loss is

```text
L_F = (1-alpha_F) a_F/s_F + alpha_F a_F/r_F.
```

The protocol must call `alpha_F` the relative fraction; a field named
`absolute_relative_mix` is ambiguous and must not be used in new summaries.

The complete-total force evaluated while holding `c_ref` fixed may be retained
as `F_label_point_aux`. It is equal to the physical relaxed force only when the
model is stationary at that density. It is not the canonical F target and
cannot be merged with relaxed-force metrics.

### 4.4 H: complete-total relaxed internal Hessian

For a nuclear direction `v`, differentiate the constrained stationarity
conditions:

```text
[ L_cc   C_c^T ] [ c_v  ]   = -[ L_cR v ]
[ C_c      0   ] [ mu_v ]      [ C_R  v ].
```

The relaxed HVP is

```text
H_theta* v
  = L_RR v + L_Rc c_v + C_R^T mu_v                    [Hartree/Bohr^2].
```

Under the currently enforced `q_R=0` boundary, let `T` be an orthonormal basis
for the coefficient tangent space, `T^T q=0`. The equivalent Schur expression
is

```text
H_theta* v
  = E_RR v
  - E_Rc T (T^T E_cc T)^(-1) T^T E_cR v.
```

Writing an unconstrained `E_cc^(-1)` without `T` is only shorthand and is not
the normative formula.

Let `B in R^(d x 3N)` be a complete orthonormal Cartesian internal basis at the
reference geometry,

```text
B B^T = I_d,
P_R = B^T B,
d = 3N - r_external.
```

For nonlinear QM9 molecules `r_external=6`; linear molecules use their actual
rank. Define the symmetric physical reference before projection as

```text
H_ref_sym = (H_ref + H_ref^T)/2.
```

Reference antisymmetry must be reported as a source-quality diagnostic.

For one fresh Rademacher probe

```text
z in {-1,+1}^d,
v = B^T z,
||v||^2 = d,
```

the canonical training error is represented in the internal output basis:

```text
e_H(z) = B [H_theta* - H_ref_sym] B^T z.
```

The canonical loss is

```text
L_H(z) = ||e_H(z)||_2^2 / [d^2 s_H^2].
```

Because `E[z z^T]=I_d`,

```text
E_z L_H(z)
  = ||B (H_theta* - H_ref_sym) B^T||_F^2 / [d^2 s_H^2].
```

This is the required correspondence between stochastic H training and the
`d x d` internal-matrix element MSE used by full-Hessian evaluation. It is
norm-equivalent to the Cartesian two-sided operator because
`P_R=B^T B` and `B B^T=I_d`:

```text
||B Delta_H B^T||_F = ||P_R Delta_H P_R||_F.
```

Applying only the input/right projection trains a different operator and is
not canonical v4. Given a Cartesian HVP error, the output projection and loss
can be evaluated without materializing `P_R` by applying `B` to that error.

The present KKT/parameter-response graph is analytic, while directional second
integral derivatives are numerical. New results must therefore say
`semi-analytic relaxed-HVP`, not `fully analytic Hessian`, unless the integral
implementation is replaced and independently verified.

## 5. Training boundary

The v4 mainline combines two explicitly different, compatible supervision
points under the same scalar owner:

| quantity | evaluation point | role |
| --- | --- | --- |
| E | fixed PBE label density `c_ref` | identify `A_theta` |
| G | fixed PBE label density `c_ref` | identify `grad_c A_theta` modulo electron-number gauge |
| F | model-self-consistent density `c_theta*` | supervise the production relaxed PES |
| H | model-self-consistent density `c_theta*` | supervise curvature of the same production relaxed PES |

The default alternating objective is

```text
replay update: lambda_E L_E + lambda_G L_G + lambda_F L_F
H update:      lambda_H L_H.
```

Only the selected update family is backpropagated. Per-family loss values,
parameter-gradient norms, optimizer identity, and update kind must be logged
explicitly. A `total_loss` from an H-only step is not comparable with a
`total_loss` from a replay step.

Solver quantities such as center stationarity, KKT response residual,
constraint residual, predictor merit, corrector cycles, and density drift are
diagnostics/gates, never extra samples in E/G/F/H reductions. Loss weights are
valid only for the exact loss definitions and reductions recorded by the
protocol. v3 weights must be recalibrated before v4 training.
The H-weight calibration uses the directly differentiated aggregate replay
gradient

```text
||grad_theta(lambda_E L_E + lambda_G L_G + lambda_F L_F)||,
```

not `sqrt(||g_E||^2+||g_G||^2+||g_F||^2)`, because the latter discards
cross-term alignment and is not the norm of the replay update.
Formal and resume invocations must pass the v4 calibration artifact and its
SHA256 into the runner. The artifact must bind the canonical physical ID, the
exact protocol hash, zero-learning-rate training-curve and summary hashes,
canonical code provenance, and the requested formal `lambda_H`; launcher-only
validation is not sufficient.

## 6. Fail-closed preflight and runtime gates

Every formal v4 run must bind and check at least:

1. one valid common `k_ref` for `c_ref`, `A_ref`, and `g_A_ref`;
2. electron-number residual of `c_ref`;
3. the matched-total energy identity from Section 2;
4. `r_total_label = r_G + r_ref_closure` in one coefficient basis;
5. a protocol-frozen reference Euler-closure threshold chosen before formal
   training from a train-only audit;
6. strict model-center stationarity and electron-number constraint residuals;
7. KKT response stationarity and constraint residuals;
8. finite, nonzero HVP parameter gradients;
9. `B` orthonormality, internal/external overlap, and `P_R` idempotence;
10. finite and sufficiently symmetric PBE reference Hessians;
11. complete artifact, protocol, source-code, model-weight, label, direction,
    and environment provenance; and
12. frozen validation/Test100 access boundaries.

The existing numerical targets of `5e-9` for the density solver and `1e-8`
for the training-graph center and response residuals may be carried into a v4
protocol, but the protocol must record them rather than relying on defaults.
Reference-closure thresholds must be frozen only after the dedicated label
audit; they must not be selected after observing a v4 training outcome.

## 7. Evaluation boundary

Primary physical evaluation uses one strictly converged `c_theta*` at the
reference geometry and reports

```text
Ebar_theta versus E_KS_ref,
F_theta* versus F_ref,
H_int_theta = P_R H_theta* P_R,
H_int_ref   = P_R H_ref_sym P_R.
```

The primary Hessian relative Frobenius error is

```text
||H_int_theta - H_int_ref||_F / ||H_int_ref||_F.
```

Raw and symmetric errors and
`||antisym(H_int_theta)||_F / ||sym(H_int_theta)||_F` must all be retained.
The symmetric internal Hessian is used for mass weighting and vibrational
eigendecomposition. Frequency MAE/RMSE, imaginary-mode count, and mode overlap
are diagnostics at the supplied geometry. When the reference force is not
small or the reference itself has imaginary modes, the result must not be
described as a production equilibrium harmonic spectrum.

Hessian-derived frequencies and normal modes do not determine infrared
intensities. IR intensity requires a separately defined dipole derivative
`d mu_dipole/dQ`; it is outside E/G/F/H v1.

A strict Cartesian or directional finite difference of the model's relaxed
force is a closure audit of the semi-analytic HVP. It may be reported only when
actually run with converged displaced densities and a recorded step-size
stability check.

Functional-identification metrics at `c_ref` (`Delta_A`, `r_G`, label-point
force, energy/Euler closure) remain a separate table. They must not be merged
with self-consistent physical E/F/H metrics.

## 8. Version and compatibility policy

The implicit-pilot v1--v3 artifacts are immutable historical evidence. Their
fixed-label E/G/F replay, G-to-zero surrogate, stationarity averaging, and
right-projected H training semantics must remain documented as originally
executed. Existing checkpoints, CSV files, summaries, hashes, and conclusions
must not be overwritten or relabeled as canonical v4.

The first v4 controlled run must:

- initialize model weights from the same released article Graphformer
  checkpoint used by the controlled historical baseline;
- create fresh replay and H optimizer states;
- create fresh density/trust/corrector state;
- prohibit resume of a v1--v3 capacity optimizer;
- record the canonical semantic version and its document/protocol hashes; and
- use new v4 output directories, summaries, and checkpoint provenance.

A legacy checkpoint may be examined only as an explicitly named weight-only
secondary comparison. It is not the v4 main trajectory. Loss curves, component
values, and aggregate `total_loss` from legacy and v4 semantics are not
numerically comparable. Hessian metrics may be placed in the same report only
when recomputed by the same canonical evaluator and visibly labeled with their
different training semantics.

Train-only parent and raw PBE Hessian assets may be reused only through a new
hash-bound v4 manifest. Direction targets must be regenerated or verified to
implement the symmetric, two-sided projected operator above. No validation or
Test100 result may be used to select v4 definitions, weights, thresholds, or
continuation policy.

## 9. Required run metadata

At minimum, each new summary and checkpoint must persist explicit values for:

```text
semantic_version
scalar_owner
energy_label_key
gradient_label_key
label_index
energy_evaluation_point
gradient_evaluation_point
force_evaluation_point
hessian_evaluation_point
gradient_target_semantics
gradient_reduction
hvp_probe_distribution
hvp_input_projection
hvp_output_projection
hvp_loss_reduction
reference_hessian_symmetrization
density_stationarity_role
reference_closure_gate
optimizer_states_fresh
source_weight_sha256
```

An unused CLI default must not be emitted as if it defined an active loss. A
summary field must describe the branch that actually executed.
