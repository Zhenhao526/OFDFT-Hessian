# QM9 Conservative total-OFDFT Force and Hessian Audit

Last updated: 2026-07-15

## Status and scope

This document records the implementation gate for replacing the historical
`incomplete-derived-force Hessian proxy` with a conservative derivative of the complete relaxed
OFDFT scalar energy. P1 molecule `0000010`, `sample_id=0`, provides the detailed derivative and
conditioning implementation audit. The continuation adds five representative random1000 strict
directional checks, a two-step full-matrix anchor and an independent Test100 energy/force screen.
It remains a random1000 pilot rather than a full-QM9 scientific conclusion.

The old random1000 density-relaxed Hessians remain proxies because they differentiated only the
learned `kin_plus_xc` term. They omitted Hartree, electron-nuclear, nuclear-nuclear, moving-basis
and constraint derivatives. They must not be relabelled as total-OFDFT Hessians.

## Implemented scalar and derivative definition

The tensor-native energy is

\[
E(c,R)=E_{\mathrm{model}}(c,R)+\frac12 c^T J(R)c+c^Tv_{\mathrm{ext}}(R)
       +E_{\mathrm{nn}}(R),
\]

with electron-number constraint

\[
g(c,R)=q(R)^Tc-N_e=0,
\qquad
L(c,R,\mu)=E(c,R)+\mu g(c,R).
\]

At a strict stationary density, the total force is the envelope derivative

\[
F_R=-\partial_R L(c^*(R),R,\mu^*(R)).
\]

Implemented terms and interfaces:

- graph-preserving `TensorEnergies` and tensor functional assembly;
- learned `kin_plus_xc`, Hartree, electron-nuclear and nuclear-nuclear scalar terms;
- electron number and Lagrange multiplier diagnostics;
- exact zero derivative of normalized auxiliary-basis integrals;
- analytic moving-centre overlap derivative from `int1e_ipovlp`;
- complete reference finite differences for Coulomb and electron-nuclear integral derivatives;
- self-pair-safe nuclear repulsion;
- strict tangent L-BFGS plus Newton-PCG density refinement;
- stable Loewner derivative for symmetric overlap square root/inverse square root;
- deterministic, translation-covariant dummy local-frame positions;
- preservation of caller-owned graph-connected overlap tensors through preprocessing, so the
  natural-basis overlap/Pulay response remains in transform autograd;
- numerical fixed-physical-density model geometry VJP retained only as an acceptance oracle.

The analytic mismatch was caused by `AddOverlapMatrix` rebuilding and silently detaching an
overlap tensor already injected by `prepare_differentiable_geometry`. The transform now preserves
an existing overlap. Pure transform autograd and the independently rebuilt scalar VJP differ by
only `7.35e-5 Ha/Bohr` in the audited force component instead of about `4.69 Ha/Bohr` before the
fix. The numerical VJP is no longer part of the default force path.

## Code map

- `mldft/ofdft/energies.py`: graph-preserving energy container.
- `mldft/ofdft/functional_factory.py`: complete tensor scalar energy and constraint Lagrangian.
- `mldft/ofdft/basis_integrals.py`: analytic overlap coordinate derivatives.
- `mldft/ofdft/geometry_integrals.py`: moving-basis integral values/derivatives.
- `mldft/ofdft/conservative_force.py`: fixed-geometry scalar evaluator and complete total force.
- `mldft/ofdft/stationary_density.py`: strict tangent L-BFGS/Newton-PCG refiners.
- `mldft/ofdft/implicit_response.py`: KKT response, PCG/MINRES, dense audit and deflated PCG.
- `scripts/qm9_total_ofdft_force_audit.py`: scalar-force and closed-loop acceptance test.
- `scripts/qm9_total_ofdft_hvp_audit.py`: strict relaxed FD versus implicit-response HVP.
- `scripts/qm9_total_ofdft_hessian_audit.py`: strict full total-force Hessian, scalar diagonal
  curvature, PBE error and raw/symmetric/antisymmetric diagnostics.
- `scripts/qm9_hessian_vibrational_metrics.py`: mass weighting, external-mode projection,
  frequency and mode-overlap diagnostics for total-Hessian artifacts.
- `scripts/qm9_total_ofdft_model_optimization_analysis.py`: merged model-funnel CSV/JSON/plots.

Persistent raw output root:

```text
_runtime/qm9_p1_models/eval/qm9_total_ofdft_conservative_audit/20260715
```

It contains CSV/JSON summaries, all optimization traces, final raw-basis coefficient snapshots and
compressed HVP/spectrum arrays.

## Strict density stationarity

The staged density solve uses the established two-stage Adam warm start, followed by tangent
L-BFGS and Newton-PCG for the derivative audit. All reported force/HVP points have final projected
density-gradient norms around `6e-11` to `1e-10`; electron-number residuals are around `1e-13` or
smaller. Tightening density stationarity therefore no longer changes the conclusions below.

## Total force versus relaxed scalar energy

Coordinate 0 of molecule `0000010` was checked against independently optimized scalar energies at
`R +/- h`.

| outer scalar FD h (Bohr) | total force | relaxed-energy FD force | absolute difference | max displaced projected grad |
|---:|---:|---:|---:|---:|
| `3e-5` | -68.14936925 | -68.26138870 | 1.1202e-1 | 9.19e-11 |
| `1e-5` | -68.14936927 | -68.16186097 | 1.2492e-2 | 6.62e-11 |
| `3e-6` | -68.14936922 | -68.15009293 | 7.2371e-4 | 8.89e-11 |
| `1e-6` | -68.14936925 | -68.14946448 | 9.5233e-5 | 9.55e-11 |

The error decreases by more than three orders of magnitude while stationarity is unchanged. The
previous `0.112 Ha/Bohr` discrepancy was outer scalar-FD truncation, not a missing density response
or failed density solve.

Inner model-VJP calibration:

| model geometry derivative | integral derivative h | absolute force difference |
|---|---:|---:|
| central `h=1e-6` | `1e-4` | 9.5233e-5 |
| central `h=3e-7` | `3e-5` | 3.3785e-3 |
| Richardson from `2e-6` and `1e-6` | `1e-4` | 1.260e-4 |

`1e-6` is the measured optimum. Smaller steps enter cancellation; Richardson roughly doubles the
model geometry work and is slightly less accurate.

After preserving the graph-connected overlap, the pure transform-autograd force is
`-68.14929575 Ha/Bohr`; the same relaxed-energy FD reference is `-68.14946454 Ha/Bohr`, an absolute
difference of `1.69e-4 Ha/Bohr`. The old numerical model VJP gives `-68.14936925 Ha/Bohr`. The
analytic force pipeline takes `0.85 s` at this geometry and removes the `2*3N` model rebuilds.

## Closed-loop conservativity

The two-dimensional loop uses flattened coordinates 0 and 1 and trapezoidal edge integration of
the complete total force. Every corner is strictly stationary.

| loop half-width (Bohr) | loop work (Ha) | area-normalized discrete curl | max corner grad |
|---:|---:|---:|---:|
| `3e-4` | -6.45098e-4 | -1.79194e3 | 9.73e-11 |
| `3e-5` | -1.33685e-7 | -3.71348e1 | 8.65e-11 |
| `1e-5` | -7.98510e-9 | -1.99627e1 | 9.45e-11 |

Loop work decreases by about 80,800 times from `3e-4` to `1e-5`. The remaining normalized curl is
the expected finite-force/finite-quadrature floor: each force component uses an inner `1e-6`
scalar difference, so dividing a bounded force error by a shrinking loop width eventually stops
converging. The cross-Hessian `H01-H10` check is the stricter follow-up acceptance metric.

With the overlap fix and pure transform autograd, the `1e-5` loop work is further reduced to
`-7.76e-10 Ha` with 4/4 strict corners and maximum projected density gradient `9.10e-11`.

## KKT density response and HVP

For a geometry direction `v`, the constrained response solves

\[
\begin{bmatrix}E_{cc}&q\\q^T&0\end{bmatrix}
\begin{bmatrix}c_Rv\\\mu_Rv\end{bmatrix}
=-
\begin{bmatrix}E_{cR}v\\0\end{bmatrix}.
\]

`E_cc` is an autograd HVP. `E_cR v` is evaluated from coefficient gradients after independently
rebuilding the full geometry at `R +/- h v`. This preserves the mixed derivative that the earlier
first-coordinate-gradient replacement intentionally detached.

On coordinate 0 with `h=1e-5`:

- base projected gradient: `6.02e-11`;
- dense tangent solve residual: `2.19e-14` relative;
- implicit versus strict relaxed density response relative Frobenius: `9.93e-5`;
- implicit versus strict relaxed total-force HVP relative Frobenius: `1.4089e-4`;
- scalar relaxed-energy curvature: `-199455.7672 Ha/Bohr^2`;
- strict relaxed-force directional curvature: `-199484.0277 Ha/Bohr^2`;
- curvature absolute/relative difference: `28.2604` / `1.417e-4`;
- strict HVP norm: `1.4057e6 Ha/Bohr^2`.

The implementation is internally consistent at the `1e-4` relative level. The enormous negative
curvature is therefore a model pathology exposed by the complete physical derivative, not a NaN,
nonconservative-force artifact or density residual.

The independently evaluated coordinate-1 column gives:

- implicit versus strict relaxed HVP relative Frobenius: `2.33e-7`;
- implicit versus strict relaxed density response relative Frobenius: `4.22e-6`;
- strict `H10=-5211.2256` and `H01=-5209.9186 Ha/Bohr^2`;
- strict cross difference `H01-H10=1.3070`, or `2.51e-4` relative to the pair scale;
- implicit cross difference `0.7850`, or `1.51e-4` relative.

This cross-column test is a stronger conservativity check than dividing a finite loop quadrature
error by a vanishing area. At the measured `1e-4` relative derivative accuracy, it does not support
a structural nonconservative-force diagnosis.

At `h=1e-3`, the response is not linear: the direct response and strict density secant differ by
106%, and force/energy curvatures disagree. A local total-OFDFT Hessian for this model must use a
smaller step and demonstrate a stability interval; the historical `h=1e-3` proxy protocol cannot
be transferred to the complete total derivative without revalidation.

After the overlap fix, transform-autograd force differences reproduce the independent strict HVP
columns to relative errors `3.61e-10` and `3.44e-9` for coordinates 0 and 1. The strict cross pair
has relative mismatch `1.86e-4`; the implicit-response pair has `8.63e-5`.

## Strict full total-Hessian step scan

The full evaluator used 36 independently optimized displaced points for the 18 coordinates of
`0000010`. Every point reached projected density-gradient norm below `1e-10`. The PBE reference
Frobenius norm is `3.063 Ha/Bohr^2`.

| model | force FD h (Bohr) | raw rel Fro vs PBE | sym rel Fro vs PBE | asym/sym Fro | max asym element | force/energy diagonal MAE |
|---|---:|---:|---:|---:|---:|---:|
| EGF lambda=1 | `1e-3` | 3.753e5 | 2.924e5 | 8.043e-1 | 6.721e5 | 7.165e4 |
| EGF lambda=1 | `3e-4` | 1.391e6 | 1.265e6 | 4.576e-1 | 1.548e6 | 9.204e4 |
| EGF lambda=1 | `3e-5` | 2.278e6 | 2.278e6 | 1.047e-2 | 8.378e4 | 8.014e2 |
| EGF lambda=1 | `1e-5` | 2.285e6 | 2.285e6 | 1.171e-3 | 9.402e3 | 9.754e1 |
| EG | `1e-5` | 1.508e6 | 1.508e6 | 9.407e-4 | see raw artifact | 4.185e2 |

At `1e-3` and `3e-4`, several heavy-atom displacements converge to different strict stationary
branches. Tight density residuals therefore do not make those steps local derivatives. The
`3e-5` and `1e-5` symmetric Hessians agree to `7.91e-3` relative Frobenius, while the asymmetry
ratio decreases by about nine-fold. This identifies `1e-5` as the current local audit step for
this pathological molecule, not a universal production setting.

The stable local Hessian still contains negative heavy-atom curvatures of order
`1e5-1e6 Ha/Bohr^2`. EGF's symmetric Hessian norm is `1.515x` EG's and its PBE relative error is
about 51% worse. Thus the old force-weight-1 objective improves the incomplete learned-energy
proxy but does not improve the complete relaxed total-energy Hessian. This is a model/objective
failure, not evidence of structural nonconservativity and not repairable by symmetrization.

## Response conditioning and performance

The 394 raw coefficients give a 393-dimensional tangent system.

- tangent Hessian relative asymmetry: `2.67e-16`;
- eigenvalue range: `1.019e-5` to `77.4713`;
- all eigenvalues positive;
- condition number: `7.60e6`;
- dense assembly/factorization and solve: about `7.3 s` in this small audit;
- unpreconditioned PCG at 500 iterations: not converged, relative residual about `5e-4` for the
  physical RHS;
- the Hutchinson-Jacobi preconditioner is worse on this system;
- matrix-free smallest-mode ARPACK did not finish its first 8-mode solve within two minutes and
  was stopped;
- on the saved real tangent matrix, exact low-mode deflation needs 64 modes/492 CG iterations or
  96 modes/304 iterations to reach `1e-8`.

Consequently, dense factorization reused across all geometry directions is the correct small-system
audit reference. A scalable Krylov implementation must reuse a Ritz/deflation subspace across
directions and molecules with related geometries, and should add a physics-based Hartree/local
block preconditioner. Recomputing low modes per direction is not viable.

After the overlap fix, a complete transform-autograd force takes about `0.7-0.9 s` per geometry
after density optimization. The strict five-point `1e-5` loop took `40.2 s` end to end, with about
1.70 GB max RSS. Full 18-coordinate Hessians took `137-379 s` depending on step/model and density
optimization cycles. The dominant remaining cost is density optimization plus CPU construction of
all-coordinate moving-basis Coulomb and electron-nuclear integral derivatives, not model GPU
memory.

## Current acceptance decision

Supported now:

- the complete scalar energy terms and electron constraint agree with the legacy scalar energy to
  around `1e-13 Ha` at every audited point;
- strict total force agrees with the relaxed scalar derivative as the outer step is reduced;
- loop work tends toward zero under step refinement;
- the KKT implicit density response and total-energy HVP agree with strict relaxed finite
  differences at a sufficiently local step;
- pure transform autograd passes the scalar VJP, loop and strict HVP acceptance checks;
- a stable local full total-Hessian interval exists between `3e-5` and `1e-5` for this molecule.

Not yet supported:

- calling historical random1000/Test100 proxy Hessians physical total-OFDFT Hessians;
- general model-quality claims from the single P1 molecule;
- using `h=1e-3` for the complete derivative without a new stability scan;
- treating post-hoc symmetrization as a physical correction;
- claiming that the current incomplete-force EGF objective improves the complete total Hessian.

## Reproduction

Environment prefix:

```bash
cd /mnt/afs/home/xiazhenhao/dft/structures25
export DFT_DATA="$PWD/_runtime/data"
export PYTHONPATH="$PWD"
RUN='EGF_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000/checkpoints/epoch_000.ckpt'
```

Strict force/scalar check:

```bash
.venv/bin/python scripts/qm9_total_ofdft_force_audit.py \
  --dataset-dir _runtime/qm9_p1/QM9PBEForcePilot \
  --run "$RUN" --molecules 0000010 --sample-id 0 \
  --output-dir /path/to/output --coordinates 0 \
  --force-fd-step 1e-6 --integral-derivative-step 1e-4 \
  --model-geometry-derivative autograd --no-run-loop --device cuda:0
```

Dense-KKT HVP audit:

```bash
.venv/bin/python scripts/qm9_total_ofdft_hvp_audit.py \
  --dataset-dir _runtime/qm9_p1/QM9PBEForcePilot \
  --run "$RUN" --molecules 0000010 --sample-id 0 \
  --output-dir /path/to/output --direction-coordinate 0 \
  --hvp-step 1e-5 --mixed-derivative-step 1e-5 \
  --integral-derivative-step 1e-4 --model-geometry-derivative autograd \
  --response-solver dense --device cuda:0
```

Strict full Hessian:

```bash
.venv/bin/python scripts/qm9_total_ofdft_hessian_audit.py \
  --dataset-dir _runtime/qm9_p1/QM9PBEForcePilot \
  --reference-dir _runtime/qm9_p1_models/eval/qm9_p1_410_hessian_ref_benchmark/pbe_hessians \
  --run "$RUN" --molecules 0000010 --sample-id 0 \
  --output-dir /path/to/output --displacement 1e-5 \
  --integral-derivative-step 1e-4 --model-geometry-derivative autograd --device cuda:0
```

## Next gates

1. Add more geometry directions per molecule before expanding from the completed five-molecule
   random1000 strict HVP set to a full Test100 total-Hessian allocation.
2. Keep `0003027` as a mandatory complete-response regression and `0040728` as the branch/step
   hard case. A candidate must report both rather than only aggregate means.
3. Cache geometry integrals, parallelize coordinate derivative construction and reuse the density
   response factorization/Ritz space.
4. Move training force supervision from `-dE_model/dR` to the same complete constrained total
   scalar used here; retain parent-grouped scalar secants and no force head.
5. Re-run the three-tier fixed/HVP -> representative strict -> frozen Test100 funnel only after a
   validation-only candidate is frozen.

## 2026-07-15 model-objective continuation

The scalar-secant-only candidate completed but was rejected: removing force supervision reduced
energy error while making Test100 force component MAE about 39 times worse and losing all ten
fixed-density proxy comparisons.

The retained parent-safe candidate combines actual force weight 1.0 with scalar `kin_plus_xc`
energy-secant weight 0.01. Slurm 880 completed ten epochs on eight A100 GPUs in `45:58`; epoch 9
is validation-best. On independent Test100, energy MAE changes `0.027571 -> 0.021752` and force
component MAE changes `0.002463 -> 0.002177`. It wins all ten fixed-density proxy Hessians.

The strict complete-total five-molecule HVP gate uses `0000777`, `0040728`, `0003027`, `0000242`
and `0000835`. Candidate wins 4/5 and changes mean MAE `0.265930 -> 0.235458` and mean RMSE
`0.665832 -> 0.574531`; `0003027` regresses by 9.46%. Mean relative Frobenius worsens because that
hard direction has a tiny PBE reference, while median relative Frobenius improves 47.9%.

The frozen coordinate-0 Test100 complete-total HVP run then confirms the aggregate trend. Candidate
wins 92/100 molecules and changes mean MAE/RMSE/relative Frobenius from
`0.191748/0.572946/10.9965` to `0.139365/0.423238/9.5081`. Median MAE falls 45.3%. Eight explicit
regressions remain, led by `0129309`, `0024788`, `0002686`, `0053219`, and `0003027`; this is not
a claim that every Hessian direction improves.

The full `0000777` total Hessian is stable between `h=1e-5` and `3e-5 Bohr`. At `1e-5`, candidate
changes MAE `0.127711 -> 0.099883`, RMSE `0.287738 -> 0.211838`, and relative Frobenius
`2.05393 -> 1.51214`; both models have 42/42 strict points. Frequency MAE changes
`1850 -> 1564 cm-1` and mode overlap `0.706 -> 0.742`, but both still predict 10 imaginary modes.

The directional scalar-energy scan resolves the earlier apparent curvature inconsistency. Total
force curvature is stable down to `1e-5`, but relaxed total-energy second differences enter
cancellation below roughly `3e-4`; `3e-3--1e-3` reproduces force curvature at about
`2e-5--1.7e-2` relative depending on the hard case. This is numerical cancellation, not evidence
that the accepted complete force is structurally nonconservative.

Full raw tables, plots and commands are in
`docs/qm9_random1000_total_ofdft_hessian_model_optimization.md` and remote artifact
`/scratch/xzh/models/eval/qm9_random1000_force_secant_total_analysis/20260715`.
