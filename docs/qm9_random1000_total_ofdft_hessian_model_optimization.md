# QM9 random1000 Conservative total-OFDFT Hessian Diagnosis and Model Optimization

Last updated: 2026-07-15

## Scope and decision boundary

This report closes the random1000 diagnostic loop between the historical
`incomplete-derived-force Hessian proxy` and the new conservative derivative of the complete
relaxed OFDFT scalar energy. It also records the parent-safe model ablations performed after the
derivative protocol was made auditable.

All model-selection claims are limited to `QM9PBEForceRandom1000`: 1000 parent molecules, four
geometries per parent, with parent-grouped 800/100/100 train/validation/test splits. No parent is
shared across splits. No new label was generated for this continuation, no force head was added,
and every force used by training or evaluation is obtained by autograd from a scalar energy.

Two force definitions must remain distinct:

1. The historical training/evaluation force is `-d E_model(kin_plus_xc) / dR`. It is a useful
   regression proxy but omits the other total-energy nuclear derivatives.
2. The accepted total force is the envelope derivative of the complete constrained scalar energy,
   including learned energy, Hartree, electron-nuclear, nuclear-nuclear, moving-basis/Pulay and
   electron-number-constraint terms.

Historical random1000/Test100 Hessian matrices are not retroactively renamed. They remain
incomplete-derived-force proxies.

## Quantitative answers

- The large smooth-case non-symmetry in the historical density-relaxed proxy is structural. It is
  caused by differentiating an incomplete force after density relaxation. Tightening the density
  threshold by two orders of magnitude does not remove its loop curl or antisymmetric Hessian.
- Density residual and float precision are secondary for the historical smooth cases. Hard molecule
  `0040728` additionally exhibits stationary-branch/finite-step instability.
- The complete total force is conservative within measured numerical accuracy: relaxed scalar
  first derivatives, closed-loop work and cross-Hessian columns converge together after the Pulay
  path is preserved.
- Scalar total-energy second differences need a much larger audit step than total-force first
  differences. At very small `h`, cancellation of energies of order hundreds of Hartree creates a
  false force/energy curvature discrepancy even at strict density stationarity.
- The new force-weight-1 plus scalar-energy-secant candidate improves independent Test100 energy,
  force and complete-total coordinate-0 HVPs, plus all ten fixed-density proxy Hessians. Its
  complete-total full-Hessian anchor is also positive; the conclusion remains a random1000 pilot.

## Data, checkpoints, and representative molecules

Dataset and split:

```text
/scratch/xzh/data/QM9PBEForceRandom1000
/scratch/xzh/data/QM9PBEForceRandom1000/split.pkl
```

Reference PBE analytic Hessians:

```text
/scratch/xzh/models/eval/qm9_random1000_pbe_hessian_refs/test100_pbe_hessians
/scratch/xzh/models/eval/qm9_random1000_test100_hessian_node04_gpu4pyscf/
  20260714_135600/pbe_hessian_manifest_test100_gpu4pyscf.json
```

Accepted control and new candidate:

```text
Baseline actual force weight 1.0:
/scratch/xzh/models/train/runs/qm9_random1000_egf_forcew1p0_e10_20260714_202203/
  checkpoints/epoch_009.ckpt

Candidate force weight 1.0 + scalar energy-secant weight 0.01:
/scratch/xzh/models/train/runs/
  qm9_random1000_forcew1p0_secantw0p01_e10_20260715_210000/
  checkpoints/epoch_009.ckpt
```

The strict complete-total HVP set is `0000777`, `0040728`, `0003027`, `0000242`, and
`0000835`. It includes the small force-error outlier, the structural branch/step hard case, an
observed complete-response hard case, and two small/medium representatives from the existing frozen
ten-molecule test set. The broader historical protocol scan used eight representatives spanning
7--27 atoms and both central and tail errors.

## Error-source evidence chain

### Historical incomplete force

The complete 8-molecule `3x3` scan used `h={3e-3,1e-3,3e-4} Bohr` and requested projected
density-gradient thresholds `{1e-4,3e-5,1e-5}`. At `h=1e-3`, tightening `1e-4 -> 3e-5`
changed mean raw MAE by less than `1e-6 Hartree/Bohr^2` and mean antisymmetric/symmetric Frobenius
ratio by less than `4e-6`. Across 72 molecule/condition rows per model, the Pearson correlation
between mean density residual and asymmetry ratio was only `0.188` for EG and `0.008` for EGF.

On `0000777`, the relaxed incomplete-force closed-loop curl remained essentially invariant while
the density residual fell by two orders of magnitude:

| model | threshold | max density gradient | relaxed curl |
|---|---:|---:|---:|
| EG | `1e-4` | `9.72e-5` | -0.107629 |
| EG | `1e-6` | `9.93e-7` | -0.107581 |
| EGF | `1e-4` | `9.41e-5` | -0.059058 |
| EGF | `1e-6` | `9.62e-7` | -0.058985 |

The fixed-density loop work was near `1e-12--1e-8 Ha`. Float32/float64 fixed-density autograd
Hessians differed by only `1e-5` relative or less, and float64 autograd-vs-FD error was about
`1e-4` relative. These controls rule out matrix formatting, ordinary float precision and density
residual as leading explanations for the smooth-case non-symmetry.

`0040728` is a separate hard case. Its asymmetry changes strongly with step, many requested
`1e-6` density solves do not realize that threshold within 10000 cycles, and displaced points can
land on different stationary branches. It cannot define a single trusted local tensor without an
explicit branch and step-stability check.

### Complete scalar-derived total force

The implemented constrained energy is

\[
E(c,R)=E_{\mathrm{model}}+\tfrac12c^T J(R)c+c^T v_{\mathrm{ext}}(R)+E_{nn}(R),
\qquad q(R)^Tc=N_e,
\]

and the stationary total force is `F=-partial_R[E+mu(q^Tc-Ne)]`. The tensor path preserves the
moving overlap graph and includes all integral and constraint derivatives. On P1 molecule
`0000010`, the total force agrees with independently relaxed scalar-energy differences to
`1.69e-4 Ha/Bohr`; a strict `1e-5 Bohr` two-coordinate loop has work `-7.76e-10 Ha`; and implicit
KKT HVPs agree with strict relaxed force differences at about `1e-4` relative. The cross-column
mismatch is also about `1e-4` relative. This rejects structural nonconservativity for the complete
force at the measured derivative precision.

## Recommended numerical protocol

Use two distinct finite-difference scales:

1. Complete total-force Hessian/HVP: establish a local force-difference stability interval. The
   current pathological P1 audit uses `h=3e-5` and `1e-5 Bohr`; the random1000 directional scan
   checks `3e-3` down to `1e-5` before fixing a local value.
2. Relaxed scalar-energy curvature audit: use the largest step in the local quadratic interval,
   normally near `3e-3--1e-3 Bohr`. Do not compute a second energy difference at `1e-5` for a
   total energy of hundreds of Hartree and interpret cancellation as nonconservativity.

Every formal point must record the realized projected density-gradient norm, cycles, final energy,
constraint residual and stationary-branch diagnostics. The current strict total protocol uses
two-stage Adam, tangent L-BFGS and Newton-PCG refinement, with final gradients around
`1e-8--1e-10`. Report raw, symmetric and antisymmetric matrices; symmetrization is allowed only for
vibrational diagnostics and is not a physical repair.

## Model ablations

### Existing force-weight sweep

The parent-safe validation-only weight sweep tested actual force weights `0.1,0.3,1,3,10`.
Weight 3 had the best energy-eligible historical proxy score and weight 1 passed the frozen Test100
energy gate. Weight 10 failed that gate. A paired `R-/R+` train-parent augmentation marginally
improved force/Hessian proxy values but failed the energy gate. No candidate used Test100 for
selection.

Strict complete-total HVPs later showed that weight 3 is not a robust replacement for weight 1:
it improved `0000777` and `0040728`, but the `0003027` relative Frobenius error increased from
`423.3` to `1058.3`; the three-molecule mean increased from `146.0` to `357.2`.

### Scalar-secant-only candidate

The first scalar-energy-secant candidate disabled force supervision. It improved energy but made
Test100 force component MAE about 39 times worse and lost all ten fixed-density proxy comparisons.
Its three-molecule strict complete-total HVP result was also worse. This candidate is rejected.

### Force plus scalar-secant candidate

The retained experiment uses energy/gradient/force/secant weights `0.1/0.8/1.0/0.01`. Pair batches
contain exact deterministic train-parent `R-/R+` geometries; validation and test remain unchanged.
The secant compares scalar `kin_plus_xc` energies and therefore does not reinterpret total PBE force
as the derivative of that component. Force remains `-dE_model/dR`; no force head exists.

Slurm 880 completed ten epochs on 8 A100 GPUs in `45:58`, with aggregate MaxRSS `14.48 GiB`.
Validation energy/force/total losses changed from `0.0706/0.00294/0.0100` after epoch 0 to
`0.0205/0.00241/0.00448` after epoch 9. Epoch 9 is the selected checkpoint; no final checkpoint was
chosen from Test100.

### Second-order smoothness and supervision choice

The model uses GELU/SiLU nonlinearities with finite float64 first and second derivatives. Cutoffs
are disabled, displaced evaluations retain fixed complete directed graphs, and self-loop distances
use the second-order-safe constant/masked path rather than `norm(0)`. Fixed-density float32/float64
Hessians differ by about `1e-5` relative or less, so evaluation uses float64 but ordinary precision
is not the leading model error.

Three curvature-supervision choices were considered:

- full Hessian loss is rejected for this random1000 training pass because differentiating a model
  Hessian loss with respect to parameters requires a costly third-order graph and full `3N x 3N`
  targets;
- direct autograd HVP loss has the same higher-order parameter-gradient issue, though HVP remains
  the correct evaluation/screening primitive;
- local force differences are cheaper but need physically consistent complete-total forces at both
  paired geometries. Those labels/path are not yet exposed in the batched trainer.

The implemented scalar `R-/R+` energy secant is therefore a conservative, low-order regularizer
combined with the existing scalar-derived force loss. It is not presented as equivalent to HVP or
full-Hessian supervision. The next trainer should first expose batched complete-total forces, then
compare complete-total force secants with stochastic HVP directions on validation only.

## Three-tier evaluation funnel

### Tier 1: energy, force, and fixed-density proxy

Slurm 885 completed the independent fast evaluation in `5:14` with MaxRSS `3.73 GiB`.

| Test100 metric | weight-1 baseline | force+secant candidate | relative change |
|---|---:|---:|---:|
| energy MAE | 0.0275710 | 0.0217518 | -21.11% |
| force component MAE | 0.00246285 | 0.00217705 | -11.60% |
| force vector MAE | 0.00500111 | 0.00443860 | -11.25% |

On the frozen ten-molecule fixed-density incomplete proxy, candidate mean Hessian MAE is
`0.011493` versus baseline `0.014711`, and mean relative Frobenius error is `0.31281` versus
`0.45473`. Candidate wins 10/10 molecules. Autograd-vs-FD relative error remains below `1.1e-4`.
These are screening results, not complete physical Hessians.

### Tier 2: strict complete-total HVP and full Hessian

The final five-molecule aggregate is generated from:

```text
/scratch/xzh/models/eval/qm9_random1000_force_secant_total_confirm/
  20260715_221500_5mol_hvp
```

All ten model/molecule runs are finite and use dense tangent reference solves after PCG/MINRES
failed the explicit `1e-8` response-residual gate. Realized displaced density gradients are below
`7.3e-9`. Strict relaxed total-force differences, not the implicit approximation, define the PBE
comparison.

| molecule | natoms | baseline Hv MAE | candidate Hv MAE | change | baseline/candidate rel Fro |
|---|---:|---:|---:|---:|---:|
| 0000777 | 7 | 0.171739 | 0.095086 | -44.63% | 12.310 / 5.549 |
| 0040728 | 21 | 0.061168 | 0.051934 | -15.10% | 2.265 / 1.719 |
| 0003027 | 12 | 0.805932 | 0.882150 | +9.46% | 423.325 / 450.262 |
| 0000242 | 12 | 0.165862 | 0.081438 | -50.90% | 3.300 / 1.549 |
| 0000835 | 14 | 0.124950 | 0.066683 | -46.63% | 3.144 / 1.689 |
| mean | | 0.265930 | 0.235458 | -11.46% | 88.869 / 92.154 |

Candidate wins 4/5 molecules; mean RMSE changes `0.665832 -> 0.574531` (-13.71%). Mean relative
Frobenius worsens 3.70% because `0003027` has a very small PBE directional reference and dominates
that normalized aggregate; median relative Frobenius improves `3.300 -> 1.719` (-47.91%). The
candidate is a supported improvement trend with an explicit hard-case regression, not a universal
per-molecule improvement.

Implicit-vs-strict relative error is below 0.15% except `0003027`, where it is 1.60% for baseline
and 0.76% for candidate. Dense KKT residuals are near machine precision, so this remaining mismatch
comes from mixed-geometry finite differencing/nonlinearity rather than the linear solver. Formal
metrics continue to use strict force differences on this hard case.

The scalar-energy/total-force curvature scan selects `h=3e-3` for most model/molecule cases and
`h=1e-3` for the harder `0040728`/`0003027` cases, with best relative discrepancies from
`2.2e-5` to `1.68e-2`.
At `1e-5`, energy second differences are cancellation dominated while force differences remain
stable. This directly resolves the apparent force/energy mismatch in the original HVP logs.

The complete full-Hessian anchor and vibrational outputs are:

```text
/scratch/xzh/models/eval/qm9_random1000_force_secant_total_confirm/
  20260715_220000
```

At `0000777`, `h=1e-5 Bohr`, both models have 42/42 strict displaced points:

| metric | weight-1 baseline | force+secant candidate | change |
|---|---:|---:|---:|
| Hessian MAE | 0.127711 | 0.099883 | -21.79% |
| Hessian RMSE | 0.287738 | 0.211838 | -26.38% |
| relative Frobenius | 2.05393 | 1.51214 | -26.38% |
| asym/sym Fro | 7.95e-4 | 2.91e-3 | higher, both below 0.3% |
| symmetry max abs | 0.00187 | 0.00439 | higher |

The `h=3e-5` step check is stable: baseline/candidate MAE is `0.127697/0.099921`, relative
Frobenius is `2.05335/1.51348`, and all 42/42 points per model are strict. Relative to `h=1e-5`,
the MAE changes by only 0.011% for baseline and 0.038% for candidate. The asym/sym ratios fall to
`4.24e-4/6.29e-4`, showing that the small residual antisymmetry is numerical and step dependent,
whereas the candidate ordering and PBE error are stable. The check took `14.85/13.99 min` per model,
so `1e-5` is the cheaper measured local point and `3e-5` is an appropriate confirmation step.

For the same anchor, the candidate changes frequency MAE `1850 -> 1564 cm-1`, RMSE
`2147 -> 1901 cm-1`, and mean mode overlap `0.706 -> 0.742`. Both models still predict 10
imaginary modes while PBE has zero. This is improvement, not physical-quality convergence.

### Tier 3 boundary

The frozen candidate was evaluated once on coordinate-0 complete-total HVPs for all 100 Test100
parents. Both models used the same strict force-difference protocol; implicit KKT response was only
a cross-check. All 200 model/molecule rows are finite and have maximum realized density gradient
below `9.6e-9`.

| Test100 complete-total HVP metric | weight-1 baseline | force+secant candidate | change |
|---|---:|---:|---:|
| mean MAE | 0.191748 | 0.139365 | -27.32% |
| median MAE | 0.086873 | 0.047511 | -45.31% |
| mean RMSE | 0.572946 | 0.423238 | -26.13% |
| median RMSE | 0.269217 | 0.134201 | -50.15% |
| mean relative Frobenius | 10.9965 | 9.5081 | -13.53% |
| median relative Frobenius | 3.20885 | 1.66287 | -48.18% |

Candidate wins 92/100 per-molecule MAEs. The eight regressions are `0129309` (+111.6%),
`0024788` (+71.8%), `0002686` (+50.0%), `0053219` (+19.8%), `0003027` (+9.46%), `0004063`
(+3.52%), `0047663` (+2.74%), and `0021889` (+2.28%). These IDs are required future regression
cases. The 92/100 result supports a stable independent improvement trend; it does not mean every
direction or full Hessian improves.

Historical fixed and relaxed Test100 Hessian tables remain proxy-only. This final confirmation is
one Cartesian direction per molecule, not 100 complete `3N x 3N` matrices or a Test100 frequency
benchmark. Before allocating that work, evaluate several deterministic/random directions on the
eight regressions and a stratified subset.

## Performance and bottlenecks

- Force+secant 8-GPU training: `45:58`, about 5 training steps/s near steady state.
- Fast Test100 force plus ten fixed Hessians: `5:14`.
- Strict dense-response HVP for three molecules and one model: about `18 min`, MaxRSS about
  `3.8--4.3 GiB`.
- Five-molecule two-model strict HVP plus six curvature steps: `46:08` wall in parallel;
  baseline/candidate process times `45.5/24.5 min`, Slurm MaxRSS `5.23 GiB`.
- Frozen Test100 complete-total coordinate-0 HVP: `56:44` wall, 16 workers on eight A100 GPUs,
  Slurm MaxRSS `38.6 GiB`, 7.57 allocated GPU-hours. Baseline/candidate summed process times are
  4.56/6.03 hours; candidate has a 32% larger mean but nearly identical median per-molecule time,
  so a small number of convergence tails dominate.
- Strict complete full Hessian for 7-atom `0000777`: `283--345 s` per model in the latest run.
- Fresh GPU4PySCF PBE analytic Hessians measured about `30--54 s/molecule` on a five-molecule
  sample. The current density-relaxed OFDFT full finite-difference Hessian has no demonstrated speed
  advantage over analytic PBE; only fixed-density autograd/HVP is presently a fast screen.

PCG, tangent MINRES and full KKT MINRES frequently report success internally before their explicit
residual reaches `1e-8`; the evaluator now rejects those false positives. Systems with 708--1227
tangent dimensions currently fall back to dense reference solves. The next scalable solver needs a
reused Ritz/deflation space or Hartree/local-block preconditioner. Prepared PySCF moving-basis
integrals and repeated strict density optimization dominate CPU time and host memory, not GPU model
memory. Reuse base densities, cache immutable geometry terms with bounded lifetime, extrapolate
responses between nearby displacements and schedule coordinate shards dynamically.

## Artifact inventory

| purpose | artifact |
|---|---|
| force+secant training | `/scratch/xzh/models/train/runs/qm9_random1000_forcew1p0_secantw0p01_e10_20260715_210000` |
| Slurm training log | `/scratch/xzh/logs/qm9_random1000_force_secant/880.out` |
| Test100 force and 10-molecule fixed proxy | `/scratch/xzh/models/eval/qm9_random1000_energy_secant_posttrain/20260715_214900_force_secant` |
| strict five-molecule total HVP and curvature points | `/scratch/xzh/models/eval/qm9_random1000_force_secant_total_confirm/20260715_221500_5mol_hvp` |
| full `0000777`, `h=1e-5`, plus frequencies | `/scratch/xzh/models/eval/qm9_random1000_force_secant_total_confirm/20260715_220000` |
| full `0000777`, `h=3e-5` | `/scratch/xzh/models/eval/qm9_random1000_force_secant_total_confirm/20260715_222500_full_h3e5` |
| weight-3 strict total HVP ablation | `/scratch/xzh/models/eval/qm9_random1000_forcew3_total_hvp/20260715_strict_residual` |
| merged tables and figures | `/scratch/xzh/models/eval/qm9_random1000_force_secant_total_analysis/20260715` |
| frozen Test100 strict total HVP | `/scratch/xzh/models/eval/qm9_random1000_force_secant_total_hvp_test100/20260715_232500` |

The strict run directories retain `points.csv`, `curvature_scan.csv`, all per-point optimization
trace NPZs, final raw-basis coefficients, response attempts, HVP arrays, full Hessian NPZs and
`/usr/bin/time -v` records. Nothing was written back into labels, `.chk` files or PBE references.

## Reproduction

Training:

```bash
cd /scratch/xzh/code/structures25
source /scratch/xzh/env.sh
RUN_STAMP=qm9_random1000_forcew1p0_secantw0p01_e10 \
  bash scripts/launch_qm9_random1000_force_secant_8xa100.sh
```

Strict total HVP with independent curvature steps:

```bash
python scripts/qm9_total_ofdft_hvp_audit.py \
  --dataset-dir /scratch/xzh/data/QM9PBEForceRandom1000 \
  --reference-dir /scratch/xzh/models/eval/qm9_random1000_pbe_hessian_refs/test100_pbe_hessians \
  --run NAME=RUN_DIR=CHECKPOINT --molecules 0000777 --direction-coordinate 0 \
  --hvp-step 1e-5 --curvature-step 3e-3 --curvature-step 1e-3 \
  --curvature-step 3e-4 --curvature-step 1e-4 --curvature-step 3e-5 \
  --mixed-derivative-step 1e-4 --response-solver auto \
  --krylov-tolerance 1e-8 --max-krylov-iterations 1200 \
  --dense-fallback-max-coefficients 2048 \
  --fallback-always --lbfgs-refine --newton-refine \
  --device cuda:0 --output-dir /path/to/output
```

Full Hessian step check:

```bash
CANDIDATE_RUN=/path/to/run CANDIDATE_CKPT=/path/to/checkpoint \
MOLECULE=0000777 DISPLACEMENT=3e-5 OUT_DIR=/path/to/output \
  sbatch --export=ALL scripts/slurm_qm9_random1000_total_full_hessian_step_node02.sbatch
```

Machine-readable merge and plots:

```bash
python scripts/qm9_total_ofdft_model_optimization_analysis.py \
  --training-log /scratch/xzh/logs/qm9_random1000_force_secant/880.out \
  --fast-eval-dir /scratch/xzh/models/eval/qm9_random1000_energy_secant_posttrain/\
20260715_214900_force_secant \
  --total-confirm-dir /scratch/xzh/models/eval/qm9_random1000_force_secant_total_confirm/\
20260715_221500_5mol_hvp \
  --full-confirm-dir /scratch/xzh/models/eval/qm9_random1000_force_secant_total_confirm/\
20260715_220000 \
  --weight3-hvp-summary /scratch/xzh/models/eval/qm9_random1000_forcew3_total_hvp/\
20260715_strict_residual/hvp/summary.json \
  --test100-total-hvp-dir /scratch/xzh/models/eval/\
qm9_random1000_force_secant_total_hvp_test100/20260715_232500 \
  --output-dir /scratch/xzh/models/eval/qm9_random1000_force_secant_total_analysis/\
20260715
```

Aggregate CSV/JSON/PNG outputs are mirrored locally at
`_runtime/remote_artifacts/qm9_random1000_force_secant_total_analysis/20260715`; full optimization
traces, coefficient snapshots and Hessian arrays remain under the remote run directories listed
above.

Focused remote verification passes 21/21 tests covering complete-force graph replacement,
moving-basis integral derivatives, parallel derivative consistency, strict stationary solvers,
PCG/deflated-PCG/MINRES/KKT response, parent-pair DDP sharding, loss shapes, curvature accounting
and total-Hessian vibrational artifact parsing. All modified Python launch/evaluation modules pass
`py_compile`; shell launchers pass `bash -n`.

## Implementation recommendation

The force+secant checkpoint is accepted as the leading random1000 candidate because it improves
independent Test100 energy, force and complete-total HVP aggregates, with 92/100 directional wins,
and improves the two-step full-matrix anchor. It is not a final full-QM9 model and its eight Test100
regressions must remain visible. Its force loss still differentiates the learned energy component
rather than the complete constrained total scalar. The next model version should expose a batched
differentiable total-energy assembly during training and supervise `-dE_total/dR` at label
densities. The same scalar must own energy, force, secant and HVP paths.

For a production relaxed Hessian, implement the implicit Schur complement

\[
H_{RR}^{rel}=E_{RR}-E_{Rc}E_{cc}^{-1}E_{cR}
\]

in the electron-number tangent space, with explicit residual checks and reused Krylov subspaces.
Until that implementation is validated on all hard cases, strict finite difference of the complete
scalar-derived total force remains the formal reference and implicit HVP remains an acceleration
cross-check.
