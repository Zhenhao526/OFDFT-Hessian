# Graphformer Complete-Total Analytic Relaxed-HVP Refactor

Last updated: 2026-07-24 21:10 Asia/Singapore

## Status

The single-parent implementation and focused tests are complete. The formal `0028399`
correctness/performance audit has **not** run because node02 does not contain the registered
original-A checkpoint or frozen stable5 parent/direction assets. The launcher failed closed before
loading data or a model. No stable5/train20 training was started, no proxy checkpoint was used,
and validation/Test100 remain unread.

The active frozen protocol is:

```text
configs/audit/qm9_graphformer_complete_total_analytic_relaxed_hvp_v2.yaml
```

The active node02 root is:

```text
/home/shenwei01/xzh_node02_20260724
```

`/scratch` is abandoned for this workflow and must not be probed or used as a fallback.

## Implemented Definition

One scalar owns energy, force, and curvature:

```text
L(R,c,mu;theta) = E_total(R,c;theta) + mu (n(c,R) - N)
E_total = E_Graphformer_kin_plus_xc + E_H + E_ext + E_nn
F = -d E_total(R,c_star(R;theta);theta)/dR
```

At one strictly converged center density, with `y=(c,mu)` and constrained stationarity
`G(y,R,theta)=0`, the geometry response and relaxed HVP are:

```text
G_y y_v = -G_R v
H_relaxed v = L_RR v + L_Ry y_v
```

The small-parent formal path builds the tangent KKT matrix and solves it with a differentiable
direct factorization. Backward uses the implicit adjoint of the linear solve:

```text
A x = b
A^T lambda = d loss / d x
d loss / d b = lambda
d loss / d A = -lambda x^T
```

There is no fixed-density, stale-density, detached-HVP, or independent force/Hessian-head
fallback. A missing response, non-finite value, nonzero constraint derivative unsupported by the
current solver, or excessive KKT residual raises immediately.

### Geometry-integral boundary

PySCF/libcint is outside torch autograd. The implementation is therefore semianalytic:

- density response, Graphformer response, KKT solve, relaxed-HVP assembly, and parameter adjoint
  are analytic/autograd connected;
- complete first nuclear derivatives of moving-basis PySCF integrals are evaluated as before;
- their directional second derivatives are central differences of those complete first-derivative
  bundles;
- only the center density is optimized during training; displaced density optimization is absent;
- strict `R+/-h v` density reoptimization remains only as the independent audit oracle.

This is a complete-total relaxed HVP with a numerical integral-HVP boundary, not a fully analytic
libcint second-integral implementation.

## Stochastic Training Change

The four structured directions per step were replaced by one fresh Rademacher probe in the full
orthonormal `3N-6` internal basis:

```text
z_i in {-1,+1}
v = B^T z
target = H_PBE v
```

For internal-space error matrix `A=(H_model-H_PBE)B^T`,

```text
E_z ||A z||^2 = ||A||_F^2.
```

The implemented mean-internal-matrix reduction divides by the internal dimension. It is therefore
an unbiased estimator of the internal Hessian Frobenius mean-squared error. A new probe is sampled
every optimizer step. The complete basis is used only for step 0 and periodic/final evaluation.

## Response Solvers and Density Prediction

The correctness path for `0028399` is `dense_direct_implicit_adjoint`. The audit also benchmarks:

| Solver | Purpose |
|---|---|
| matrix-free tangent PCG | scalable SPD candidate |
| full KKT MINRES | indefinite constrained-system candidate |
| low-mode deflated PCG | ill-conditioned tangent response |
| block PCG | reuse across the full internal direction basis |
| dense direct factorization | small-parent correctness oracle and reusable adjoint |

The failed simple-Jacobi strategy is not used.

After an optimizer update, the code evaluates the linear parameter response at the old stationary
point and predicts:

```text
c_next,initial = c_star + dc_star/dtheta * Delta theta.
```

The next step always runs the strict density corrector and requires projected density-gradient
norm below `1e-8`. For the finite-difference audit only, endpoint optimization starts from
`c_plus/minus = c0 +/- h c_v` and must also reach the same strict threshold.

## Files

New or materially changed files for this refactor:

```text
configs/audit/qm9_graphformer_complete_total_analytic_relaxed_hvp_v2.yaml
mldft/ofdft/geometry_integrals.py
mldft/ofdft/conservative_force.py
mldft/ofdft/implicit_response.py
mldft/ofdft/complete_total_training.py
scripts/qm9_complete_total_capacity_train.py
scripts/qm9_graphformer_analytic_relaxed_hvp_audit.py
scripts/launch_qm9_graphformer_analytic_hvp_node02.sh
tests/ofdft/test_geometry_integrals.py
tests/ofdft/test_implicit_response.py
tests/ofdft/test_complete_total_training.py
tests/test_qm9_complete_total_capacity_train.py
```

Training logs now separate density refresh, center graph/integral preparation, KKT solve, relaxed
HVP assembly, high-order backward, optimizer, density predictor, total step time, and peak GPU
memory.

## Verification Completed

Local commands:

```bash
.venv/bin/python -m py_compile \
  mldft/ofdft/geometry_integrals.py \
  mldft/ofdft/conservative_force.py \
  mldft/ofdft/implicit_response.py \
  mldft/ofdft/complete_total_training.py \
  scripts/qm9_complete_total_capacity_train.py \
  scripts/qm9_graphformer_analytic_relaxed_hvp_audit.py

.venv/bin/python -m pytest -q \
  tests/ofdft/test_geometry_integrals.py \
  tests/ofdft/test_implicit_response.py \
  tests/ofdft/test_complete_total_training.py \
  tests/test_qm9_complete_total_capacity_train.py
```

Result: `44 passed`. Covered checks include:

- unbiased internal Hutchinson loss and fresh probe/target construction;
- exact requested directional integral HVP at the torch/PySCF boundary;
- direct relaxed KKT HVP against a constrained quadratic oracle;
- parameter gradients through the implicit linear-solve adjoint;
- nonlinear stationary-density parameter response against strict finite difference;
- density parameter-step predictor;
- PCG, MINRES, deflated PCG, and block-PCG solve behavior;
- strict density and stale-density rejection.
- analytic checkpoint definition and frozen protocol/direction/source provenance.

These are implementation tests, not the required real Graphformer/PySCF scientific audit.

## Node02 Formal Preflight

Command:

```bash
ssh -J root@101.200.216.125 -p 2200 shenwei01@localhost
ssh node02
export QM9_ANALYTIC_HVP_OUTPUT=/home/shenwei01/xzh_node02_20260724/runs/graphformer_analytic_hvp/preflight_missing_assets_20260724
bash /home/shenwei01/xzh_node02_20260724/work/structures25/scripts/launch_qm9_graphformer_analytic_hvp_node02.sh
```

Recorded output:

```text
/home/shenwei01/xzh_node02_20260724/runs/graphformer_analytic_hvp/
preflight_missing_assets_20260724/failure.json
```

The post-sync repeat is:

```text
/home/shenwei01/xzh_node02_20260724/runs/graphformer_analytic_hvp/
preflight_missing_assets_v2_20260724/failure.json
```

The preflight stopped on:

```text
FileNotFoundError:
/home/shenwei01/xzh_node02_20260724/artifacts/original_A/last.ckpt
```

The post-sync repeat exited in `6.00 s`, used `828584 KiB` maximum RSS, left no
training/audit Python process, and recorded:

```text
passed=false
proxy_fallback_used=false
validation_accessed=false
test100_accessed=false
```

The following exact assets must be restored under node02 local `/home`:

| Asset | Required SHA256 |
|---|---|
| original-A `last.ckpt` | `e6516b04917a9dfaae3c4d960b10d77c768f3e5e784f95290ff41c1202f9d9bc` |
| stable5 parent manifest | `72ecaab022402fb59de405487e4259e33d3438691013c2fcb67ae2d3379d774e` |
| stable5 direction manifest | `1d9d235719805b3bf298fe3baf783b1b17a262e352b4193388f0be4f9b65bd74` |

The local `trained-on-qm9` checkpoint has a different hash and is prohibited as a substitute.

## Unresolved Acceptance Gates

The following results are intentionally **not reported yet**:

- analytic relaxed HVP versus strict reoptimized force FD relative error `<=1e-5`;
- HVP-loss parameter gradient versus reoptimized parameter FD error `<5%`;
- complete 39-direction Hessian non-regression and symmetry;
- measured center density, KKT, high-order backward, and full39 timing;
- at least `10x` speedup and less than `60 s/step`.

Until the exact assets are restored and all gates pass:

- do not start stable5 or train20;
- do not claim the new path is scientifically validated;
- do not replace the formal density-relaxed FD Hessian conclusion;
- do not start the conservative graph curvature-corrector alternative solely from the asset
  preflight failure.

If the real audit runs and either speed is below `10x` or one training step exceeds one minute,
the next experiment is the explicit energy-conserving graph curvature corrector specified by the
protocol. Correctness-gate failure instead requires fixing the KKT/integral/gradient path first.
