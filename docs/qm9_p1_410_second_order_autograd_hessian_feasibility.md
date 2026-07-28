# QM9 P1-410 second-order autograd Hessian feasibility audit

Date: 2026-07-07

## Scope

This audit only tests feasibility. It does not change training, labels, checkpoints, or the existing finite-difference Hessian conclusions.

Constraints kept:

- no new labels generated;
- no existing labels/checkpoints/references deleted;
- no independent force head added;
- forces and Hessians are derived from scalar model energy;
- primary test is fixed-density, not full density-relaxed implicit Hessian.

## Inputs and artifacts

Code:

- `scripts/qm9_second_order_autograd_hessian_audit.py`

Run output:

- JSON: `_runtime/qm9_p1_models/eval/qm9_p1_410_second_order_autograd_hessian_feasibility/second_order_autograd_audit.json`
- log: `_runtime/qm9_p1_models/eval/qm9_p1_410_second_order_autograd_hessian_feasibility/second_order_autograd_audit.log`
- time: `_runtime/qm9_p1_models/eval/qm9_p1_410_second_order_autograd_hessian_feasibility/second_order_autograd_audit_time.txt`
- Hessian npz files: `_runtime/qm9_p1_models/eval/qm9_p1_410_second_order_autograd_hessian_feasibility/hessians/`

Models:

- EG_s3000: `_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200/checkpoints/epoch_006.ckpt`
- EGF_lam1_s3000: `_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000/checkpoints/epoch_000.ckpt`

Molecules:

- `0000010`, sample_id 0, scf_iteration 1, 6 atoms
- `0000062`, sample_id 0, scf_iteration 1, 8 atoms

Reference:

- `_runtime/qm9_p1_models/eval/qm9_p1_410_hessian_ref_benchmark/pbe_hessians/`

Resource summary:

- wall time: 48.49 s
- max RSS: 1.81 GB
- GPU: CUDA device 0 inside `CUDA_VISIBLE_DEVICES=5`

## Implementation definition

The audit bypasses `MLDFTLitModule.forward_predictions()` and calls `model.net(batch)` directly. This is necessary because the existing eval force path sets `create_graph=self.net.training`; in eval mode it cannot retain the graph needed for second derivatives.

Definitions used:

- fixed density: `batch.coeffs` is held fixed at the evaluator/test sample density;
- force: `F = -dE_model/dR`;
- full Hessian: `H = d2E_model/dR2 = -dF/dR`;
- HVP: `Hv = d/dR[(dE_model/dR) dot v]`;
- FD comparison: central finite difference of derived force with displacement `1e-3`.

This is still a model-energy Hessian at fixed density. It is not a density-relaxed physical Hessian and does not include implicit density response.

## Full-edge fixed-density full Hessian

Current production samples use `AddFullEdgeIndex`, which creates all atom pairs including self loops. The Graphformer `GaussianLayer` then computes distances with `torch.norm(pos_i - pos_j)`. For self loops this is `torch.norm(0)`.

Result: full-edge second-order autograd is not stable. All tested full Hessian matrices are all-NaN, while the finite-difference Hessians remain finite.

| run | molecule | shape | autograd finite | NaN count | autograd wall s | peak CUDA MB | FD finite | FD symmetry max abs | FD vs PBE MAE | FD vs PBE RMSE | FD vs PBE rel Fro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EG_s3000 | 0000010 | 18x18 | no | 324 | 0.701 | 177.8 | yes | 1.15e-3 | 0.500218 | 3.600186 | 21.156991 |
| EG_s3000 | 0000062 | 24x24 | no | 576 | 0.483 | 188.3 | yes | 3.73e-3 | 0.149725 | 0.462450 | 3.493497 |
| EGF_lam1_s3000 | 0000010 | 18x18 | no | 324 | 0.179 | 176.6 | yes | 8.45e-5 | 0.011801 | 0.023857 | 0.140202 |
| EGF_lam1_s3000 | 0000062 | 24x24 | no | 576 | 0.235 | 187.2 | yes | 8.04e-5 | 0.016381 | 0.046167 | 0.348758 |

Autograd-vs-FD MAE/RMSE/relative Frobenius are therefore invalid for the full-edge current model. EG/EGF autograd ranking is also invalid. The FD side still reproduces the previous fixed-density trend: EGF is much better than EG on these two molecules.

## Self-edge diagnostic

A diagnostic-only run removed self-loop edges before forward on `0000010`. This changes the model graph and is not a valid formal benchmark, but it tests the suspected source of NaNs.

| run | molecule | edges original -> kept | autograd finite | autograd symmetry max abs | autograd-vs-FD MAE | RMSE | rel Fro |
|---|---:|---:|---:|---:|---:|---:|---:|
| EG_s3000 | 0000010 | 36 -> 30 | yes | 5.05e-7 | 4.94e-4 | 4.13e-3 | 8.79e-4 |
| EGF_lam1_s3000 | 0000010 | 36 -> 30 | yes | 2.78e-8 | 4.09e-6 | 1.16e-5 | 8.13e-5 |

This strongly indicates that the current NaNs come from second derivatives through zero-length self-edge distances, not from scalar-energy autograd in general.

Independent minimal check:

```python
x = torch.zeros(3, dtype=torch.float64, requires_grad=True)
y = torch.norm(x)
g = torch.autograd.grad(y, x, create_graph=True)[0]  # finite zero
torch.autograd.grad(g[0], x)                         # NaN row
```

Code locations:

- self loops are introduced by `AddFullEdgeIndex` in `mldft/ml/data/components/convert_transforms.py`;
- zero-distance norm is computed in `mldft/ml/models/components/gbf_module.py`.

## HVP audit

HVP was tested on random unit directions and existing perturbation directions for the same two molecules.

Result: full-edge HVP is also not stable. Every tested HVP vector is all-NaN.

| run | molecule | direction | HVP finite | NaN count | HVP wall s | peak CUDA MB |
|---|---:|---|---:|---:|---:|---:|
| EG_s3000 | 0000010 | random_unit | no | 18 | 0.386 | 174.1 |
| EG_s3000 | 0000010 | perturb_unit | no | 18 | 0.396 | 174.1 |
| EG_s3000 | 0000062 | random_unit | no | 24 | 0.018 | 182.0 |
| EG_s3000 | 0000062 | perturb_unit | no | 24 | 0.471 | 182.0 |
| EGF_lam1_s3000 | 0000010 | random_unit | no | 18 | 0.018 | 172.8 |
| EGF_lam1_s3000 | 0000010 | perturb_unit | no | 18 | 0.018 | 172.8 |
| EGF_lam1_s3000 | 0000062 | random_unit | no | 24 | 0.018 | 180.7 |
| EGF_lam1_s3000 | 0000062 | perturb_unit | no | 24 | 0.018 | 180.7 |

Conclusion: HVP is not more stable under the current full-edge distance implementation. After fixing the self-edge zero-norm path, HVP is still the more attractive next implementation target because it avoids materializing the full Hessian and is the natural primitive for implicit density response.

## Unrolled density optimization prototype

Prototype definition:

- molecule: `0000010`
- run: EGF_lam1_s3000
- differentiable coefficient updates using model energy only
- projected coefficient gradient
- lr `1e-4`
- steps: 10, 50, 100
- not the production OFDFT density optimizer

| steps | final energy | HVP finite | NaN count | wall s | peak CUDA MB |
|---:|---:|---:|---:|---:|---:|
| 10 | 113.860294 | no | 18 | 0.655 | 426.1 |
| 50 | 112.353047 | no | 18 | 2.670 | 1441.0 |
| 100 | 110.470476 | no | 18 | 5.337 | 2709.7 |

Graph retention itself was feasible up to 100 unrolled steps on this small molecule, but the final second-order HVP is still NaN because the same full-edge coordinate second derivative path is used. This is not ready as a physical density-relaxed Hessian path.

## Implicit density-relaxed Hessian design

For a relaxed density/coefficient vector `rho*(R)` satisfying projected stationarity `E_rho(R, rho*) = 0`, the relaxed coordinate Hessian is:

```text
H_relaxed = E_RR - E_Rrho (E_rhorho)^(-1) E_rhoR
```

HVP form for a coordinate vector `v`:

```text
rhs = E_rhoR v
solve E_rhorho x = rhs
H_relaxed v = E_RR v - E_Rrho x
```

A practical implementation should use autograd HVP/JVP primitives plus conjugate gradient in coefficient space:

- compute `E_RR v` with coordinate HVP;
- compute `E_rhoR v` by differentiating coefficient gradient with respect to coordinates;
- solve the projected coefficient Hessian system with CG using `E_rhorho x` HVP;
- compute `E_Rrho x`;
- enforce the electron-number constraint/projected tangent space consistently.

Missing or unconfirmed interfaces:

- a differentiable scalar total OFDFT energy `E(R, rho)` with the same terms used in density optimization;
- second-order-safe geometry features, especially self-edge distance handling;
- coefficient-space Hessian-vector product with projection and constraints;
- stable CG solve, stopping criteria, and preconditioning;
- non-mutating density optimization or implicit solve wrapper;
- clear unit handling if mass-weighted Hessian/frequencies are added later.

## Conclusions

1. Fixed-density second-order autograd is not stable for the current full-edge Graphformer model. Full Hessian and HVP both produce all-NaN outputs on `0000010` and `0000062`.

2. The failure is localized. Removing self-loop edges only for diagnosis makes the second-order Hessian finite and closely consistent with finite difference. This points to `torch.norm(0)` on self-loop edges as the immediate blocker.

3. Autograd Hessian cannot currently replace finite-difference Hessian for formal P1-410 reporting. Current fixed-density and density-relaxed Hessian conclusions should continue to use finite differences.

4. HVP is still the right next primitive after fixing the zero-distance path, but it is not currently usable as-is.

5. Unrolled density optimization is computationally feasible for 100 toy coefficient-update steps on one small molecule, but it inherits the same NaN second-order coordinate path and is not a formal physical Hessian method.

6. Do not start implicit density-relaxed Hessian implementation until the model forward is made second-order-safe while preserving first-order energy/force behavior. A likely minimal fix is to special-case self-loop distances as constant zero, or otherwise remove self-loop distance dependence without changing non-self edge behavior. This needs a separate validation before replacing finite differences.

## Recommendation

For the next step, keep finite-difference Hessian as the official P1-410 evaluation path. In parallel, prototype a second-order-safe distance module:

- preserve current forward values for self edges;
- preserve first-order forces against the existing model within numerical tolerance;
- verify full Hessian and HVP finite on `0000010` and `0000062`;
- compare autograd-vs-FD Hessian on the same fixed-density setting before considering density-relaxed implicit Hessian.
