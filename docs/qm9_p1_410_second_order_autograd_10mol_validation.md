# QM9 P1-410 Second-Order Autograd 10 Molecule Validation

Date: 2026-07-07

## Scope

This validates fixed-density full-edge second-order autograd Hessian and HVP after the self-loop distance fix.

Constraints kept:

- no new labels generated;
- no existing labels, checkpoints, or Hessian references deleted;
- no independent force head;
- Hessian and HVP are derived from scalar model energy autograd;
- density-relaxed Hessian conclusions still use finite difference as the official path.

This is a fixed-density validation, not a density-relaxed physical Hessian evaluation.

## Inputs

Molecules are the first 10 successful entries from the existing 20 molecule PBE Hessian reference manifest:

```text
0000010, 0000323, 0000109, 0000171, 0000062,
0000116, 0000144, 0000019, 0000052, 0000350
```

Models:

| model | checkpoint |
| --- | --- |
| EG_s3000 | `_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200/checkpoints/epoch_006.ckpt` |
| EGF_lam1_s3000 | `_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000/checkpoints/epoch_000.ckpt` |

Evaluator:

- `scripts/qm9_second_order_autograd_hessian_audit.py`

Output:

- `_runtime/qm9_p1_models/eval/qm9_p1_410_second_order_autograd_10mol_validation/second_order_autograd_10mol.json`
- `_runtime/qm9_p1_models/eval/qm9_p1_410_second_order_autograd_10mol_validation/second_order_autograd_10mol.log`
- `_runtime/qm9_p1_models/eval/qm9_p1_410_second_order_autograd_10mol_validation/second_order_autograd_10mol_time.txt`
- `_runtime/qm9_p1_models/eval/qm9_p1_410_second_order_autograd_10mol_validation/hessians/`

Resource:

| item | value |
| --- | ---: |
| wall time | 1:01.60 |
| max RSS | 1.75 GB |
| exit status | 0 |

## Command

```bash
OUT=_runtime/qm9_p1_models/eval/qm9_p1_410_second_order_autograd_10mol_validation
DFT_DATA=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1 \
DFT_MODELS=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models \
CUDA_VISIBLE_DEVICES=4 \
  .venv/bin/python scripts/qm9_second_order_autograd_hessian_audit.py \
  --molecules 0000010,0000323,0000109,0000171,0000062,0000116,0000144,0000019,0000052,0000350 \
  --scf-iteration 1 \
  --run EG_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200=_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200/checkpoints/epoch_006.ckpt \
  --run EGF_lam1_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000/checkpoints/epoch_000.ckpt \
  --output-dir "$OUT" \
  --output-json "$OUT/second_order_autograd_10mol.json" \
  --device cuda:0 \
  --no-run-unrolled
```

## Full-Edge Fixed-Density Hessian Stability

All full-edge autograd Hessians are finite:

| model | rows | success | finite | NaN rows |
| --- | ---: | ---: | ---: | ---: |
| EG_s3000 | 10 | 10 | 10 | 0 |
| EGF_lam1_s3000 | 10 | 10 | 10 | 0 |

Aggregate autograd-vs-FD Hessian comparison:

| model | MAE mean | RMSE mean | relative Fro mean | relative Fro max | autograd sym mean | autograd sym max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EG_s3000 | 2.801e-4 | 1.666e-3 | 1.137e-3 | 5.190e-3 | 4.250e-7 | 8.850e-7 |
| EGF_lam1_s3000 | 2.875e-6 | 8.730e-6 | 7.428e-5 | 1.187e-4 | 2.281e-8 | 4.948e-8 |

Timing:

| model | autograd Hessian sum | autograd Hessian mean | FD Hessian sum | FD Hessian mean | peak CUDA autograd | peak CUDA FD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EG_s3000 | 5.03 s | 0.503 s | 4.91 s | 0.491 s | 195.48 MB | 173.40 MB |
| EGF_lam1_s3000 | 2.32 s | 0.232 s | 3.66 s | 0.366 s | 194.17 MB | 172.09 MB |

## Per-Molecule Hessian Comparison

Autograd-vs-FD metrics:

| model | molecule | natoms | finite | MAE | RMSE | relative Fro | autograd symmetry | autograd s | FD s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EG_s3000 | 0000010 | 6 | yes | 3.108e-4 | 2.548e-3 | 6.800e-4 | 6.27e-7 | 0.799 | 0.958 |
| EG_s3000 | 0000323 | 6 | yes | 9.292e-4 | 3.153e-3 | 1.139e-3 | 7.90e-7 | 0.425 | 0.285 |
| EG_s3000 | 0000109 | 7 | yes | 1.186e-4 | 5.787e-4 | 2.339e-4 | 5.09e-7 | 0.803 | 0.721 |
| EG_s3000 | 0000171 | 7 | yes | 2.389e-5 | 1.243e-4 | 5.307e-4 | 3.04e-8 | 0.569 | 0.333 |
| EG_s3000 | 0000062 | 8 | yes | 3.306e-4 | 2.218e-3 | 5.190e-3 | 1.89e-7 | 0.378 | 0.537 |
| EG_s3000 | 0000116 | 8 | yes | 5.217e-4 | 4.370e-3 | 1.742e-3 | 5.68e-7 | 0.489 | 0.383 |
| EG_s3000 | 0000144 | 8 | yes | 1.031e-4 | 9.751e-4 | 3.819e-4 | 9.52e-8 | 0.241 | 0.384 |
| EG_s3000 | 0000019 | 9 | yes | 2.577e-4 | 1.603e-3 | 1.093e-3 | 8.85e-7 | 0.431 | 0.430 |
| EG_s3000 | 0000052 | 9 | yes | 5.633e-6 | 1.998e-5 | 9.513e-5 | 5.48e-8 | 0.616 | 0.436 |
| EG_s3000 | 0000350 | 9 | yes | 2.002e-4 | 1.075e-3 | 2.810e-4 | 5.03e-7 | 0.277 | 0.444 |
| EGF_lam1_s3000 | 0000010 | 6 | yes | 4.309e-6 | 1.287e-5 | 8.059e-5 | 7.56e-9 | 0.186 | 0.282 |
| EGF_lam1_s3000 | 0000323 | 6 | yes | 4.112e-6 | 1.164e-5 | 6.075e-5 | 2.86e-8 | 0.181 | 0.282 |
| EGF_lam1_s3000 | 0000109 | 7 | yes | 3.375e-6 | 1.201e-5 | 7.128e-5 | 1.89e-8 | 0.211 | 0.331 |
| EGF_lam1_s3000 | 0000171 | 7 | yes | 1.385e-6 | 3.375e-6 | 5.915e-5 | 7.41e-9 | 0.213 | 0.332 |
| EGF_lam1_s3000 | 0000062 | 8 | yes | 3.022e-6 | 9.977e-6 | 9.486e-5 | 1.30e-8 | 0.239 | 0.380 |
| EGF_lam1_s3000 | 0000116 | 8 | yes | 3.386e-6 | 1.175e-5 | 9.379e-5 | 1.73e-8 | 0.239 | 0.383 |
| EGF_lam1_s3000 | 0000144 | 8 | yes | 2.412e-6 | 6.031e-6 | 4.847e-5 | 3.83e-8 | 0.239 | 0.379 |
| EGF_lam1_s3000 | 0000019 | 9 | yes | 3.208e-6 | 9.238e-6 | 1.187e-4 | 3.22e-8 | 0.271 | 0.428 |
| EGF_lam1_s3000 | 0000052 | 9 | yes | 1.353e-6 | 4.035e-6 | 6.298e-5 | 1.53e-8 | 0.271 | 0.429 |
| EGF_lam1_s3000 | 0000350 | 9 | yes | 2.189e-6 | 6.369e-6 | 5.223e-5 | 4.95e-8 | 0.272 | 0.435 |

## EG/EGF Ranking Consistency

Ranking is defined by model Hessian vs PBE reference MAE, comparing EG_s3000 and EGF_lam1_s3000. Autograd and FD choose the same better model on every molecule.

| molecule | autograd best | FD best | consistent |
| ---: | --- | --- | ---: |
| 0000010 | EGF_lam1_s3000 | EGF_lam1_s3000 | yes |
| 0000323 | EGF_lam1_s3000 | EGF_lam1_s3000 | yes |
| 0000109 | EGF_lam1_s3000 | EGF_lam1_s3000 | yes |
| 0000171 | EGF_lam1_s3000 | EGF_lam1_s3000 | yes |
| 0000062 | EGF_lam1_s3000 | EGF_lam1_s3000 | yes |
| 0000116 | EGF_lam1_s3000 | EGF_lam1_s3000 | yes |
| 0000144 | EGF_lam1_s3000 | EGF_lam1_s3000 | yes |
| 0000019 | EGF_lam1_s3000 | EGF_lam1_s3000 | yes |
| 0000052 | EGF_lam1_s3000 | EGF_lam1_s3000 | yes |
| 0000350 | EGF_lam1_s3000 | EGF_lam1_s3000 | yes |

Conclusion: 10/10 ranking consistency.

## HVP Validation

HVP was run because full Hessian autograd was finite for all 10 molecules.

All HVP rows are finite:

| model | direction | rows | finite | MAE mean | RMSE mean | relative Fro mean | relative Fro max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EG_s3000 | random_unit | 10 | 10 | 1.755e-4 | 3.979e-4 | 4.180e-4 | 1.799e-3 |
| EG_s3000 | perturb_unit | 10 | 10 | 2.282e-4 | 4.295e-4 | 2.201e-4 | 5.519e-4 |
| EGF_lam1_s3000 | random_unit | 10 | 10 | 7.906e-6 | 1.182e-5 | 2.241e-4 | 8.127e-4 |
| EGF_lam1_s3000 | perturb_unit | 10 | 10 | 5.687e-6 | 1.012e-5 | 1.084e-4 | 2.471e-4 |

HVP remains numerically consistent with finite-difference directional response on the 10 molecule set.

## Interpretation

The fixed-density full-edge second-order autograd path is now stable on the 10 molecule reference set:

- no NaN/Inf in full Hessians;
- no NaN/Inf in HVPs;
- autograd Hessians match fixed-density FD Hessians closely;
- autograd and FD give the same EG/EGF ordering on all 10 molecules.

EGF lambda=1.0 remains better than EG by the PBE-reference Hessian comparison in both autograd and FD fixed-density evaluations.

## Recommendation

Because all 10 molecules are stable, it is reasonable to expand fixed-density autograd Hessian/HVP validation to the existing 20 molecule reference set.

Keep the following limits:

- do not replace the official density-relaxed FD Hessian conclusion yet;
- do not use this fixed-density validation as final P1/P2 evidence;
- use the 20 molecule expansion to decide whether HVP can become a routine proxy or a building block for an implicit density-relaxed Hessian design.
