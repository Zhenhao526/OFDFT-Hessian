# QM9 P1-410 second-order autograd self-loop fix

Date: 2026-07-07

## Scope

This fixes the fixed-density second-order autograd NaN blocker caused by self-loop edge distances. It does not generate labels, delete data, add a force head, or change the official P1-410 Hessian conclusion path. Formal P1-410 Hessian evaluation remains finite difference until the autograd path is validated at larger scale.

## Code Changes

Distance computation is centralized in `mldft/ml/models/components/gbf_module.py`:

- added `_safe_edge_lengths(pos, edge_index)`;
- `GBFModule.forward()` now uses it;
- `GaussianLayer.forward()` now uses it.

The key behavior is:

- for non-self edges `i != j`, compute `torch.norm(pos_i - pos_j)` exactly as before;
- for self-loop edges `i == j`, return a constant zero distance without calling `torch.norm(0)`.

This preserves forward values:

- old self-loop length was `torch.norm(0) == 0`;
- new self-loop length is constant `0`;
- non-self edge lengths are unchanged.

It also preserves the intended one-force path:

- old self-loop first derivative from PyTorch was zero;
- new self-loop derivative is zero because the value is constant;
- non-self force derivatives are unchanged.

The difference is only in second order: old `torch.norm(0)` produced NaN second derivatives, while the new self-loop constant has finite zero second derivative.

## Tests

Added/updated:

- `tests/ml/test_gbf_module.py`
  - self-loop second-order autograd finite for `GBFModule`;
  - self-loop second-order autograd finite for `GaussianLayer`;
  - non-self edge outputs in full-edge graph match the filtered no-self graph exactly;
  - self-loop lengths are constant zero.
- `tests/ml/test_qm9_second_order_autograd_runtime.py`
  - opt-in runtime test using local P1-410 checkpoint;
  - skipped by default unless `MLDFT_RUN_RUNTIME_TESTS=1`;
  - verifies full-edge fixed-density Hessian is finite and close to FD.

Commands run:

```bash
.venv/bin/python -m pytest tests/ml/test_gbf_module.py -q
```

Result: `9 passed in 0.34s`.

```bash
.venv/bin/python -m pytest tests/ml/test_qm9_second_order_autograd_runtime.py -q
```

Result: `1 skipped in 0.03s` by default.

```bash
MLDFT_RUN_RUNTIME_TESTS=1 MLDFT_RUNTIME_DEVICE=cuda:0 CUDA_VISIBLE_DEVICES=4 \
  .venv/bin/python -m pytest tests/ml/test_qm9_second_order_autograd_runtime.py -q
```

Result: `1 passed in 23.68s`.

## Audit Run

Command shape:

```bash
DFT_DATA=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1 \
DFT_MODELS=/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p1_models \
CUDA_VISIBLE_DEVICES=4 \
  .venv/bin/python scripts/qm9_second_order_autograd_hessian_audit.py \
  --molecules 0000010,0000062 \
  --scf-iteration 1 \
  --run EG_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200=_runtime/qm9_p1_models/train/runs/qm9_p1_410_eg_resume_s3000_vci200/checkpoints/epoch_006.ckpt \
  --run EGF_lam1_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000=_runtime/qm9_p1_models/train/runs/qm9_p1_410_egf_lam1_resume_s3000/checkpoints/epoch_000.ckpt \
  --output-dir _runtime/qm9_p1_models/eval/qm9_p1_410_second_order_autograd_self_loop_fix \
  --output-json _runtime/qm9_p1_models/eval/qm9_p1_410_second_order_autograd_self_loop_fix/second_order_self_loop_fix_audit.json \
  --device cuda:0 \
  --no-run-unrolled
```

Artifacts:

- JSON: `_runtime/qm9_p1_models/eval/qm9_p1_410_second_order_autograd_self_loop_fix/second_order_self_loop_fix_audit.json`
- log: `_runtime/qm9_p1_models/eval/qm9_p1_410_second_order_autograd_self_loop_fix/second_order_self_loop_fix_audit.log`
- time: `_runtime/qm9_p1_models/eval/qm9_p1_410_second_order_autograd_self_loop_fix/second_order_self_loop_fix_audit_time.txt`
- Hessian npz files: `_runtime/qm9_p1_models/eval/qm9_p1_410_second_order_autograd_self_loop_fix/hessians/`

Resource:

- wall time: 34.44 s
- max RSS: 1.75 GB
- exit status: 0

## Full-Edge Fixed-Density Hessian

All full-edge fixed-density autograd Hessians are now finite.

| run | molecule | shape | autograd finite | autograd symmetry max abs | autograd-vs-FD MAE | RMSE | rel Fro | autograd-vs-PBE MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EG_s3000 | 0000010 | 18x18 | yes | 6.27e-7 | 3.11e-4 | 2.55e-3 | 6.80e-4 | 0.500493 |
| EG_s3000 | 0000062 | 24x24 | yes | 1.89e-7 | 3.31e-4 | 2.22e-3 | 5.19e-3 | 0.150044 |
| EGF_lam1_s3000 | 0000010 | 18x18 | yes | 7.56e-9 | 4.31e-6 | 1.29e-5 | 8.06e-5 | 0.011799 |
| EGF_lam1_s3000 | 0000062 | 24x24 | yes | 1.30e-8 | 3.02e-6 | 9.98e-6 | 9.49e-5 | 0.016382 |

Ranking against PBE is now consistent between autograd Hessian and FD Hessian:

- `0000010`: EGF_lam1_s3000 better by both autograd and FD;
- `0000062`: EGF_lam1_s3000 better by both autograd and FD.

## HVP

HVP was run only after full-edge fixed-density Hessian was finite. All tested HVP vectors are finite and agree with finite-difference directional response.

| run | molecule | direction | HVP finite | HVP-vs-FD MAE | RMSE | rel Fro |
|---|---:|---|---:|---:|---:|---:|
| EG_s3000 | 0000010 | random_unit | yes | 4.61e-4 | 1.34e-3 | 2.43e-4 |
| EG_s3000 | 0000010 | perturb_unit | yes | 7.95e-5 | 2.07e-4 | 5.81e-5 |
| EG_s3000 | 0000062 | random_unit | yes | 2.70e-4 | 4.95e-4 | 6.30e-4 |
| EG_s3000 | 0000062 | perturb_unit | yes | 3.50e-5 | 7.16e-5 | 3.11e-4 |
| EGF_lam1_s3000 | 0000010 | random_unit | yes | 6.65e-6 | 8.45e-6 | 3.88e-5 |
| EGF_lam1_s3000 | 0000010 | perturb_unit | yes | 7.66e-6 | 1.44e-5 | 7.94e-5 |
| EGF_lam1_s3000 | 0000062 | random_unit | yes | 1.15e-5 | 1.82e-5 | 1.13e-4 |
| EGF_lam1_s3000 | 0000062 | perturb_unit | yes | 4.86e-6 | 6.93e-6 | 1.26e-4 |

HVP is now a viable next primitive for fixed-density experiments. It should still not replace FD for official P1-410 conclusions until larger validation is done.

## No-Self-Loop Comparison

The no-self-loop diagnostic remains finite:

| run | molecule | edges original -> kept | autograd-vs-FD MAE | rel Fro |
|---|---:|---:|---:|---:|
| EG_s3000 | 0000010 | 36 -> 30 | 4.94e-4 | 8.79e-4 |
| EGF_lam1_s3000 | 0000010 | 36 -> 30 | 4.09e-6 | 8.13e-5 |

Full-edge safe and no-self-loop model outputs are not expected to be identical at model level: full-edge safe still includes self-loop messages with constant zero distance, while no-self-loop removes those messages entirely. The important invariants are:

- non-self edge features are unchanged;
- self-loop distance is zero and second-order safe;
- full-edge autograd Hessian agrees with full-edge FD Hessian.

## Conclusion

The self-loop NaN blocker is fixed for fixed-density second-order autograd Hessian and HVP on the tested P1-410 molecules.

Supported now:

- full-edge fixed-density Hessian from scalar energy autograd;
- fixed-density HVP from scalar energy autograd;
- EG/EGF ranking consistency with FD on `0000010` and `0000062`.

Still not changed:

- no new labels were generated;
- no existing labels/checkpoints/references were deleted;
- no force head was added;
- official P1-410 Hessian evaluation remains finite difference for now;
- unrolled density optimization Hessian was not advanced in this run.

Next recommended step: validate full-edge autograd Hessian/HVP on the existing 10-molecule or 20-molecule fixed-density reference set before considering implicit density-relaxed Hessian.
